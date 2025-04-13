# Real-World Federated Learning with Flower + Network Intrusion Detection Dataset (e.g., NSL-KDD)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import flwr as fl
import copy

# Load and preprocess NSL-KDD dataset (simplified)
def load_nsl_kdd(path):
    df = pd.read_csv(path)
    # Assume last column is label, do basic encoding
    X = pd.get_dummies(df.iloc[:, :-1])
    y = df.iloc[:, -1].astype('category').cat.codes
    return torch.tensor(X.values, dtype=torch.float32), torch.tensor(y.values, dtype=torch.long)

# Simple MLP model
class IntrusionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IntrusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Load and prepare data
X, y = load_nsl_kdd("dataset/kdd_train.csv")
dataset = TensorDataset(X, y)
num_clients = 5
client_datasets = random_split(dataset, [len(dataset)//num_clients]*num_clients)

# Server-side validation set
val_size = len(dataset) // 10
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=32)

# Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1):
            for x, y in self.train_loader:
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimizer.step()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.train_loader.dataset), {}

# Drift detection and fine-tuning

def detect_drift(old_model, new_model, threshold=0.1):
    diff = 0
    for old_param, new_param in zip(old_model.parameters(), new_model.parameters()):
        diff += torch.norm(old_param.data - new_param.data)
    return diff.item() > threshold

def fine_tune(model, val_loader, epochs=2):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for x, y in val_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

# Custom Flower Strategy
from flwr.server.strategy import FedAvg

class FineTuningStrategy(FedAvg):
    def __init__(self, model_fn, server_val_loader):
        super().__init__()
        self.previous_model = model_fn()
        self.server_val_loader = server_val_loader
        self.model_fn = model_fn

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, _ = super().aggregate_fit(rnd, results, failures)

        new_model = self.model_fn()
        state_dict = new_model.state_dict()
        for k, v in zip(state_dict.keys(), aggregated_parameters):
            state_dict[k] = torch.tensor(v)
        new_model.load_state_dict(state_dict)

        if detect_drift(self.previous_model, new_model):
            print(f"Round {rnd}: Drift detected. Fine-tuning...")
            fine_tune(new_model, self.server_val_loader)

        self.previous_model = copy.deepcopy(new_model)
        return [val.cpu().numpy() for val in new_model.state_dict().values()], {}




if __name__ == "__main__":
    # Load data and start server/client
    # Start Flower Server
    fl.server.start_server(strategy=FineTuningStrategy(lambda: IntrusionModel(X.shape[1], len(torch.unique(y))), val_loader))

    # Start Flower Clients
    for i in range(num_clients):
        model = IntrusionModel(X.shape[1], len(torch.unique(y)))
        train_loader = DataLoader(client_datasets[i], batch_size=32, shuffle=True)
        fl.client.start_numpy_client("localhost:8080", client=FlowerClient(model, train_loader))
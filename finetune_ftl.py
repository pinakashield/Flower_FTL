# Implementation 1: Custom Federated Learning with Fine-tuning Strategy (Simplified Example)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import numpy as np

# Sample model
def get_model():
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )

# Sample server-side validation data
x_val = torch.randn(100, 10)
y_val = torch.randint(0, 2, (100,))
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=16)

# Detect drift by comparing weight norms
def detect_drift(old_model, new_model, threshold=0.1):
    diff = 0
    for old_param, new_param in zip(old_model.parameters(), new_model.parameters()):
        diff += torch.norm(old_param.data - new_param.data)
    return diff.item() > threshold

# Fine-tuning step using server-side data
def fine_tune(model, val_loader, epochs=2):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for x_batch, y_batch in val_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

# Federated averaging

def aggregate_models(models):
    global_model = copy.deepcopy(models[0])
    for key in global_model.state_dict().keys():
        avg = torch.stack([m.state_dict()[key].float() for m in models], 0).mean(0)
        global_model.state_dict()[key].copy_(avg)
    return global_model

# Simulation: Assume 3 clients
client_models = [get_model() for _ in range(3)]

# Simulate local training (mock updates)
for model in client_models:
    for param in model.parameters():
        param.data += torch.randn_like(param) * 0.01

# Server-side aggregation
previous_global_model = get_model()
new_global_model = aggregate_models(client_models)

# Fine-tuning condition
if detect_drift(previous_global_model, new_global_model):
    print("Drift detected. Fine-tuning the global model...")
    fine_tune(new_global_model, val_loader)
else:
    print("No significant drift. Skipping fine-tuning.")


# Implementation 2: Flower (FL) Framework Integration with Fine-tuning

# Flower Client Implementation
import flwr as fl

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

# Server Strategy with Fine-tuning
from flwr.server.strategy import FedAvg

class FineTuningStrategy(FedAvg):
    def __init__(self, server_val_loader):
        super().__init__()
        self.previous_model = get_model()
        self.server_val_loader = server_val_loader

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, _ = super().aggregate_fit(rnd, results, failures)

        new_model = get_model()
        state_dict = new_model.state_dict()
        for k, v in zip(state_dict.keys(), aggregated_parameters):
            state_dict[k] = torch.tensor(v)
        new_model.load_state_dict(state_dict)

        # Check for drift and fine-tune
        if detect_drift(self.previous_model, new_model):
            print(f"Round {rnd}: Drift detected. Fine-tuning...")
            fine_tune(new_model, self.server_val_loader)

        self.previous_model = copy.deepcopy(new_model)
        return [val.cpu().numpy() for val in new_model.state_dict().values()], {}

if __name__ == "__main__":
# Start Flower server
    fl.server.start_server(strategy=FineTuningStrategy(server_val_loader=val_loader))

import flwr as fl
import torch
import numpy as np
from torch.utils.data import DataLoader
from model import IntrusionModel
from utils_cicids import load_dataset, get_dataloaders
import random
import time

# Flower client for federated learning for normal condition
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        self.model.load_state_dict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in self.train_loader:
            optimizer.zero_grad()
            outputs = self.model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(self.train_loader)

        print(f"Client Training - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return self.get_parameters(config), len(self.train_loader.dataset), {"loss": avg_loss, "accuracy": accuracy}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.train_loader.dataset), {}

# Base class for common functionality
class BaseClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        self.model.load_state_dict({
            k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)
        })

class DDoSClient(BaseClient):
    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Random spike delay to simulate flooding behavior and jitter
        delay = random.uniform(0.1, 0.5)
        print(f"[DDoSClient] Introducing artificial delay: {delay:.2f}s")
        time.sleep(delay)

        # Large abnormal weights to simulate payload spike
        updated_params = [100 * np.random.randn(*p.shape) for p in self.get_parameters(config)]
        print(f"[DDoSClient] Sending high-magnitude random updates.")

        return updated_params, len(self.train_loader.dataset), {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, parameters, config):
        return 0.0, len(self.train_loader.dataset), {}

class MITMClient(BaseClient):
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for x, y in self.train_loader:
            optimizer.zero_grad()
            loss = criterion(self.model(x), y)
            loss.backward()
            optimizer.step()

        poisoned_params = [-val.cpu().numpy() for val in self.model.state_dict().values()]
        print("[MITMClient] Sending poisoned/reversed weights.")
        return poisoned_params, len(self.train_loader.dataset), {"loss": 0.0, "accuracy": 0.0}

    def evaluate(self, parameters, config):
        return 0.0, len(self.train_loader.dataset), {}

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise ValueError("Client ID not provided. Usage: python client.py <client_id>")
    client_id = int(sys.argv[1])

    X, y = load_dataset("/Users/peenalgupta/PinakaShield/GitHub/Flower_FTL_IDS/dataset/CICIDS_2017.csv")
    client_loaders, _ = get_dataloaders(X, y, num_clients=2)

    if client_id < 0 or client_id >= len(client_loaders):
        raise IndexError(f"Client ID {client_id} is out of range. Must be 0 or 1.")

    input_dim = X.shape[1]
    num_classes = len(torch.unique(y))
    model = IntrusionModel(input_dim, num_classes)

    if client_id == 0:
        client = DDoSClient(model, client_loaders[client_id])
    else:
        client = FlowerClient(model, client_loaders[client_id])

    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

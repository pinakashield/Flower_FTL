import flwr as fl
import torch
import numpy as np
from torch.utils.data import DataLoader
from model import IntrusionModel
from utils_cicids import load_dataset, get_dataloaders
import random
import time
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Accessing path variables
LOG_PATH = os.getenv("LOG_PATH")
DATA_PATH = os.getenv("DATA_PATH")
DATASET_PATH = os.getenv("DATASET_PATH")
GRAPHS_PATH = os.getenv("GRAPHS_PATH")
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS")

# Accessing client parameters
CLIENT_NUM = int(os.getenv("CLIENT_NUM"))
CLIENT_NUM_IN_ROUND = int(os.getenv("CLIENT_NUM_IN_ROUND"))



# Flower client for federated learning with transfer learning
class FlowerTransferLearningClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, freeze_base=False):
        self.model = model
        self.train_loader = train_loader
        self.freeze_base = freeze_base

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Safely load parameters into model, handling shape mismatches"""
        try:
            new_state_dict = {}
            state_dict = self.model.state_dict()
            for (name, current_tensor), param_array in zip(state_dict.items(), parameters):
                try:
                    param_tensor = torch.tensor(param_array)
                    if param_tensor.shape == current_tensor.shape:
                        new_state_dict[name] = param_tensor
                    else:
                        print(f"Skipping parameter {name} due to shape mismatch: got {param_tensor.shape}, expected {current_tensor.shape}")
                        new_state_dict[name] = current_tensor
                except Exception as e:
                    print(f"Error loading parameter {name}: {str(e)}")
                    new_state_dict[name] = current_tensor
                
            self.model.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            print(f"Error setting parameters: {str(e)}")
            return False
        return True

    def fit(self, parameters, config):
        """Train the model on the locally held training set"""
        if not self.set_parameters(parameters):
            print("Warning: Using original model parameters due to loading error")
        self.model.train()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
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

        return self.get_parameters(config), len(self.train_loader.dataset), {"loss": avg_loss, "accuracy": accuracy, "ftl": self.freeze_base}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        # Personalized fine-tuning pass
        print("[Client] Performing final personalization fine-tune...")
        fine_tune(self.model, self.train_loader, epochs=2, unfreeze_all=True)

        return 0.0, len(self.train_loader.dataset), {}

class BaseClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Safely load parameters into model, handling shape mismatches"""
        try:
            new_state_dict = {}
            state_dict = self.model.state_dict()
            for (name, current_tensor), param_array in zip(state_dict.items(), parameters):
                try:
                    param_tensor = torch.tensor(param_array)
                    if param_tensor.shape == current_tensor.shape:
                        new_state_dict[name] = param_tensor
                    else:
                        print(f"Skipping parameter {name} due to shape mismatch: got {param_tensor.shape}, expected {current_tensor.shape}")
                        new_state_dict[name] = current_tensor
                except Exception as e:
                    print(f"Error loading parameter {name}: {str(e)}")
                    new_state_dict[name] = current_tensor
                
            self.model.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            print(f"Error setting parameters: {str(e)}")
            return False
        return True

class DDoSClient(BaseClient):
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        delay = random.uniform(0.1, 0.5)
        print(f"[DDoSClient] Introducing artificial delay: {delay:.2f}s")
        time.sleep(delay)
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

# Server-side utility for client personalization

def fine_tune(model, dataloader, epochs=2, unfreeze_all=True):
    if unfreeze_all:
        for param in model.parameters():
            param.requires_grad = True

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Client ID not provided. Usage: python client.py <client_id>")
    client_id = int(sys.argv[1])

    is_new_client = os.environ.get("IS_NEW_CLIENT", "false").lower() == "true"

    X, y = load_dataset(DATASET_PATH+"CICIDS_2017.csv") #CICIDS_2017.csv
    # Ensure num_classes matches the server's configuration
    num_classes = int(os.getenv("NUM_CLASSES", len(torch.unique(y))))  # Default to dataset's unique labels
    client_loaders, _, _ = get_dataloaders(X, y, num_clients=CLIENT_NUM)

    if client_id < 0 or client_id >= len(client_loaders):
        raise IndexError(f"Client ID {client_id} is out of range. Must be between 0 and {len(client_loaders)-1}.")

    input_dim = X.shape[1]
    model = IntrusionModel(input_dim, num_classes, freeze_base=is_new_client)

    if client_id == 0:
        client = DDoSClient(model, client_loaders[client_id])
    elif client_id == 1:
        client = MITMClient(model, client_loaders[client_id])
    else:
        client = FlowerTransferLearningClient(model, client_loaders[client_id], freeze_base=is_new_client)

    server_address = os.getenv("SERVER_ADDRESS", SERVER_ADDRESS)

    fl.client.start_numpy_client(server_address=server_address, client=client)
    print(f"Client {client_id} started.")
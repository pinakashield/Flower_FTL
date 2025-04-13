import flwr as fl
import torch
from torch.utils.data import DataLoader
from model import IntrusionModel
from utils import load_nsl_kdd, get_dataloaders

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

if __name__ == "__main__":
    import sys

    # Check if client_id is provided as a command-line argument
    if len(sys.argv) > 1:
        client_id = int(sys.argv[1])
    else:
        raise ValueError("Client ID not provided. Usage: python client.py <client_id>")

    # Load dataset and dataloaders
    X, y = load_nsl_kdd("/Users/peenalgupta/PinakaShield/GitHub/Flower_FTL_IDS/dataset/KDDTrain.csv")
    client_loaders, _ = get_dataloaders(X, y)

    #Ensure client_id is within range
    if client_id < 0 or client_id >= len(client_loaders):
        raise IndexError(f"Client ID {client_id} is out of range. Must be between 0 and {len(client_loaders) - 1}.")

    # Initialize model and client
    #client_id = 0  # Replace with the actual client ID
    input_dim = X.shape[1]
    num_classes = len(torch.unique(y))
    model = IntrusionModel(input_dim, num_classes)
    flower_client = FlowerClient(model, client_loaders[client_id])

    # Start the Flower client and connect to the server
    fl.client.start_numpy_client(server_address="212.227.61.42:8080", client=flower_client)

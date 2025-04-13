import flwr as fl
import torch
import copy
from model import IntrusionModel
from utils import load_nsl_kdd, get_dataloaders
from flwr.common import parameters_to_ndarrays

def detect_drift(old_model, new_model, threshold=0.1):
    diff = sum(torch.norm(p1 - p2) for p1, p2 in zip(old_model.parameters(), new_model.parameters()))
    return diff.item() > threshold

def fine_tune(model, val_loader, epochs=2):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in val_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
 
class FineTuningStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model_fn, val_loader):
        super().__init__(
            fit_metrics_aggregation_fn=self.aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=self.aggregate_evaluate_metrics,
        )
        self.previous_model = model_fn()
        self.model_fn = model_fn
        self.val_loader = val_loader

    @staticmethod
    def aggregate_fit_metrics(metrics):
        accuracies = [m["accuracy"] for m in metrics if "accuracy" in m]
        return {"accuracy": sum(accuracies) / len(accuracies)} if accuracies else {}

    @staticmethod
    def aggregate_evaluate_metrics(metrics):
        losses = [m["loss"] for m in metrics if "loss" in m]
        return {"loss": sum(losses) / len(losses)} if losses else {}

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, _ = super().aggregate_fit(rnd, results, failures)

        # Convert aggregated_parameters to a list of NumPy arrays
        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

        # Load the aggregated parameters into the model
        model = self.model_fn()
        model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), aggregated_ndarrays)})

        # Detect drift and fine-tune if necessary
        if detect_drift(self.previous_model, model):
            print(f"Round {rnd}: Drift detected. Fine-tuning...")
            fine_tune(model, self.val_loader)

            # Evaluate accuracy after fine-tuning
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in self.val_loader:
                    outputs = model(x)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == y).sum().item()
                    total += y.size(0)

            accuracy = 100 * correct / total
            print(f"Fine-Tuning Completed - Accuracy: {accuracy:.2f}%")

        # Update the previous model
        self.previous_model = copy.deepcopy(model)

        # Convert the model's state_dict back to Parameters and return
        return aggregated_parameters, {}

if __name__ == "__main__":
    X, y = load_nsl_kdd("/Users/peenalgupta/PinakaShield/GitHub/Flower_FTL_IDS/dataset/KDDTrain.csv")
    _, val_loader = get_dataloaders(X, y)
    input_dim = X.shape[1]
    num_classes = len(torch.unique(y))
    
    # Start the Flower server
    fl.server.start_server(
        server_address="212.227.61.42:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=FineTuningStrategy(lambda: IntrusionModel(input_dim, num_classes), val_loader),
    )

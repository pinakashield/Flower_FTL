import flwr as fl
import torch
import copy
import numpy as np
import csv
from datetime import datetime
from model import IntrusionModel
from utils_cicids import load_dataset, get_dataloaders
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitIns
from typing import List, Tuple, Dict

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

def compute_update_norm(parameters: List[np.ndarray]) -> float:
    return np.sqrt(sum(np.linalg.norm(p)**2 for p in parameters))

def cosine_similarity(params1: List[np.ndarray], params2: List[np.ndarray]) -> float:
    flat1 = np.concatenate([p.flatten() for p in params1])
    flat2 = np.concatenate([p.flatten() for p in params2])
    dot = np.dot(flat1, flat2)
    return dot / (np.linalg.norm(flat1) * np.linalg.norm(flat2) + 1e-10)

class FineTuningStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model_fn, val_loader):
        super().__init__(
            fit_metrics_aggregation_fn=self.aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=self.aggregate_evaluate_metrics,
        )
        self.previous_model = model_fn()
        self.model_fn = model_fn
        self.val_loader = val_loader
        self.previous_parameters = None
        self.quarantined_clients = set()
        self.log_file = "suspicious_clients_log.csv"

        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "round", "client_id", "update_norm", "cosine_similarity", "behavior"])

    @staticmethod
    def aggregate_fit_metrics(metrics):
        accuracies = [m["accuracy"] for m in metrics if "accuracy" in m]
        return {"accuracy": sum(accuracies) / len(accuracies)} if accuracies else {}

    @staticmethod
    def aggregate_evaluate_metrics(metrics):
        losses = [m["loss"] for m in metrics if "loss" in m]
        return {"loss": sum(losses) / len(losses)} if losses else {}

    def log_client(self, rnd, cid, norm, sim, label):
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.utcnow().isoformat(), rnd, cid, norm, sim, label])

    def mitigate_attack(self, client_id):
        print(f"🛡️ Mitigation in progress for client {client_id}...")
        with open("mitigation_actions.log", mode='a') as log_file:
            log_file.write(f"{datetime.utcnow().isoformat()} - Mitigated client {client_id}\n")

    # def configure_fit(self, server_round, parameters, client_manager, cid):
    #     config = {}
    #     fit_ins = []

    #     for client in client_manager.clients:
    #         blocked = client.cid in self.quarantined_clients
    #         client_config = {"blocked": blocked}
    #         if blocked:
    #             print(f"🚫 Notifying client {client.cid} they are blocked.")
    #         fit_ins.append((client, FitIns(parameters, client_config)))

    #     return fit_ins

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[fl.common.Parameters, Dict[str, fl.common.Scalar]]:

        print(f"\n[Round {rnd}] Client Update Monitoring:")
        filtered_results = []

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            if cid in self.quarantined_clients:
                print(f"❌ Skipping quarantined client {cid}")
                continue

            params = parameters_to_ndarrays(fit_res.parameters)
            update_norm = compute_update_norm(params)
            sim = cosine_similarity(params, self.previous_parameters) if self.previous_parameters else 1.0

            print(f" - {cid}: Update norm={update_norm:.2f}, CosSim={sim:.2f}")

            if update_norm > 100 or sim < -0.5:
                print(f"🚫 Quarantining client {cid} for suspicious behavior")
                self.quarantined_clients.add(cid)
                label = "DDoS" if update_norm > 100 else "MITM"
                self.log_client(rnd, cid, update_norm, sim, label)
                self.mitigate_attack(cid)
                continue

            self.log_client(rnd, cid, update_norm, sim, "Normal")
            filtered_results.append((client_proxy, fit_res))

            if self.previous_parameters is None:
                self.previous_parameters = params

        if not filtered_results:
            print("⚠️ No valid clients to aggregate. Skipping round.")
            return self.previous_model.state_dict(), {}

        aggregated_parameters, metrics = super().aggregate_fit(rnd, filtered_results, failures)
        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

        model = self.model_fn()
        model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), aggregated_ndarrays)})

        if detect_drift(self.previous_model, model):
            print(f"Round {rnd}: Drift detected. Fine-tuning...")
            fine_tune(model, self.val_loader)

        self.previous_model = copy.deepcopy(model)
        self.previous_parameters = aggregated_ndarrays

        return ndarrays_to_parameters(aggregated_ndarrays), metrics

if __name__ == "__main__":
    X, y = load_dataset("/Users/peenalgupta/PinakaShield/GitHub/Flower_FTL_IDS/dataset/CICIDS_2017.csv")
    _, val_loader = get_dataloaders(X, y, num_clients=2)
    input_dim = X.shape[1]
    num_classes = len(torch.unique(y))  # Ensure num_classes is calculated dynamically

    print("🔧 Pre-training global model before starting federated server...")
    model = IntrusionModel(input_dim, num_classes)  # Use consistent num_classes
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(3):
        for x, y_batch in val_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y_batch)
            loss.backward()
            optimizer.step()
    print("✅ Pre-training complete.")

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=FineTuningStrategy(lambda: copy.deepcopy(model), val_loader),
    )
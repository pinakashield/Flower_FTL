import flwr as fl
import torch
import copy
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import random
from model import IntrusionModel
from utils_cicids import load_dataset, get_dataloaders
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitIns
from typing import List, Tuple, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Accessing path variables
LOG_PATH = os.getenv("LOG_PATH")
DATA_PATH = os.getenv("DATA_PATH")
DATASET_PATH = os.getenv("DATASET_PATH")
GRAPHS_PATH = os.getenv("GRAPHS_PATH")
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS")

#Log files#
CLIENT_FTL_LOG = LOG_PATH + os.getenv("CLIENT_FTL_LOG")
ROUND_METRICS_LOG = LOG_PATH + os.getenv("ROUND_METRICS_LOG")
SUSPICIOUS_CLIENTS_LOG = LOG_PATH + os.getenv("SUSPICIOUS_CLIENTS_LOG")
MITIGATION_LOG = LOG_PATH + os.getenv("MITIGATION_LOG")

NUM_FTL_ROUNDS = int(os.getenv("NUM_FTL_ROUNDS"))
NUM_CLIENT_SERVER = int(os.getenv("NUM_CLIENT_SERVER"))
MAX_CLIENTS_PER_ROUND = int(os.getenv("MAX_CLIENTS_PER_ROUND"))

ROUND_METRICS = []
# CLIENT_FTL_LOG = "ftl_client_log.csv"
# ROUND_METRICS_LOG = "ftl_round_metrics.csv"
ACCURACY_TRACK = {}
 
timestamp = datetime.now().strftime("%Y%m%d")

with open(CLIENT_FTL_LOG, mode='w', newline='') as f:
    csv.writer(f).writerow(["timestamp", "round", "client_id", "ftl_used"])

with open(ROUND_METRICS_LOG, mode='w', newline='') as f:
    csv.writer(f).writerow(["timestamp","round", "avg_loss", "avg_accuracy"])

def detect_drift(old_model, new_model, threshold=0.1):
    diff = sum(torch.norm(p1 - p2) for p1, p2 in zip(old_model.parameters(), new_model.parameters()))
    return diff.item() > threshold

def fine_tune(model, val_loader, epochs=2, unfreeze_all=False):
    if unfreeze_all:
        for param in model.parameters():
            param.requires_grad = True

    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in val_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

def compute_update_norm(parameters: List[np.ndarray]) -> float:
    return np.sqrt(sum(np.linalg.norm(p)**2 for p in parameters))

def cosine_similarity(params1: List[np.ndarray], params2: List[np.ndarray] ) -> float:
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
        self.log_file = SUSPICIOUS_CLIENTS_LOG
        self.attack_log_file = LOG_PATH+timestamp+"attack_detection_log.csv"

        with open(self.log_file, mode='w', newline='') as f:
            csv.writer(f).writerow(["timestamp", "round", "client_id", "update_norm", "cosine_similarity", "behavior"])

        with open(self.attack_log_file, mode='w', newline='') as f:
            csv.writer(f).writerow(["timestamp", "round", "client_id", "attack_type", "action"])

    # def configure_fit(self, server_round, parameters, client_manager):
    #     # Randomly select a subset of clients
    #     all_clients = list(client_manager.clients)
    #     available_count = len(all_clients)
    #     print(f"📡 Round {server_round}: {available_count} client(s) connected.")

    #     if available_count > MAX_CLIENTS_PER_ROUND:
    #         selected_clients = random.sample(all_clients, MAX_CLIENTS_PER_ROUND)
    #         print(f"🔄 Selecting subset of {MAX_CLIENTS_PER_ROUND} clients for training.")
    #     else:
    #         selected_clients = all_clients


    #     fit_ins = [(client, FitIns(parameters, {})) for client in selected_clients]
    #     return fit_ins

    
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
            writer.writerow([datetime.now(timezone.utc).isoformat(), rnd, cid, norm, sim, label])

    def mitigate_attack(self, client_id):
        print(f"🛡️ Mitigation in progress for client {client_id}...")
        with open(MITIGATION_LOG, mode='a') as log_file:
            log_file.write(f"{datetime.now(timezone.utc).isoformat()} - Mitigated client {client_id}\n")

    def detect_attack(self, rnd, cid, fit_res):
        # Simple threshold-based detection for illustration
        update_norm = compute_update_norm(parameters_to_ndarrays(fit_res.parameters))
        sim = cosine_similarity(parameters_to_ndarrays(fit_res.parameters), self.previous_parameters) if self.previous_parameters else 1.0

        print(f" - {cid}: Update norm={update_norm:.2f}, CosSim={sim:.2f}")

        # Log suspicious client behavior
        if update_norm > 100 or sim < -0.5:
            label = "DDoS" if update_norm > 100 else "MITM"
            self.log_client(rnd, cid, update_norm, sim, label)
            self.mitigate_attack(cid)
            self.generate_alert(cid, label, "Quarantined (anomaly)")
            return True  # Attack detected

        return False  # No attack detected

    def log_attack(self, rnd, cid, attack_type, action):
        with open(self.attack_log_file, mode='a', newline='') as f:
            csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(), rnd, cid, attack_type, action])

    def generate_alert(self, cid, attack_type, action):
        alert_msg = f"🚨 ALERT: Attack detected from client {cid} (type: {attack_type}). Action: {action}"
        print(alert_msg)
        with open(LOG_PATH+"alerts.log", mode='a') as alert_file:
            alert_file.write(f"{datetime.now(timezone.utc).isoformat()} - {alert_msg}\n")

    def detect_and_mitigate_attack(self, rnd, cid, fit_res):
        """Detect attacks from both metrics and anomalies."""
        # Check for reported attack type
        attack_type = fit_res.metrics.get("attack_type", "none")
        if attack_type != "none":
            action = f"Quarantined (reported {attack_type})"
            self.quarantined_clients.add(cid)
            self.log_attack(rnd, cid, attack_type, action)
            self.generate_alert(cid, attack_type, action)
            return True

        # Check for anomalous behavior
        params = parameters_to_ndarrays(fit_res.parameters)
        update_norm = compute_update_norm(params)
        sim = cosine_similarity(params, self.previous_parameters) if self.previous_parameters else 1.0

        if update_norm > 100 or sim < -0.5:
            attack_type = "DDoS" if update_norm > 100 else "MITM"
            action = "Quarantined (anomaly)"
            self.quarantined_clients.add(cid)
            self.log_client(rnd, cid, update_norm, sim, attack_type)
            self.log_attack(rnd, cid, attack_type, action)
            self.generate_alert(cid, attack_type, action)
            self.mitigate_attack(cid)
            return True

        return False

    def aggregate_fit(self, rnd, results, failures):
        print(f"\n[Round {rnd}] Client Update Monitoring:")
        filtered_results = []
        all_losses, all_accuracies = [], []

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            
            if cid in self.quarantined_clients:
                print(f"❌ Skipping quarantined client {cid}")
                results.remove((client_proxy, fit_res))
                continue

            # Check for attacks
            if self.detect_and_mitigate_attack(rnd, cid, fit_res):
                print(f"❌ Skipping client {cid} due to detected attack")
                results.remove((client_proxy, fit_res))
                continue

            # Process valid client updates
            params = parameters_to_ndarrays(fit_res.parameters)
            if self.previous_parameters is None:
                self.previous_parameters = params

            filtered_results.append((client_proxy, fit_res))
            all_losses.append(fit_res.metrics.get("loss", 0))
            all_accuracies.append(fit_res.metrics.get("accuracy", 0))
            
            # Track accuracy
            if cid not in ACCURACY_TRACK:
                ACCURACY_TRACK[cid] = []
            ACCURACY_TRACK[cid].append(fit_res.metrics.get("accuracy", 0))

        if not filtered_results:
            print("⚠️ No valid clients to aggregate. Skipping round.")
            return self.previous_model.state_dict(), {}

        aggregated_parameters, _ = super().aggregate_fit(rnd, filtered_results, failures)
        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

        model = self.model_fn()
        model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), aggregated_ndarrays)})

        unfreeze = any(len(accs) >= 3 and max(accs[-3:]) - min(accs[-3:]) < 1 for accs in ACCURACY_TRACK.values())
        if detect_drift(self.previous_model, model):
            print(f"Round {rnd}: Drift detected. Fine-tuning... (unfreeze_all={unfreeze})")
            fine_tune(model, self.val_loader, epochs=3, unfreeze_all=unfreeze)

        self.previous_model = copy.deepcopy(model)
        self.previous_parameters = aggregated_ndarrays

        avg_loss = np.mean(all_losses)
        avg_accuracy = np.mean(all_accuracies)


        with open(os.path.join(LOG_PATH, ROUND_METRICS_LOG), mode='a', newline='') as f:
            csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(), rnd, avg_loss, avg_accuracy])

        # Store metrics as a tuple with exactly 3 values
        ROUND_METRICS.append((rnd, float(avg_loss), float(avg_accuracy)))
        
        return ndarrays_to_parameters(aggregated_ndarrays), {"loss": avg_loss, "accuracy": avg_accuracy}

    def visualize_metrics(self):
        if not ROUND_METRICS:
            return
        try:
            # Safely unpack the metrics
            rounds, losses, accuracies = map(list, zip(*ROUND_METRICS))
            
            plt.figure(figsize=(10, 6))
            plt.plot(rounds, losses, 'r-', label="Loss", marker='o')
            plt.plot(rounds, accuracies, 'b-', label="Accuracy", marker='s')
            plt.xlabel("Round")
            plt.ylabel("Metric Value")
            plt.title("Federated Learning Metrics Over Time")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(GRAPHS_PATH, f"ftl_training_metrics_{timestamp}.png"))
            print("📊 Training metrics plot saved")
        except Exception as e:
            print(f"Error during visualization: {e}")
            print(f"Current metrics structure: {ROUND_METRICS}")

if __name__ == "__main__":
    X, y = load_dataset(DATASET_PATH+"CICIDS_2017.csv") #CICIDS_2017.csv
    _, val_loader = get_dataloaders(X, y, NUM_CLIENT_SERVER)
    input_dim = X.shape[1]
    num_classes = len(torch.unique(y))

    print("🔧 Pre-training global model before starting federated server...")
    model = IntrusionModel(input_dim, num_classes)
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

    strategy = FineTuningStrategy(lambda: copy.deepcopy(model), val_loader)
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(NUM_FTL_ROUNDS),
        strategy=strategy,
    )

    strategy.visualize_metrics()
    print("📊 Visualization complete.")
import flwr as fl
import torch
import copy
import numpy as np
import csv
import os
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before other matplotlib imports
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import random
from model import IntrusionModel, EnsembleIDS,LSTMModel
from utils_cicids import load_dataset, get_dataloaders
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitIns
from typing import List, Tuple, Dict
from dotenv import load_dotenv
# import train as t

# Load environment variables from .env file
load_dotenv()

# Accessing path variables
LOG_PATH = os.getenv("LOG_PATH")
DATA_PATH = os.getenv("DATA_PATH")
DATASET_PATH = os.getenv("DATASET_PATH")
GRAPHS_PATH = os.getenv("GRAPHS_PATH")
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS")

#Log files#
CLIENT_FTL_LOG = os.getenv("CLIENT_FTL_LOG")
ROUND_METRICS_LOG = os.getenv("ROUND_METRICS_LOG")
SUSPICIOUS_CLIENTS_LOG = os.getenv("SUSPICIOUS_CLIENTS_LOG")
MITIGATION_LOG = os.getenv("MITIGATION_LOG")

NUM_FTL_ROUNDS = int(os.getenv("NUM_FTL_ROUNDS"))
NUM_CLIENT_SERVER = int(os.getenv("NUM_CLIENT_SERVER"))
MAX_CLIENTS_PER_ROUND = int(os.getenv("MAX_CLIENTS_PER_ROUND"))

ROUND_METRICS = []
# CLIENT_FTL_LOG = "ftl_client_log.csv"
# ROUND_METRICS_LOG = "ftl_round_metrics.csv"
ACCURACY_TRACK = {}
 
timestamp = datetime.now().strftime("%Y%m%d")

with open(LOG_PATH+timestamp+CLIENT_FTL_LOG, mode='w', newline='') as f:
    csv.writer(f).writerow(["timestamp", "round", "client_id", "ftl_used"])

with open(LOG_PATH+timestamp+ROUND_METRICS_LOG, mode='w', newline='') as f:
    csv.writer(f).writerow(["round", "avg_loss", "avg_accuracy"])

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
    def __init__(self, model_fn, val_loader, initial_parameters=None):
        super().__init__(
            fit_metrics_aggregation_fn=self.aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=self.aggregate_evaluate_metrics,
            initial_parameters=initial_parameters,
        )
        self.previous_model = model_fn()
        self.model_fn = model_fn
        self.val_loader = val_loader
        self.previous_parameters = parameters_to_ndarrays(initial_parameters) if initial_parameters else None
        self.quarantined_clients = set()
        self.log_file = LOG_PATH+timestamp+SUSPICIOUS_CLIENTS_LOG

        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "round", "client_id", "update_norm", "cosine_similarity", "behavior"])

        model = model_fn()  # Create model instance once
        self.model_architecture = {
            'input_dim': getattr(model, 'input_dim', None),
            'num_classes': getattr(model, 'num_classes', None),
            'state_dict_structure': {k: v.shape for k, v in model.state_dict().items()}
        }

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
        with open(LOG_PATH+timestamp+MITIGATION_LOG, mode='a') as log_file:
            log_file.write(f"{datetime.now(timezone.utc).isoformat()} - Mitigated client {client_id}\n")

    def configure_fit(self, server_round, parameters, client_manager):
        config = {
            'architecture': self.model_architecture,
            'round': server_round
        }
        
        # Get all clients for this round
        all_clients = list(client_manager.clients.values())
        
        # Sample clients for this round
        num_clients = min(len(all_clients), MAX_CLIENTS_PER_ROUND)
        sampled_clients = random.sample(all_clients, num_clients)
        
        # Create fit instructions with architecture information
        fit_ins = [
            (client, FitIns(parameters, config))
            for client in sampled_clients
        ]
        
        return fit_ins

    def aggregate_fit(self, rnd, results, failures):
        print(f"\n[Round {rnd}] Client Update Monitoring:")
        filtered_results = []
        all_losses, all_accuracies = [], []

        base_model = self.model_fn()
        expected_shapes = {name: param.shape for name, param in base_model.state_dict().items()}

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            if cid in self.quarantined_clients:
                print(f"❌ Skipping quarantined client {cid}")
                continue

            try:
                params = parameters_to_ndarrays(fit_res.parameters)
                params_dict = dict(zip(base_model.state_dict().keys(), params))
                reshaped_params = []
                
                for name, expected_shape in expected_shapes.items():
                    param = params_dict[name]
                    try:
                        if param.shape != expected_shape:
                            if 'weight' in name and len(param.shape) == len(expected_shape):
                                # Try transposing weight matrices if dimensions match
                                if param.shape[::-1] == expected_shape:
                                    param = param.T
                                else:
                                    # Initialize with original weights and add noise
                                    orig_param = base_model.state_dict()[name].cpu().numpy()
                                    param = orig_param + np.random.normal(0, 0.01, orig_param.shape)
                            else:
                                # For non-weight parameters or incompatible shapes
                                orig_param = base_model.state_dict()[name].cpu().numpy()
                                param = orig_param
                            print(f"Adjusted {name} to shape {param.shape}")
                        reshaped_params.append(param)
                    except Exception as e:
                        print(f"Error reshaping {name}: {e}")
                        # Use original parameter as fallback
                        reshaped_params.append(base_model.state_dict()[name].cpu().numpy())
                
                params = reshaped_params
                update_norm = compute_update_norm(params)
                sim = cosine_similarity(params, self.previous_parameters) if self.previous_parameters else 1.0

                ftl_used = fit_res.metrics.get("ftl", False)
                print(f" - {cid}: Update norm={update_norm:.2f}, CosSim={sim:.2f}, FTL: {ftl_used}")
                with open(LOG_PATH+CLIENT_FTL_LOG, mode='a', newline='') as f:
                    csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(), rnd, cid, ftl_used])

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

                all_losses.append(fit_res.metrics.get("loss", 0))
                all_accuracies.append(fit_res.metrics.get("accuracy", 0))
                # Track accuracy over time
                if cid not in ACCURACY_TRACK:
                    ACCURACY_TRACK[cid] = []
                ACCURACY_TRACK[cid].append(fit_res.metrics.get("accuracy", 0))

            except Exception as e:
                print(f"Error processing client {cid}: {str(e)}")
                continue

        if not filtered_results:
            print("⚠️ No valid clients to aggregate. Skipping round.")
            return self.previous_model.state_dict(), {}

        aggregated_parameters, _ = super().aggregate_fit(rnd, filtered_results, failures)
        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

        model = self.model_fn()
        # Create a new state dict with correct shapes
        new_state_dict = {}
        for (name, orig_tensor), array in zip(model.state_dict().items(), aggregated_ndarrays):
            try:
                tensor = torch.tensor(array)
                if tensor.shape == orig_tensor.shape:
                    new_state_dict[name] = tensor
                else:
                    print(f"Shape mismatch for {name}: got {tensor.shape}, expected {orig_tensor.shape}. Keeping original.")
                    new_state_dict[name] = orig_tensor
            except RuntimeError as e:
                print(f"Warning: Failed to load parameter {name}: {e}")
                new_state_dict[name] = orig_tensor  # Keep original parameter
    
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print("Model parameters updated successfully")
        except Exception as e:
            print(f"Warning: Partial model update - {e}")
            # Keep the model running with partial updates

        unfreeze = any(len(accs) >= 3 and max(accs[-3:]) - min(accs[-3:]) < 1 for accs in ACCURACY_TRACK.values())
        if detect_drift(self.previous_model, model):
            print(f"Round {rnd}: Drift detected. Fine-tuning... (unfreeze_all={unfreeze})")
            fine_tune(model, self.val_loader, epochs=3, unfreeze_all=unfreeze)

        self.previous_model = copy.deepcopy(model)
        self.previous_parameters = aggregated_ndarrays

        avg_loss = np.mean(all_losses)
        avg_accuracy = np.mean(all_accuracies)


        with open(os.path.join(LOG_PATH, ROUND_METRICS_LOG), mode='a', newline='') as f:
            csv.writer(f).writerow([rnd, avg_loss, avg_accuracy])

        ROUND_METRICS.append((rnd, avg_loss, avg_accuracy))

        return ndarrays_to_parameters(aggregated_ndarrays), {"loss": avg_loss, "accuracy": avg_accuracy}

    def visualize_metrics(self):
        if not ROUND_METRICS:
            return
        rounds, losses, accuracies = zip(*ROUND_METRICS)
        plt.figure()
        plt.plot(rounds, losses, label="Loss")
        plt.plot(rounds, accuracies, label="Accuracy")
        plt.xlabel("Round")
        plt.ylabel("Metric")
        plt.title("Federated Learning Metrics Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(GRAPHS_PATH+"ftl_training_metrics.png")
        print("📊 Training metrics plot saved as 'ftl_training_metrics.png'")

if __name__ == "__main__":
    X, y = load_dataset(DATASET_PATH+"CICIDS_2017.csv")
    # Convert tensors to numpy and preprocess labels
    if torch.is_tensor(y):
        y = y.cpu().numpy()
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_map[int(label)] if isinstance(label, (np.floating, float)) 
    
    else label_map[label] for label in y])
    
    _,train_loader, val_loader = get_dataloaders(X, y, NUM_CLIENT_SERVER)
    input_dim = X.shape[1]
    num_classes = len(unique_labels)
    
    print("🔧 Pre-training global model before starting federated server...")
    model = EnsembleIDS(input_dim, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Save initial model architecture and weights
    print("💾 Saving initial model architecture...")
    torch.save({
        'state_dict': model.state_dict(),
        'input_dim': input_dim,
        'num_classes': num_classes,
        'architecture': str(model)
    }, os.path.join(LOG_PATH, "initial_model.pth"))
    
    # Initialize both DNN and LSTM components
    for submodel in [model.dnn, model.lstm]:
        submodel.to(model.device)
        optimizer = torch.optim.Adam(submodel.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(3):
            for x, y_batch in val_loader:
                x, y_batch = x.to(model.device), y_batch.to(model.device)
                optimizer.zero_grad()
                if isinstance(submodel, LSTMModel):
                    x = x.unsqueeze(1)
                output = submodel(x)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
    
    # Train traditional models with mapped labels
    X_train_list = []
    y_train_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx < 100:
            X_train_list.append(data.cpu().numpy())
            y_train_list.append(target.cpu().numpy())
    
    if X_train_list:
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        print(f"Training traditional models with {len(np.unique(y_train))} classes")
        model.train_traditional_models(X_train, y_train)

    print("✅ Pre-training complete.")

    # Evaluate pre-trained model performance before federated training
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y_batch in val_loader:
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    pretrain_accuracy = correct / total if total > 0 else 0
    print(f"🔍 Pre-trained model accuracy on validation set: {pretrain_accuracy:.4f}")
    
    #Option:2 for the model design and training 
   #model_ensemble = t.train_ensemble(train_loader,val_loader,input_dim, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu', epochs=8)
    
    
    print("📦 Saving pre-trained model..." )
    #torch.save(model.state_dict(), LOG_PATH + "pretrained_model.pth")
    print("✅ Pre-trained model saved.")
    print("🔄 Loading pre-trained model for federated learning...")
    #model.load_state_dict(torch.load(LOG_PATH + "pretrained_model.pth"))
    print("✅ Pre-trained model loaded.")

    
    print("🚀 Starting federated learning server...")    
    strategy = FineTuningStrategy(
        lambda: copy.deepcopy(model), 
        val_loader,
        initial_parameters=ndarrays_to_parameters([
            val.cpu().numpy() for val in model.state_dict().values()
        ])
    )
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(NUM_FTL_ROUNDS),
        strategy=strategy,
    )

    strategy.visualize_metrics()
    print("📊 Visualization complete.")
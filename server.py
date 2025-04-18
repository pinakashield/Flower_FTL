import numpy as np
import io
import flwr as fl
import tensorflow as tf
import json
import os
import logging
from sklearn.metrics import f1_score  # Add this import

logging.basicConfig(level=logging.INFO)

# Load server pretrain info
if not os.path.exists("server_pretrain_info.json"):
    print("⚠️ Warning: server_pretrain_info.json not found! Creating a new one...")
    pretrain_info = {"num_features": 10, "num_classes": 3}
    with open("server_pretrain_info.json", "w") as f:
        json.dump(pretrain_info, f)
    print("✅ Created new server_pretrain_info.json.")

with open("server_pretrain_info.json", "r") as f:
    pretrain_info = json.load(f)

num_features = pretrain_info.get("num_features")
num_classes = pretrain_info.get("num_classes")
if num_features is None or num_classes is None:
    raise ValueError("❌ Missing required keys in server_pretrain_info.json!")

print(f"✅ Loaded pretrain info: {num_features} features, {num_classes} classes")

# Define server model
def create_server_model(num_features, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 1, activation='relu', input_shape=(num_features, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(128, 1, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

server_model = create_server_model(num_features, num_classes)
server_model.build((None, num_features, 1))

try:
    server_model.load_weights("pretrained_global_model.weights.h5")
    print("✅ Pretrained model weights loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load weights. Error: {e}")

# Deserialize client weights
def deserialize_weights(received_parameters):
    weights = []
    for tensor in received_parameters.tensors:
        tensor_stream = io.BytesIO(tensor)
        numpy_array = np.load(tensor_stream, allow_pickle=True)
        weights.append(numpy_array)
    return weights

# Define custom federated learning strategy
class FTLStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if failures:
            print(f"⚠️ Warning: {len(failures)} clients failed in round {server_round}")
        if not results:
            print("❌ No valid results received in this round.")
            return None

        client_weights = [deserialize_weights(res.parameters) for _, res in results]
        if not client_weights or any(w is None for w in client_weights):
            print("❌ Some clients returned invalid weights!")
            return None

        # Print received parameters from clients
        for client_idx, client_weight in enumerate(client_weights):
            print(f"🔹 Client {client_idx + 1} Parameters:")
            for layer_idx, layer_weights in enumerate(client_weight):
                print(f"  - Layer {layer_idx}: Shape {layer_weights.shape}, Mean {np.mean(layer_weights):.5f}")

        aggregated_weights = [
            np.mean([client[i] for client in client_weights], axis=0)
            for i in range(len(client_weights[0]))
        ]

        # Collect client metrics safely
        client_metrics = [res.metrics for _, res in results]

        # Extract loss and accuracy safely
        client_losses = [m.get("loss", np.nan) for m in client_metrics]  # Handle missing 'loss'
        client_accuracies = [m.get("accuracy", np.nan) for m in client_metrics]  # Handle missing 'accuracy'

        # Compute average, ignoring NaN values
        avg_client_loss = np.nanmean(client_losses)  # Use np.nanmean to avoid errors
        avg_client_acc = np.nanmean(client_accuracies)

        print(f"📊 Round {server_round}: Client Avg Loss: {avg_client_loss:.4f}, Client Avg Acc: {avg_client_acc:.4f}")

        # Evaluate the aggregated model on the server dataset
        server_model.set_weights(aggregated_weights)
        X_server_test = np.random.rand(100, num_features, 1)
        y_server_test = np.random.randint(0, num_classes, 100)
        server_loss, server_acc = server_model.evaluate(X_server_test, y_server_test, verbose=0)
        print(f"🏆 Round {server_round}: Server Loss: {server_loss:.4f}, Server Acc: {server_acc:.4f}")

        # Calculate F1 score
        y_pred = server_model.predict(X_server_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        f1 = f1_score(y_server_test, y_pred_classes, average='weighted')
        print(f"🔍 F1 Score: {f1}")
        logging.info(f"F1 Score: {f1}")

        return fl.common.ndarrays_to_parameters(aggregated_weights), {}

strategy = FTLStrategy(min_available_clients=2, min_fit_clients=2, min_evaluate_clients=2)
logging.info("🚀 Starting Flower federated learning server...")
fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=20), strategy=strategy)
print("✅ Federated Learning Server Started Successfully.")

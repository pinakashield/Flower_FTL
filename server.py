import numpy as np
import io
import flwr as fl
import tensorflow as tf
import json
import os

def deserialize_weights(received_parameters):
    """Convert received binary serialized weights into NumPy arrays"""
    weights = []
    
    # Iterate over each serialized tensor in parameters
    for tensor in received_parameters.tensors:
        tensor_stream = io.BytesIO(tensor)  # Convert binary data to a byte stream
        numpy_array = np.load(tensor_stream, allow_pickle=True)  # Deserialize to NumPy
        weights.append(numpy_array)
    
    return weights

# ---------------------------- STEP 1: LOAD SERVER INFO ---------------------------- #
# Ensure server_pretrain_info.json exists
if not os.path.exists("server_pretrain_info.json"):
    print("⚠️ Warning: server_pretrain_info.json not found! Creating a new one...")

    # Generate a default version (Update as needed)
    pretrain_info = {
        "num_features": 10,  # Default feature size
        "num_classes": 3
    }

    with open("server_pretrain_info.json", "w") as f:
        json.dump(pretrain_info, f)

    print("✅ Created new server_pretrain_info.json.")

# Load pretrain info
with open("server_pretrain_info.json", "r") as f:
    pretrain_info = json.load(f)

# Debugging: Check available keys
print("🔍 Available keys in pretrain_info:", pretrain_info.keys())

# Extract feature dimensions and number of classes
num_features = pretrain_info.get("num_features")
num_classes = pretrain_info.get("num_classes")

# Error handling
if num_features is None:
    raise ValueError("❌ Error: 'num_features' is missing in server_pretrain_info.json!")

if num_classes is None:
    raise ValueError("❌ Error: 'num_classes' is missing in server_pretrain_info.json!")

print(f"✅ Loaded pretrain info: {num_features} features, {num_classes} classes")

# ---------------------------- STEP 2: DEFINE SERVER MODEL ---------------------------- #
server_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(num_features,)),
    tf.keras.layers.Dense(32, activation='relu', name="shared_layer"),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

server_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Ensure model is built before loading weights
server_model.build((None, num_features))

# Load pretrained weights safely
try:
    server_model.load_weights("pretrained_global_model.weights.h5")
    print("✅ Pretrained model weights loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load weights. Error: {e}")

# Debugging: Print model weights before starting server
print(f"🔍 Server Model Initial Weights: {len(server_model.get_weights())} layers initialized.")

# ---------------------------- STEP 3: DEFINE FEDERATED TRANSFER LEARNING STRATEGY ---------------------------- #
class FTLStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate only the shared layer from clients' updates."""
        if failures:
            print(f"⚠️ Warning: {len(failures)} clients failed in round {server_round}")

        if not results:
            print("❌ No valid results received in this round.")
            return None  # No updates received

        # Deserialize received weights
        client_weights = [deserialize_weights(res.parameters) for _, res in results]

        # Ensure all clients returned valid weights
        if not client_weights or any(w is None for w in client_weights):
            print("❌ Some clients returned invalid weights!")
            return None

        num_layers = len(client_weights[0])

        # Aggregate weights layer-wise
        aggregated_weights = [np.mean([client[i] for client in client_weights], axis=0)
                              for i in range(num_layers)]

        return fl.common.ndarrays_to_parameters(aggregated_weights), {}

# ---------------------------- STEP 4: START FEDERATED LEARNING SERVER ---------------------------- #
strategy = FTLStrategy(
    min_available_clients=2,
    min_fit_clients=2,
    min_evaluate_clients=2,
)

fl.server.start_server(
    server_address="0.0.0.0:11080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)

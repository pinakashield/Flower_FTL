import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
import json

# Load shared feature metadata from the server
with open("server_pretrain_info.json", "r") as f:
    pretrain_info = json.load(f)
    num_classes = pretrain_info["num_classes"]
    shared_feature_dim = pretrain_info["num_features"]

# Load local dataset
df = pd.read_csv("/home/souvik-sengupta/vs_code/Tutorial_FTL/dataset/IoT_Fridge.csv")

# Debug: Print available columns
print(f"🔍 Available columns in dataset: {df.columns.tolist()}")

# Identify the correct label column
label_col = None
for possible_col in ["type", "label", "class", "category"]:
    if possible_col in df.columns:
        label_col = possible_col
        break

if label_col is None:
    raise ValueError("❌ No valid label column ('type', 'label', 'class', 'category') found in dataset!")

print(f"✅ Using label column: {label_col}")

# Extract labels
y = df[label_col].astype("category").cat.codes.values.astype(int)

# Drop non-numeric columns
df = df.select_dtypes(include=[np.number])

# Ensure no NaN values
X = df.fillna(0).astype(np.float32).values

# Debug: Print actual feature count
print(f"🔍 Client dataset shape: {X.shape}")
print(f"🔍 Expected feature dimension: {shared_feature_dim}")

# Ensure all clients have the same number of features
missing_features = shared_feature_dim - X.shape[1]
if missing_features > 0:
    print(f"⚠️ Client has {missing_features} missing features. Padding with zeros.")
    X = np.pad(X, ((0, 0), (0, missing_features)), 'constant')

elif missing_features < 0:
    print(f"⚠️ Client has {abs(missing_features)} extra features. Truncating.")
    X = X[:, :shared_feature_dim]  # Trim extra features

# Define client model
def create_client_model(input_dim, num_classes):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Initialize the model
model = create_client_model(shared_feature_dim, num_classes)

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Flower Client Definition
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
         return [w.astype(np.float32) for w in model.get_weights()]
        #return model.get_weights()

    def fit(self, parameters, config):
        print(f"🔍 Received {len(parameters)} weights from server.")
        if len(parameters) == 0:
            print("❌ Received empty weights! Skipping training.")
            return model.get_weights(), 0, {}

        parameters = [np.array(p, dtype=np.float32) for p in parameters]
        model.set_weights(parameters)

        model.fit(X, y, epochs=1, batch_size=32, verbose=0)
        return model.get_weights(), len(X), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X, y, verbose=0)
        return loss, len(X), {"accuracy": accuracy}

fl.client.start_client(
    server_address="0.0.0.0:11080",
    client=FlowerClient().to_client(),
)

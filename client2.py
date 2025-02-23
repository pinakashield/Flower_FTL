import flwr as fl
import keras
from keras import layers
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import logging
import client_model as clm

# Load shared feature metadata from the server
with open("server_pretrain_info.json", "r") as f:
    pretrain_info = json.load(f)
    pre_num_classes = pretrain_info["num_classes"]
    shared_feature_dim = pretrain_info["num_features"]

# Load the dataset
(X_train, y_train, X_test, y_test), num_classes = clm.load_data_from_csv()

# Ensure correct input shape for Conv1D
X_train = X_train.reshape(-1, shared_feature_dim, 1)
X_test = X_test.reshape(-1, shared_feature_dim, 1)

# Initialize the model
model = clm.load_model(num_classes, learning_rate=0.0001, X_train=X_train)

# Flower Client Definition
class FlowerClient(fl.client.NumPyClient):
    
    def get_parameters(self, config):
        return [w.astype(np.float32) for w in model.get_weights()]

    def fit(self, parameters, config):
        print(f"🔍 Received {len(parameters)} weights from server.")

        if len(parameters) == 0:
            print("❌ Received empty weights! Skipping training.")
            return model.get_weights(), 0, {}

        parameters = [np.array(p, dtype=np.float32) for p in parameters]
        if any(p is None for p in parameters):
            print("❌ ERROR: Received None in model parameters!")
            return model.get_weights(), 0, {}

        model.set_weights(parameters)
        print(f"✅ Model weights updated successfully.")

        # Fix: Unpack 3 returned values correctly
        model_weights, len_X, _ = clm.fit(
            model, parameters, X_train, y_train, epochs=10, batch_size=32, verbose=0
        )

        return model_weights, len_X, {}

    def evaluate(self, parameters, config):
        logging.info("Starting evaluation")
        model.set_weights(parameters)
        loss, accuracy = clm.evaluate(model, X_test, y_test)
        print(f"🔍 Loss: {loss}, Accuracy: {accuracy}")
        logging.info(f"Loss: {loss}, Accuracy: {accuracy}")
        
        return loss, len(X_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_client(
    server_address="0.0.0.0:8080",
    client=FlowerClient().to_client(),
)

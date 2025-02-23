import os
from pathlib import Path
import json
import keras
import keras.layers as layers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ------------------------------------------------------------------------------
# 1. Load and Preprocess Dataset
# ------------------------------------------------------------------------------
def load_data_from_csv():
    try:
        df = pd.read_csv("dataset/IoT_GPS_Tracker.csv")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

    if 'type' not in df.columns:
        raise ValueError("❌ 'type' column missing in dataset. Check dataset structure.")

    num_classes = df['type'].nunique()
    print(f"✅ Number of classes: {num_classes}")

    # Encode labels
    label_encoder = LabelEncoder()
    df['type_encoded'] = label_encoder.fit_transform(df['type'])

    # Handle missing features
    required_features = ['latitude', 'longitude']
    missing_features = [col for col in required_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"❌ Missing required features: {missing_features}")

    # Normalize features
    scaler = StandardScaler()
    df[required_features] = scaler.fit_transform(df[required_features])

    X = df[required_features]
    Y = df['type_encoded']

    # Ensure X has a valid shape for Conv1D
    if X.shape[1] < 2:
        raise ValueError("❌ Insufficient features after preprocessing.")

    X = X.values.reshape(X.shape[0], X.shape[1], 1)  # Reshape for Conv1D

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    print(f"✅ Data Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"✅ Data Shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")

    return (X_train, y_train, X_test, y_test), num_classes


# ------------------------------------------------------------------------------
# 2. Define Client Model
# ------------------------------------------------------------------------------
def load_model(num_classes, learning_rate, X_train):
    """
    Load and compile a CNN-based model for IoT intrusion detection.
    """
    input_shape = (X_train.shape[1], X_train.shape[2])  # Ensure correct input shape

    model = Sequential([
        Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(filters=128, kernel_size=1, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    return model


# ------------------------------------------------------------------------------
# 3. Train Model (Fix validation data issue)
# ------------------------------------------------------------------------------
def fit(model, parameters, X, y, epochs=1, batch_size=32, verbose=0):
    """
    Train the model with federated learning parameters.
    """
    model.set_weights(parameters)

    # Fix: Ensure validation data is formatted correctly
    X_train_split, X_val, y_train_split, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(
        X_train_split, y_train_split,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )

    return model.get_weights(), len(X_train_split), {}


# ------------------------------------------------------------------------------
# 4. Evaluate Model
# ------------------------------------------------------------------------------
def evaluate(model, X, y):
    """
    Evaluate the trained model and return loss and accuracy.
    """
    loss, accuracy = model.evaluate(X, y, verbose=1)
    return loss, accuracy


# ------------------------------------------------------------------------------
# 5. Main Execution for Standalone Training
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data_from_csv()
    
    if data is None:
        print("❌ Data loading failed. Exiting training.")
        exit()

    (X_train, y_train, X_test, y_test), num_classes = data
    model = load_model(num_classes, learning_rate=0.0001, X_train=X_train)
    
    # Train the model
    print("🚀 Training standalone model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"✅ Final Model Evaluation - Loss: {loss}, Accuracy: {accuracy}")
    
    # Save the model
    model.save("client_model.h5")
    print("✅ Model saved as client_model.h5")

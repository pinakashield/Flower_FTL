import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import json
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, Flatten, Dense

# Paths to your CSV files
WEATHER_PATH = "dataset/IoT_Weather.csv"
WEATHER_IN_PATH = "dataset/IoT_Weather(in).csv"

OUTPUT_WEIGHTS = "pretrained_global_model.weights.h5"
PRETRAIN_INFO = "server_pretrain_info.json"

# ------------------------------------------------------------------------------
# 1. Load CSV Data and Preprocess
# ------------------------------------------------------------------------------
def load_data_from_csv():
    try:
        df1 = pd.read_csv(WEATHER_PATH)
        df2 = pd.read_csv(WEATHER_IN_PATH)
        df = pd.concat([df1, df2], ignore_index=True)
    except Exception as e:
        print(f"❌ Error loading CSV files: {e}")
        return None

    if 'type' not in df.columns:
        raise ValueError("❌ 'type' column missing in dataset. Check dataset structure.")

    num_classes = df['type'].nunique()
    print(f"✅ Number of classes: {num_classes}")

    # Encode labels
    label_encoder = LabelEncoder()
    df['type_encoded'] = label_encoder.fit_transform(df['type'])

    # Fix missing feature columns
    required_features = ['pressure', 'humidity']
    missing_features = [col for col in required_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"❌ Missing required features: {missing_features}")

    # Encode 'humidity' if necessary
    if 'humidity' in df.columns:
        df['humidity_encoded'] = LabelEncoder().fit_transform(df['humidity'])
    else:
        df['humidity_encoded'] = 0

    # Select and normalize features
    feature_columns = ['pressure', 'humidity_encoded']
    X = df[feature_columns]
    Y = df['type_encoded']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ensure valid shape for Conv1D (samples, timesteps, features)
    if X_scaled.shape[1] < 2:
        raise ValueError("❌ Insufficient features after preprocessing.")

    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)  # Reshape for Conv1D

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size=0.2, random_state=42, stratify=Y
    )

    print(f"✅ Data Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"✅ Data Shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")

    return (X_train, y_train, X_test, y_test), num_classes

# ------------------------------------------------------------------------------
# 2. Model Definition
# ------------------------------------------------------------------------------
def create_base_model(input_shape, num_classes):
    print(f"✅ Creating model with input shape: {input_shape}")

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
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    return model

# ------------------------------------------------------------------------------
# 3. Pretrain the Global Model
# ------------------------------------------------------------------------------
def pretrain_global_model():
    data = load_data_from_csv()
    
    if data is None:
        print("❌ Data loading failed. Exiting pretraining.")
        return

    (X_train, y_train, X_test, y_test), num_classes = data
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)

    model = create_base_model(input_shape, num_classes)

    # Convert data into TensorFlow Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

    print("🚀 Starting pretraining of the global model...")
    model.fit(train_dataset, validation_data=val_dataset, epochs=10)

    # Save model weights
    model.save_weights(OUTPUT_WEIGHTS)
    print(f"✅ Pre-trained global model saved to {OUTPUT_WEIGHTS}")

    # Save metadata for clients
    metadata = {"num_features": X_train.shape[1], "num_classes": num_classes}
    with open(PRETRAIN_INFO, 'w') as f:
        json.dump(metadata, f)

    print(f"✅ Metadata saved to {PRETRAIN_INFO}")

# Run the pretraining process
if __name__ == "__main__":
    pretrain_global_model()

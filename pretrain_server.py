import os
import pandas as pd
import numpy as np
import tensorflow as tf
import json

# Paths to your CSV files
WEATHER_PATH = "/home/souvik-sengupta/vs_code/Tutorial_FTL/dataset/IoT_Weather.csv"
WEATHER_IN_PATH = "/home/souvik-sengupta/vs_code/Tutorial_FTL/dataset/IoT_Weather(in).csv"

# Where to save weights after pre-training
OUTPUT_WEIGHTS = "pretrained_global_model.weights.h5"  # Fixed the filename issue
PRETRAIN_INFO = "server_pretrain_info.json"

# ------------------------------------------------------------------------------
# 1. Utility: Load CSV as NumPy arrays
# ------------------------------------------------------------------------------
def load_data_from_csv(csv_path, label_column="type"):
    df = pd.read_csv(csv_path)

    # Drop non-numeric columns if they exist
    if "date" in df.columns:
        df.drop(columns=["date"], inplace=True)
    if "time" in df.columns:
        df.drop(columns=["time"], inplace=True)

    # Ensure the label column exists
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {csv_path}. Available columns: {df.columns}")

    # Convert categorical labels to integers
    df[label_column] = df[label_column].astype('category').cat.codes

    # Fill missing values (if any) with column mean
    df.fillna(df.mean(), inplace=True)

    # Ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == object:
            raise ValueError(f"Non-numeric column detected: {col}. Check data preprocessing.")

    X = df.drop(columns=[label_column]).values
    y = df[label_column].values

    return X, y

# ------------------------------------------------------------------------------
# 2. Model Definition for Tabular Data (MLP)
# ------------------------------------------------------------------------------
def create_base_model(input_dim, num_classes):
    inputs = tf.keras.Input(shape=(input_dim,))

    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# ------------------------------------------------------------------------------
# 3. Pre-train on WEATHER and WEATHER_IN
# ------------------------------------------------------------------------------
def pretrain_global_model():
    # Load the two CSVs
    X_w, y_w = load_data_from_csv(WEATHER_PATH)
    X_win, y_win = load_data_from_csv(WEATHER_IN_PATH)

    # Combine the features and labels
    X_combined = np.concatenate([X_w, X_win], axis=0)
    y_combined = np.concatenate([y_w, y_win], axis=0)

    # Ensure labels are integer
    y_combined = y_combined.astype(int)

    # Handle NaN values
    X_combined = np.nan_to_num(X_combined)

    # Determine number of classes
    num_classes = len(np.unique(y_combined))
    print("Detected number of classes:", num_classes)

    # Create the model
    input_dim = X_combined.shape[1]
    model = create_base_model(input_dim, num_classes)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fit (pre-train) the model
    model.fit(
        X_combined,
        y_combined,
        epochs=5,
        batch_size=32,
        validation_split=0.1
    )

    # Save the weights (Fixed filename issue)
    model.save_weights(OUTPUT_WEIGHTS)

    # Save model metadata for later use
    metadata = {
        "num_features": input_dim,
        "num_classes": num_classes
    }
    with open(PRETRAIN_INFO, 'w') as f:
        json.dump(metadata, f)

    print(f"Pre-trained global model saved to {OUTPUT_WEIGHTS}")
    print(f"Metadata saved to {PRETRAIN_INFO}")

if __name__ == "__main__":
    pretrain_global_model()

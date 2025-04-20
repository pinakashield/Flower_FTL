import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler


def load_dataset(file_path):
    # Updated to load CICIDS2017 CSV files
    df = pd.read_csv(file_path)

    # Calculate the length of the dataframe
    print(f"Length of the dataframe: {len(df)}")
    
    # Sample the dataframe to create a smaller dataset
    df=df.sample(n=654321,replace=True)
    
    # Rename columns for consistency and remove the spaces in the column names
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]

    # Drop columns not needed and clean up
    df.drop(columns=['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], errors='ignore', inplace=True)

    # Convert categorical to numeric if any
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Label':
            df[col] = pd.factorize(df[col])[0]

    # Replace infinity and drop NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Encode label
    df['Label'] = df['Label'].astype('category').cat.codes

    return preprocess(df)


def preprocess(df):
    y = torch.tensor(df['Label'].values, dtype=torch.long)
    X = df.drop(columns=['Label'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    return X_tensor, y



def get_dataloaders(X, y, num_clients=5, val_split=0.1):
    dataset = TensorDataset(X, y)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=32)

    base_size = len(train_dataset) // num_clients
    remainder = len(train_dataset) % num_clients
    client_split_sizes = [base_size + 1 if i < remainder else base_size for i in range(num_clients)]

    assert sum(client_split_sizes) == len(train_dataset), "Client splits do not match the dataset size!"

    client_splits = random_split(train_dataset, client_split_sizes)
    client_loaders = [DataLoader(split, batch_size=32, shuffle=True) for split in client_splits]

    return client_loaders, val_loader

if __name__ == "__main__":
    # Example usage
    X, y = load_dataset("/Users/peenalgupta/PinakaShield/GitHub/Flower_FTL_IDS/dataset/CICIDS_2017.csv")
    client_loaders, val_loader = get_dataloaders(X, y, num_clients=2)
    print(f"Client loaders: {len(client_loaders)}")
    print(f"Validation loader: {len(val_loader)}")
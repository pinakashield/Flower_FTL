import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
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
    labels = torch.tensor(df['Label'].values, dtype=torch.long)
    features = df.drop(columns=['Label'])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    return features_tensor, labels

class CICIDSDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def create_dataloaders(features, labels, num_clients=5, val_ratio=0.1):
    # Create custom dataset
    dataset = CICIDSDataset(features, labels)
    
    # Calculate splits
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    # Create splits
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create client splits
    client_sizes = [len(train_dataset) // num_clients] * num_clients
    client_sizes[-1] += len(train_dataset) % num_clients
    
    client_datasets = random_split(
        train_dataset, 
        client_sizes,
        generator=torch.Generator().manual_seed(42)
    )
    
    client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]
    
    return client_loaders, train_loader, val_loader

def get_dataloaders(features, labels, num_clients=5):
    return create_dataloaders(features, labels, num_clients)

if __name__ == "__main__":
    # Example usage
    X, y = load_dataset("/Users/peenalgupta/PinakaShield/GitHub/Flower_FTL_IDS/dataset/CICIDS_2017.csv")
    client_loaders, train_loader, val_loader = get_dataloaders(X, y, num_clients=2)
    print(f"Client loaders: {len(client_loaders)}")
    print(f"Training loader: {len(train_loader)}")
    print(f"Validation loader: {len(val_loader)}")
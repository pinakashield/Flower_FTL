import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import TensorDataset
import joblib  # For saving/loading scaler
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader


def load_nsl_kdd(file_path):
    # X, y = load_nsl_kdd("/Users/peenalgupta/PinakaShield/GitHub/Flower_FTL_IDS/dataset/kdd_train.csv")
    data_train = pd.read_csv(file_path,)
    
    # Define column names
    columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
            'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
            'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate',
            'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
            'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
            'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','outcome','level']
    data_train.columns = columns    
    
    #  Preprocess the data
    scaled_train = preprocess(data_train)
    
    #tensor
    X, y = train_test_split(scaled_train)
    
    return X, y
    
def create_tensors(df):
    # Encode the label column
    y = df['labels'].astype(str).astype('category').cat.codes

    # One-hot encode all features (automatically includes protocol_type, service, flag)
    X = df.drop(columns=['labels'])
    X = pd.get_dummies(X)

    # Ensure data is all numeric and float64
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)
    # Convert categorical columns to numeric
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    # Remove rows with NaN values
    X = X.dropna()
    y = y.dropna()
    # Remove rows with NaN values in y
    df = df.dropna()
    # Remove rows with NaN values in X
    # Remove rows with NaN values in y
    # Convert to PyTorch tensors
    return torch.tensor(X.values, dtype=torch.float32), torch.tensor(y.values, dtype=torch.long)

def get_dataloaders(X, y, num_clients=5, val_split=0.1):
    dataset = TensorDataset(X, y)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    # Split into training and validation datasets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Dynamically calculate client splits
    base_size = len(train_dataset) // num_clients
    remainder = len(train_dataset) % num_clients
    client_split_sizes = [base_size + 1 if i < remainder else base_size for i in range(num_clients)]

    # Ensure the sum of client_split_sizes matches the length of the training dataset
    assert sum(client_split_sizes) == len(train_dataset), "Client splits do not match the dataset size!"

    # Split the training dataset among clients
    client_splits = random_split(train_dataset, client_split_sizes)
    client_loaders = [DataLoader(split, batch_size=32, shuffle=True) for split in client_splits]

    return client_loaders, val_loader

def clean_dataset(df):
    # Define column names
    columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
            'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
            'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate',
            'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
            'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
            'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','outcome','level']
    df.columns = columns
    # Remove rows with missing values
    df = df.dropna()
    # Remove rows with invalid values (if any)
    df = df[df.applymap(lambda x: isinstance(x, (int, float)))]
    # Remove rows with invalid data types
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows with invalid labels (if any)
    valid_labels = set(range(0, 5))  # Assuming labels are integers from 0 to 4
    df = df[df['labels'].isin(valid_labels)]

    return df
#  Function to scale numerical features and save the scaler
def Scaling(df_num, cols, save_path="/Users/peenalgupta/PinakaShield/GitHub/Flower_FTL_IDS/scaler.pkl"):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df_num)
    scaled_df = pd.DataFrame(scaled_data, columns=cols)

    #  Save the trained scaler for inference
    joblib.dump(scaler, save_path)
    print(f" Scaler saved successfully at {save_path}")

    return scaled_df
#  Function to preprocess data
def preprocess(dataframe):
    cat_cols = ['is_host_login', 'protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_guest_login', 'level', 'outcome']
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.columns

    # Scale numerical features and save the scaler
    # The line `scaled_df = Scaling(df_num, num_cols)` appears to be calling a function named
    # `Scaling` with arguments `df_num` and `num_cols`. However, in the provided code snippet, there
    # is no definition or implementation of the `Scaling` function.
    scaled_df = Scaling(df_num, num_cols)

    # Replace numerical features with scaled values
    dataframe.drop(labels=num_cols, axis="columns", inplace=True)
    dataframe[num_cols] = scaled_df[num_cols]

    # Convert outcome to binary classification (0 = Normal, 1 = Attack)
    dataframe.loc[dataframe['outcome'] == "normal", "outcome"] = 0
    dataframe.loc[dataframe['outcome'] != 0, "outcome"] = 1

    # One-hot encode categorical features
    dataframe = pd.get_dummies(dataframe, columns=['protocol_type', 'service', 'flag'])

    return dataframe

def train_test_split(dataframe, test_size=0.2):
    # Split the dataset into training and testing sets
    # Prepare the training and test datasets
    x = dataframe.drop(['outcome', 'level'], axis=1).values  # Already a NumPy array
    y = dataframe['outcome'].values  # Already a NumPy array
    
    # Convert to numeric types
    x = x.astype(np.float32)  
    y = y.astype(np.int64)    

    # Handle NaN or infinite values
    x = np.nan_to_num(x)

    # Convert to PyTorch tensors (remove .values)
    X_tensor = torch.tensor(x, dtype=torch.float32)
    Y_tensor = torch.tensor(y, dtype=torch.long)

    return X_tensor, Y_tensor

#if __name__ == "__main__":

#  Get dataloaders
# # client_loaders, val_loader = get_dataloaders(X, y)
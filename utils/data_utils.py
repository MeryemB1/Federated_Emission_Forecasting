import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch


def apply_ema(df, features_to_ema=None, span=10):

    df = df.sort_values(['Root Detector ID', 'Minute'])

    if features_to_ema is None:
        features_to_ema = [
            'Sedan Count', 
            'Vans Count', 
            'SUV Count', 
            'Heavy_Truck Count',
            'Total Average Speed (km/h)', 
            'Total Emissions (g)'
        ]
    
    for feature in features_to_ema:
        ema_col_name = feature + '_EMA'
        df[ema_col_name] = df.groupby('Root Detector ID')[feature] \
                             .transform(lambda x: x.ewm(span=span, adjust=False).mean())
    
    return df


def split_train_test(df, time_col="Minute", train_ratio=0.8):
  
    unique_times = sorted(df[time_col].unique())
    cutoff_idx = int(train_ratio * len(unique_times))
    cutoff_time = unique_times[cutoff_idx]

    train_df = df[df[time_col] <= cutoff_time].copy()
    test_df = df[df[time_col] > cutoff_time].copy()

    print(f"✅ Train set: {train_df[time_col].min()} → {train_df[time_col].max()} ({len(train_df[time_col].unique())} unique {time_col}s)")
    print(f"✅ Test set: {test_df[time_col].min()} → {test_df[time_col].max()} ({len(test_df[time_col].unique())} unique {time_col}s)")
    
    return train_df, test_df

def normalize_features(train_df, test_df, cols_to_normalize):
   
    scaler = MinMaxScaler()
    scaler.fit(train_df[cols_to_normalize])

    train_df_norm = train_df.copy()
    test_df_norm = test_df.copy()

    train_df_norm[cols_to_normalize] = scaler.transform(train_df[cols_to_normalize])
    test_df_norm[cols_to_normalize] = scaler.transform(test_df[cols_to_normalize])

    print("✅ Normalization complete. Returning scaled dataframes and scaler.")
    return train_df_norm, test_df_norm, scaler



def build_tensors(train_df, L, features):
 
    # 1. Prepare time steps and nodes
    timesteps = sorted(train_df['Minute'].unique())
    line_nodes = list(L.nodes)

    T = len(timesteps)
    N = len(line_nodes)
    F = len(features)

    # 2. Create empty numpy arrays
    X = np.zeros((T, N, F), dtype=np.float32)

    # 3. Build mapping from node to index
    node_to_idx = {edge: idx for idx, edge in enumerate(line_nodes)}

    # 4. Fill X with feature values
    for _, row in train_df.iterrows():
        t = timesteps.index(row['Minute'])
        edge_str = row['Root Detector ID']
        try:
            u, v = map(int, edge_str.split('_'))
            if (u, v) in node_to_idx:
                n = node_to_idx[(u, v)]
                X[t, n] = row[features].values
        except ValueError:
            continue  # skip malformed detector IDs

    # 5. Build adjacency matrix
    adj_matrix = np.zeros((N, N), dtype=np.float32)
    for u, v in L.edges:
        i = node_to_idx[u]
        j = node_to_idx[v]
        adj_matrix[i, j] = 1.0  # directed edge

    # 6. Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

    return X, adj_matrix, node_to_idx, timesteps

def create_sequences( X ,seq_len =12,pred_len = 6):

    T, N, F_in = X.shape
    samples = []

    for t in range(T - seq_len - pred_len + 1):
        # ---------- input window ----------
        X_window = X[t : t + seq_len]              # (L, N, F_in)
        # ---------- target window (single feature) -----
        Y_window = X[t + seq_len : t + seq_len + pred_len ]  # (pred_len, N, F)

        samples.append((X_window, Y_window))

    X_seq = torch.stack([s[0] for s in samples], dim=0)  # (num_samples, seq_len, N, F)
    Y_seq = torch.stack([s[1] for s in samples], dim=0)  # (num_samples, pred_len, N, F)

    return X_seq, Y_seq
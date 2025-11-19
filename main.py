
import torch
import numpy as np
from config import config
from utils.data_utils import*
from utils.training_utils import*
import pandas as pd
import pickle
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Model.model import*


def main():
    
    df = pd.read_csv(r"C:\Users\km_ka\Desktop\Demo\Dataset\Meta_Data\datasets.csv")
    train_df, test_df = split_train_test(df, time_col="Minute", train_ratio=0.8)
    normalized_train_df, normalized_test_df, scaler = normalize_features(train_df, test_df, cols_to_normalize=['Sedan Count', 'Vans Count', 'SUV Count', 'Heavy_Truck Count','Total Average Speed (km/h)','Total Emissions (g)' ]) 
    with open(r"C:\Users\km_ka\Desktop\Demo\Dataset\Meta_Data\line_graph.pkl", "rb") as f:
        L = pickle.load(f)
    X,adj_matrix,node_to_idx,timesteps = build_tensors(normalized_train_df, L , features=['Sedan Count', 'Vans Count', 'SUV Count', 'Heavy_Truck Count','Total Average Speed (km/h)','Total Emissions (g)'])
    missing_ratio = config.training.missing_ratio
    node_mask, missing_indexes = generate_missing_nodes(adj_matrix, missing_ratio)
    print("___________________________________")
    print("Indexes of Missing Nodes:")
    print(missing_indexes)
    print("___________________________________")
    X_masked= apply_node_mask(X, node_mask)
    seq_len = config.training.seq_len
    pred_len = config.training.pred_len
    batch_size = config.training.batch_sz
    epochs = config.training.epochs
    learning_rate = config.training.lr
    epsilon = config.training.eps
    T= config.model.T
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_seq, Y_seq= create_sequences(X_masked,seq_len,pred_len)
    dataset = TensorDataset(X_seq, Y_seq)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    modele = SpatioTemporalAttentionDiffusionNet(
            in_features=X.shape[-1],
            hidden_dim=config.model.hidden_dim,
            forecast_len=pred_len,
            sequence_len=seq_len,
            num_nodes=X.shape[1],
            prejection=config.model.prejection,
            k_hop=config.model.k_hops,

        ).to(device)
    dynamic_masks = generate_dynamic_masks(loader, adj_matrix, missing_ratio=0.20, missing_indexes=missing_indexes, generate_missing_nodes=generate_missing_nodes)
    save_path = r"C:\Users\km_ka\Desktop\Demo\Checkpoints\saved_model.pth"
    modele = train_and_save_model(modele,loader,dynamic_masks,node_mask,adj_matrix,missing_indexes,save_path, lr=learning_rate,epochs=epochs,temperature=T,device="cpu")

if __name__ == "__main__":
    main()

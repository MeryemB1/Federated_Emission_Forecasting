import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
def load_subgraphs(load_dir, num_subgraphs=9):
   
    subgraphs_loaded = []

    for i in range(num_subgraphs):
        file_path = os.path.join(load_dir, f'subgraph_{i}.npz')
        if not os.path.exists(file_path):
            print(f"⚠ Skipping missing file: {file_path}")
            continue

        data = np.load(file_path, allow_pickle=True)
        Xq = data['X']
        Aq = data['adj']

        subgraphs_loaded.append((Xq, Aq))
        print(f"✔ Loaded subgraph {i}: X shape {Xq.shape}, A shape {Aq.shape}")

    return subgraphs_loaded


def get_shared_keys(model: torch.nn.Module):
    """
    Returns the list of state_dict keys that belong to the shared encoder.
    Assumes encoder module is called 'proj', 'temporal*', 'transition', 'diffusion', etc.
    We'll mark shared by prefix 'encoder.' by wrapping model below; but this
    helper finds keys that are NOT in the two heads `decoder` and `decoder_for`.
    """
    all_keys = list(model.state_dict().keys())
    # keys for heads:
    head_prefixes = ("decoder.", "decoder_for.")
    shared = [k for k in all_keys if not k.startswith(head_prefixes)]
    return shared

def numpy_from_state(model: torch.nn.Module, keys):
    sd = model.state_dict()
    return [sd[k].detach().cpu().numpy() for k in keys]

def load_into_state(model: torch.nn.Module, keys, params_np):
    sd = model.state_dict()
    for k, arr in zip(keys, params_np):
        t = torch.tensor(arr, dtype=sd[k].dtype, device=sd[k].device)
        sd[k] = t
    model.load_state_dict(sd)
class GatedGroupedTemporalConv(nn.Module):
    def __init__(self, hidden_dim, kernel_size=6, num_nodes=189):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(kernel_size, 1),
            groups=1
        )
        self.gate = nn.Conv2d(
            in_channels=hidden_dim ,
            out_channels=hidden_dim ,
            kernel_size=(kernel_size, 1),
            groups=1
        )
      #B, T, N, Fe
    def forward(self, x):
        B, T, N, H = x.shape
        x = x.permute(0,3,1,2)  # (B, H, T, N)
        x = F.pad(x, (0, 0, self.kernel_size - 1, 0))
        out = torch.tanh(self.conv(x)) * torch.sigmoid(self.gate(x))
        #.view(B, N, H, T).permute(0, 2, 3, 1)  # (B, N, T, H)

        out = out.permute(0,2,3,1)

        return out

class AttentionTransitionMatrix(nn.Module):
    def __init__(self, in_features, sequence_len, hidden_dim):
        super().__init__()
        self.att_q = nn.Linear(in_features * sequence_len, hidden_dim)
        self.att_k = nn.Linear(in_features * sequence_len, hidden_dim)

    def forward(self, X_seq, adj):
        # X_seq: (B, T, N, F)
        B, T, N, Fe = X_seq.shape
        X_temporal = X_seq.permute(0, 2, 1, 3).reshape(B, N, T * Fe)
        Q = self.att_q(X_temporal)
        K = self.att_k(X_temporal)

        d_k = Q.size(-1)
        raw_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        adj_with_loops = adj + torch.eye(N, device=adj.device, dtype=adj.dtype)
        # Optional: If binary, clamp to 0/1
        adj_with_loops = (adj_with_loops > 0).to(adj.dtype)
        # 2. Expand to batch size B
        adj_batch = adj_with_loops.unsqueeze(0).expand(B, N, N)
        # 3. Use in masking
        scores = raw_scores.masked_fill(adj_batch == 0, float('-inf'))
        P = F.softmax(scores, dim=-1)
        return P  # (B, N, N)

class BidirectionalDiffusion(nn.Module):
    def __init__(self, hidden_dim, k_hop):
        super().__init__()
        self.k_hop = k_hop
        self.hop_alpha_f = nn.Parameter(torch.zeros(k_hop))
        self.hop_alpha_b = nn.Parameter(torch.zeros(k_hop))
        self.output_proj_f = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj_b = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X_seq, P):
        # X_seq: (B, T, N, H), P: (B, N, N)
        B, T, N, H = X_seq.shape
        f_sum = X_seq.clone()
        b_sum = X_seq.clone()
        v_f = X_seq
        v_b = X_seq
        P = P.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, N, N)
        for k in range(1, self.k_hop + 1):
            v_b = torch.bmm(P, v_b.reshape(B*T, N, H)).reshape(B, T, N, H)
            b_sum += self.hop_alpha_b[k - 1] * v_b

            v_f = torch.bmm(P, v_f.reshape(B*T, N, H)).reshape(B, T, N, H)
            f_sum += self.hop_alpha_f[k - 1] * v_f

        Fwd = torch.tanh(self.output_proj_f(f_sum))
        Bwd = torch.tanh(self.output_proj_b(b_sum))
        return Fwd , Bwd

class STAD_Encoder(nn.Module):
    def __init__(self, in_features, hidden_dim, sequence_len, num_nodes, k_hop=3):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.sequence_len = sequence_len
        self.num_nodes = num_nodes

        # Increase input features by 1 to account for mask channel
        self.proj = nn.Linear(in_features + 1, hidden_dim)
        self.temporal1 = GatedGroupedTemporalConv(hidden_dim, kernel_size=5)
        self.transition = AttentionTransitionMatrix(hidden_dim, sequence_len, hidden_dim)
        self.diffusion = BidirectionalDiffusion(hidden_dim, k_hop)
        self.temporal_f = GatedGroupedTemporalConv(hidden_dim, kernel_size=5)
        self.temporal_b = GatedGroupedTemporalConv(hidden_dim, kernel_size=5)

        self.last_P = None  # for debugging/inspection

    def encode(self, X, adj, mask):  # returns (concat, fwd_out)
        # X: (B, T, N, Fe), mask: (B, T, N, Fe) or (B,T,N,1)
        B, T, N, Fe = X.shape

        # Ensure mask has a single channel if needed
        if mask.shape[-1] != 1:
            mask_channel = mask[..., :1]  # take one channel
        else:
            mask_channel = mask

        # Concatenate mask as extra feature
        X_aug = torch.cat([X, mask_channel.float()], dim=-1)  # (B, T, N, Fe+1)

        # Project to hidden dimension
        x = self.proj(X_aug)                    # (B, T, N, H)
        x = self.temporal1(x)                   # (B, T, N, H)

        # Diffusion / attention
        P = self.transition(x, adj)             # (B, N, N)
        self.last_P = P.detach().cpu()
        Fwd, Bwd = self.diffusion(x, P)         # (B, T, N, H)

        Fwd_out = self.temporal_f(Fwd)          # (B, T, N, H)
        Bwd_out = self.temporal_b(Bwd)          # (B, T, N, H)

        # reshape to (B, N, T*H) for heads
        Fwd_out = Fwd_out.permute(0, 2, 1, 3).reshape(B, N, T * self.hidden_dim)
        Bwd_out = Bwd_out.permute(0, 2, 1, 3).reshape(B, N, T * self.hidden_dim)
        concat = torch.cat([Fwd_out, Bwd_out], dim=-1)  # (B, N, 2*T*H)

        return concat, Fwd_out
    
class ImputationHead(nn.Module):
    def __init__(self, sequence_len, in_features, hidden_dim, projection_dim):
        super().__init__()
        self.sequence_len = sequence_len
        self.in_features = in_features
        self.net = nn.Sequential(
            nn.Linear(sequence_len * hidden_dim * 2, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, sequence_len * in_features),
            nn.ReLU(),
        )

    def forward(self, concat):  # concat: (B, N, 2*T*H)
        B, N, _ = concat.shape
        out = self.net(concat).view(B, N, self.sequence_len, self.in_features)
        return out.permute(0, 2, 1, 3)  # (B, T, N, F)

class ForecastHead(nn.Module):
    def __init__(self, forecast_len, in_features, hidden_dim, sequence_len, projection_dim):
        super().__init__()
        self.forecast_len = forecast_len
        self.in_features = in_features
        self.sequence_len = sequence_len
        self.net = nn.Sequential(
            nn.Linear(sequence_len * hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, forecast_len * in_features),
            nn.ReLU(),
        )

    def forward(self, fwd_out):  # fwd_out: (B, N, T*H)
        B, N, _ = fwd_out.shape
        out = self.net(fwd_out).view(B, N, self.forecast_len, self.in_features)
        return out.permute(0, 2, 1, 3)  # (B, F_len, N, F)
class FedRepModel(nn.Module):
    def __init__(self, in_features, hidden_dim, forecast_len, sequence_len, num_nodes, projection_dim, k_hop=3):
        super().__init__()
        self.encoder = STAD_Encoder(in_features, hidden_dim, sequence_len, num_nodes, k_hop=k_hop)
        self.head_impute = ImputationHead(sequence_len, in_features, hidden_dim, projection_dim)
        self.head_forecast = ForecastHead(forecast_len, in_features, hidden_dim, sequence_len, projection_dim)

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.sequence_len = sequence_len
        self.forecast_len = forecast_len

    def encode(self, X, adj ,mask):
        return self.encoder.encode(X, adj, mask)  # (concat, fwd_out)

    def forward(self, X, adj, mask):
        # Full pass (useful for quick local tests)
        concat, fwd_out = self.encode(X, adj, mask)
        imputed = self.head_impute(concat)
        forecast = self.head_forecast(fwd_out)
        return imputed, forecast
def get_shared_keys(model: nn.Module):
    # Everything under 'encoder.' is shared; heads are local
    return list(model.state_dict().keys())

def numpy_from_state(model: nn.Module, keys):
    sd = model.state_dict()
    return [sd[k].detach().cpu().numpy() for k in keys]

def load_into_state(model: nn.Module, keys, params_np):
    sd = model.state_dict()
    for k, arr in zip(keys, params_np):
        t = torch.tensor(arr, dtype=sd[k].dtype, device=sd[k].device)
        sd[k] = t
    model.load_state_dict(sd)

def generate_missing_nodes(adj_matrix, missing_ratio, exception=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    N = adj_matrix.shape[0]
    num_missing = int(round(missing_ratio * N))
    exception = set(exception or [])

    # Build candidate list (exclude exception nodes)
    candidates = list(set(range(N)) - exception)

    if num_missing > len(candidates):
        raise ValueError(f"Cannot select {num_missing} missing nodes from only {len(candidates)} available candidates.")

    # Randomly select missing nodes
    selected_missing = random.sample(candidates, num_missing)

    # Build mask: 1 = available, 0 = missing
    node_mask = torch.ones(N, dtype=torch.int)
    node_mask[selected_missing] = 0

    return node_mask, selected_missing

def apply_missing_nodes(subgraphs_loaded, generate_missing_nodes, missing_ratio=0.15):

    subgraphs_with_masks = []

    for i, (Xq, Aq) in enumerate(subgraphs_loaded):
        node_mask, missing_indexes = generate_missing_nodes(Aq, missing_ratio)
        subgraphs_with_masks.append((Xq, Aq, node_mask, missing_indexes))

        print(f"\n📊 Subgraph {i}:")
        print(f"Node Mask (0 = missing, 1 = available): {node_mask}")
        print(f"Indexes of Missing Nodes: {missing_indexes}")

    return subgraphs_with_masks
def create_sequences(
        X  ,
        seq_len =12,
        pred_len =6,

):

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

import torch
from torch.utils.data import TensorDataset, DataLoader

def build_dataloaders(
    subgraphs_with_masks,
    create_sequences,
    seq_len=8,
    pred_len=6,
    batch_sz=16,
    device=None
):
    """
    Build DataLoaders for all subgraphs with missing node masks applied.

    Args:
        subgraphs_with_masks (list): List of tuples (Xq, Aq, node_mask, missing_indexes).
        create_sequences (callable): Function to create input/output sequences.
        seq_len (int): Length of input sequence.
        pred_len (int): Length of prediction sequence.
        batch_sz (int): Batch size for DataLoader.
        device (torch.device or str): Device to store tensors (e.g., 'cuda' or 'cpu').

    Returns:
        list: List of DataLoaders (one per subgraph).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_loaders = []

    for i, (Xq, Aq, node_mask, missing_indexes) in enumerate(subgraphs_with_masks):
        # Convert numpy arrays to tensors
        X_tensor = torch.tensor(Xq, dtype=torch.float32).to(device)  # shape: (T, N, F)
        node_mask_tensor = torch.tensor(node_mask, dtype=torch.float32).to(device)  # shape: (N,)

        # Expand mask for broadcasting: (1, N, 1)
        expanded_mask = node_mask_tensor.view(1, -1, 1)
        X_masked = X_tensor * expanded_mask  # apply mask to input features

        # Build sequences
        X_seq, Y_seq = create_sequences(
            X_masked,
            seq_len=seq_len,
            pred_len=pred_len
        )

        # Dataset + DataLoader
        dataset = TensorDataset(X_seq, Y_seq)
        loader = DataLoader(dataset, batch_size=batch_sz, shuffle=False)
        all_loaders.append(loader)

        print(f"✔ Subgraph {i}: {len(dataset)} sequences, {len(loader)} batches")

    return all_loaders


    


import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils.data_utils import create_sequences
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def load_trained_model(checkpoint_path, map_location="cpu"):
    # 1. Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # 2. Extract saved variables
    missing_indexes = checkpoint.get("missing_indexes", None)
    node_mask = checkpoint.get("node_mask", None)
    last_P = checkpoint.get("last_P", None)

    print(f"✅ Checkpoint loaded successfully from {checkpoint_path}")
    return checkpoint, missing_indexes, node_mask, last_P


def apply_node_mask(X, node_mask):
    if node_mask.dim() == 1:
        expanded_mask = node_mask.view(1, -1, 1)  # shape: (1, N, 1)
    elif node_mask.dim() == 3:
        expanded_mask = node_mask
    else:
        raise ValueError(f"Unexpected node_mask shape: {node_mask.shape}. Expected (N,) or (1, N, 1).")

    X_masked = X * expanded_mask
    return X_masked



def generate_missing_nodes(adj_matrix, missing_ratio, exception=None, seed=None):
  
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    N = adj_matrix.shape[0]
    num_missing = int(round(missing_ratio * N))
    exception = set(exception or [])

    # --- Determine candidate nodes ---
    candidates = []
    for i in range(N):
        out_deg = (adj_matrix[i, :] != 0).sum().item()  # outgoing edges
        in_deg = (adj_matrix[:, i] != 0).sum().item()   # incoming edges
        if out_deg > 0 and in_deg > 0 and i not in exception:
            candidates.append(i)

    if num_missing > len(candidates):
        raise ValueError(f"Cannot select {num_missing} missing nodes from only {len(candidates)} valid candidates.")

    # --- Randomly select missing nodes ---
    selected_missing = random.sample(candidates, num_missing)

    # --- Build mask ---
    node_mask = torch.ones(N, dtype=torch.int)
    node_mask[selected_missing] = 0

    return node_mask, selected_missing


def generate_dynamic_masks(loader, adj_matrix, missing_ratio=0.20, missing_indexes=None, generate_missing_nodes=None):
  
    if generate_missing_nodes is None:
        raise ValueError("You must pass a `generate_missing_nodes` function.")

    dynamic_masks = []
    
    for xb, yb in loader:
        B, _, N, _ = xb.shape
        
        # Generate missing node mask and indexes for this batch
        mask, indexes = generate_missing_nodes(
            adj_matrix,
            missing_ratio=missing_ratio,
            exception=missing_indexes
        )  # mask shape: (N,), indexes: List[int]
        
        # Expand mask to match batch size
        mask = mask.unsqueeze(0).expand(B, -1)  # shape: (B, N)
        dynamic_masks.append(mask)

    return dynamic_masks


def train_and_save_model(
    model,
    loader,
    dynamic_masks,
    node_mask,
    adj_matrix,
    missing_indexes,
    save_path,
    lr=1e-3,
    epochs=50,
    temperature=2.0,
    device="cpu"
):
    """
    Train a multitask GNN model with DWA (Dynamic Weight Averaging)
    for imputation + forecasting, and save model + training state.

    """
    model = model.to(device)
    adj_matrix = adj_matrix.to(device)

    # Loss + optimizer
    loss_fn = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # DWA setup
    task_num = 2
    T= temperature
    loss_history = torch.zeros((epochs, task_num))
    lambdas = torch.ones(task_num) / task_num  # equal weight init

    for epoch in range(1, epochs + 1):
        running_loss, impute_loss, forecast_loss = 0.0, 0.0, 0.0
        model.train()

        for (xb, yb), bmask in zip(loader, dynamic_masks):
            xb, yb, bmask = xb.to(device), yb.to(device), bmask.to(device)
            b, t, n, fe = xb.shape
            mask= node_mask.unsqueeze(0).unsqueeze(1).unsqueeze(-1)
            mask_prep= mask.expand(b, t, n, fe)
            # prepare masked input
            bmask_expanded = bmask.unsqueeze(1).unsqueeze(-1).expand(b, t, n, fe)
            xb_masked = xb * bmask_expanded
            target = 1 - bmask_expanded
            model_mask = bmask_expanded * mask_prep
            optimizer.zero_grad()
            imputed, pred = model(xb_masked, adj_matrix, bmask_expanded)

            # imputation loss
            imputation_loss = (loss_fn(imputed, xb) * target).sum() / target.sum()

            # forecasting loss
            b,t,n,fe = pred.shape
            mask = mask.expand(b, t, n, fe)
            forecasting_loss = (loss_fn(pred, yb) * mask).sum() / mask.sum()

            # weighted total loss using DWA
            loss = lambdas[0] * imputation_loss + lambdas[1] * forecasting_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            impute_loss += imputation_loss.item()
            forecast_loss += forecasting_loss.item()

        # ────── Epoch losses ──────
        epoch_loss = running_loss / len(loader)
        epoch_impute_loss = impute_loss / len(loader)
        epoch_forecast_loss = forecast_loss / len(loader)

        # store per-task losses
        loss_history[epoch-1, 0] = epoch_impute_loss
        loss_history[epoch-1, 1] = epoch_forecast_loss

        # ────── Update lambdas with DWA ──────
        if epoch > 2:  # need at least 2 previous epochs
            w = []
            for i in range(task_num):
                r = loss_history[epoch-1, i] / loss_history[epoch-2, i]
                w.append(r)
            w = torch.tensor(w)
            lambdas = torch.softmax(w / T, dim=0).detach()

        print(f"Epoch {epoch:02d}/{epochs:02d} | total={epoch_loss:.5f} "
            f"imputation={epoch_impute_loss:.5f} forecast={epoch_forecast_loss:.5f} "
            f"lambdas={lambdas.tolist()}")

    # --- Save checkpoint ---
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "missing_indexes": missing_indexes,
        "node_mask": node_mask,
        "last_P": model.last_P.detach().cpu()
    }

    torch.save(checkpoint, save_path)
    print(f"✅ Model + states saved to {save_path}")

    return model  



def run_imputation_inference(model, X, seq_len, pred_len, node_mask, adj_matrix, 
                             batch_size, device, loss_fn):

    model.eval()
    model.to(device)
    adj_matrix = adj_matrix.to(device)

    # === Build sequences and dataloader ===
    X_seq, Y_seq = create_sequences(X, seq_len=seq_len, pred_len=pred_len)
    dataset = TensorDataset(X_seq, Y_seq)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_truepred = [], []
    all_trueimpute, all_imputes = [], []

    total_forecast_loss, total_impute_loss = 0.0, 0.0
    count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            b, t, n, fe = xb.shape

            # Expand node mask
            mask = node_mask.unsqueeze(0).unsqueeze(1).unsqueeze(-1).to(device)
            mask = mask.expand(b, t, n, fe)
            target = 1 - mask

            # Mask input
            xb_noisy = xb * mask

            # Forward pass
            impute, pred = model(xb_noisy, adj_matrix, mask)

            # Losses
            imputation_loss = (loss_fn(impute, xb) * target).sum() / target.sum()
            forecasting_loss = loss_fn(pred, yb).mean()

            total_forecast_loss += forecasting_loss.item()
            total_impute_loss += imputation_loss.item()
            count += 1

            # Collect outputs
            all_preds.append(pred.cpu().numpy())
            all_truepred.append(yb.cpu().numpy())
            all_trueimpute.append(xb.cpu().numpy())
            all_imputes.append(impute.cpu().numpy())

    # Aggregate losses
    avg_forecast_loss = total_forecast_loss / count
    avg_impute_loss = total_impute_loss / count

    print(f"✅ Imputation Loss: {avg_impute_loss:.5f}")

    # Concatenate results
    all_preds = np.concatenate(all_preds, axis=0)
    all_truepred = np.concatenate(all_truepred, axis=0)
    all_trueimpute = np.concatenate(all_trueimpute, axis=0)
    all_imputes = np.concatenate(all_imputes, axis=0)

    return {
        "forecast_loss": avg_forecast_loss,
        "imputation_loss": avg_impute_loss,
        "predictions": all_preds,
        "true_forecast": all_truepred,
        "true_inputs": all_trueimpute,
        "imputations": all_imputes
    }


def smape(y_true, y_pred):
    """ Symmetric Mean Absolute Percentage Error (SMAPE) """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100

def denormalize(data, scaler):
    """
    Denormalize a 4D array [B, T, N, F] using MinMaxScaler fitted on features.
    """
    B, T, N, F = data.shape
    data_denorm = data.copy()
    for f in range(F):
        min_val = scaler.data_min_[f]
        max_val = scaler.data_max_[f]
        data_denorm[:, :, :, f] = data[:, :, :, f] * (max_val - min_val) + min_val
    return data_denorm

def evaluate_emissions(
    y_true, 
    y_pred, 
    node_mask=None, 
    specific_nodes=None, 
    mode="forecast", 
    print_results=True
):

    # --- Extract emissions feature ---
    y_true_em = y_true[..., 5]
    y_pred_em = y_pred[..., 5]

    if node_mask is not None:
        mask_exp = np.expand_dims(np.expand_dims(node_mask, axis=0), axis=(0, -1))  # [1,1,N,1]
        mask_exp = np.broadcast_to(mask_exp, y_true.shape)  # [B,T,N,F]
        observed_mask = mask_exp[..., 5]  # emissions feature only
    else:
        observed_mask = np.ones_like(y_true_em)

    if mode == "forecast":
        # Forecasting: use observed nodes only
        final_mask = observed_mask
    elif mode == "impute":
        # Imputation: use missing nodes (1 - observed)
        final_mask = 1 - observed_mask
        if specific_nodes is not None:
            node_filter = np.zeros_like(final_mask)
            node_filter[..., specific_nodes] = 1
            final_mask = final_mask * node_filter
    else:
        raise ValueError("mode must be 'forecast' or 'impute'")

    # --- Flatten only selected values ---
    y_true_flat = y_true_em[final_mask == 1].reshape(-1)
    y_pred_flat = y_pred_em[final_mask == 1].reshape(-1)

    # --- Compute metrics ---
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)),
        "MAE": mean_absolute_error(y_true_flat, y_pred_flat),
        "SMAPE": smape(y_true_flat, y_pred_flat),
        "R2": r2_score(y_true_flat, y_pred_flat) if len(np.unique(y_true_flat)) > 1 else np.nan
    }

    if print_results:
        print(f"{'Mode':<12} {'RMSE':>10} {'MAE':>10} {'SMAPE(%)':>12} {'R2':>10}")
        print(f"{mode:<12} {metrics['RMSE']:10.5f} {metrics['MAE']:10.5f} "
              f"{metrics['SMAPE']:12.2f} {metrics['R2']:10.4f}")

    return metrics









import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.patches as mpatches
import networkx as nx
import pickle

def plot_ema(df, feature="Sedan Count", node_index=0):
    """
    Plot a feature vs. its EMA for a selected node.

    Args:
        df (pd.DataFrame): Dataframe with EMA columns already added.
        feature (str): Feature name to plot (e.g. "Sedan Count").
        node_index (int): Index of the node (from unique Root Detector IDs) to visualize.
    """
    ema_col = feature + "_EMA"
    if ema_col not in df.columns:
        raise ValueError(f"EMA column {ema_col} not found. Run apply_ema() first.")

    node = df['Root Detector ID'].unique()[node_index]
    sub = df[df['Root Detector ID'] == node]

    plt.figure(figsize=(10, 4))
    plt.plot(sub['Minute'], sub[feature], alpha=0.4, label='Original')
    plt.plot(sub['Minute'], sub[ema_col], color='red', label='EMA-smoothed')
    plt.legend()
    plt.xlabel("Minute")
    plt.ylabel(feature)
    plt.title(f"Node {node} - {feature} vs. EMA")
    plt.show()



def visualize_adj_vs_transition(adj_matrix, P, title_suffix=""):

    # Convert A to numpy
    if isinstance(adj_matrix, torch.Tensor):
        A = adj_matrix.cpu().numpy()
    else:
        A = adj_matrix

    # Load P if path is given
    if isinstance(P, str):
        P = torch.load(P)

    # Convert P to numpy
    if isinstance(P, torch.Tensor):
        P = P.cpu().numpy()

    # If P is time-dependent (T, N, N), select the last step
    if P.ndim == 3:
        P = P[-1]

    # Ensure shapes match
    assert A.shape == P.shape, f"Shape mismatch: A {A.shape}, P {P.shape}"

    # Determine shared color scale for fair comparison
    vmax = max(A.max(), P.max())
    vmin = min(A.min(), P.min())

    # Plot side-by-side heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(A, cmap="Blues", cbar=True, ax=axes[0], vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Adjacency Matrix (A) {title_suffix}")
    axes[0].set_xlabel("Nodes")
    axes[0].set_ylabel("Nodes")

    sns.heatmap(P, cmap="Reds", cbar=True, ax=axes[1], vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Learned Transition Matrix (P) {title_suffix}")
    axes[1].set_xlabel("Nodes")
    axes[1].set_ylabel("Nodes")

    plt.tight_layout()
    plt.show()

def plot_node_inference_curves(
    all_trueimpute,
    all_imputes,
    concerned_nodes,
    feature_index=5,
    save_dir="full_prediction",
    seq_len=None,
    timesteps=None
):
    """
    Plot ground truth vs predictions for each horizon separately.
    Predictions are shifted so each horizon aligns with the true target position.
    """

    os.makedirs(save_dir, exist_ok=True)

    num_samples, T_out, num_nodes, num_features = all_imputes.shape
    if seq_len is None:
        seq_len = all_trueimpute.shape[1]

    total_len = num_samples + seq_len - 1  # only count inputs (no future extension)
    time_idx = np.arange(total_len)

    def idx_to_time(idx_arr):
        if timesteps is None:
            return idx_arr
        max_idx = len(timesteps) - 1
        idx_clipped = np.clip(idx_arr, 0, max_idx)
        return np.array([timesteps[i] for i in idx_clipped])

    for node in concerned_nodes:
        # --- True values: last input for each sliding window ---
        true_vals = all_trueimpute[:, -1, node, feature_index]  # shape: (num_samples,)
        true_timeline = np.full(total_len, np.nan)
        for s in range(num_samples):
            t = s + seq_len - 1
            true_timeline[t] = true_vals[s]

        # --- Predictions: store per horizon ---
        horizon_timelines = [np.full(total_len, np.nan) for _ in range(T_out)]

        for s in range(num_samples):
            for h in range(T_out):
                t = s + h  # <-- shifted back to align with ground truth position
                if t < total_len:
                    horizon_timelines[h][t] = all_imputes[s, h, node, feature_index]

        # --- Plot ---
        plt.figure(figsize=(14, 6))
        plt.plot(idx_to_time(time_idx), true_timeline, color="black", label="True", linewidth=2)

        colors = plt.cm.viridis(np.linspace(0, 1, T_out))  # distinct colors per horizon
        for h, timeline in enumerate(horizon_timelines):
            plt.plot(idx_to_time(time_idx), timeline,
                     linestyle="--", color=colors[h],
                     label=f"Pred horizon {h+1}")

        plt.title(f"Node {node} – Feature {feature_index} (Separate Horizons)")
        plt.xlabel("Time index" if timesteps is None else "Time")
        plt.ylabel("Feature value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(save_dir, f"node_{node}_feature_{feature_index}_per_horizon.png")
        plt.savefig(fname)
        plt.show()




def get_line_graph_positions(line_graph, metadata_path="/content/drive/MyDrive/manhattan/datasets/Meta_Data/G_custom.pkl"):
   
    # Load the original graph
    with open(metadata_path, "rb") as f:
        G_custom = pickle.load(f)

    line_graph_positions = {}
    for u, v in line_graph.nodes():
        # Get positions of the endpoints in the original graph
        x1, y1 = G_custom.nodes[u]['x'], G_custom.nodes[u]['y']
        x2, y2 = G_custom.nodes[v]['x'], G_custom.nodes[v]['y']

        # Midpoint for edge (u, v)
        line_graph_positions[(u, v)] = ((x1 + x2) / 2, (y1 + y2) / 2)

    return line_graph_positions


def draw_line_graph_with_khop(
    L,                   # line graph
    pos,                 # dict (u,v) -> (x,y)
    node_to_idx,         # map (u,v) -> int
    target_node,         # the node we want to highlight (in line-graph space!)
    k=1,                 # hop distance
    figsize=(12, 12)
):
    """
    Colours:
        • red      = target node
        • orange   = k-hop neighbors
        • skyblue  = all other nodes
    """

    # --- compute k-hop neighbors in line graph ---
    khop_neighbors = nx.single_source_shortest_path_length(L, target_node, cutoff=k)
    khop_neighbors = set(khop_neighbors.keys()) - {target_node}  # remove target itself

    # --- assign colors ---
    color_map = []
    for edge in L.nodes():
        if edge == target_node:
            color_map.append('red')
        elif edge in khop_neighbors:
            color_map.append('orange')
        else:
            color_map.append('skyblue')

    # --- numeric labels ---
    numeric_labels = {edge: str(node_to_idx[edge]) for edge in L.nodes()}

    # --- draw ---
    plt.figure(figsize=figsize)
    nx.draw(
        L,
        pos           = pos,
        node_color    = color_map,
        node_size     = 300,
        edge_color    = 'gray',
        arrows        = True,
        with_labels   = False,
    )
    nx.draw_networkx_labels(L, pos, labels=numeric_labels, font_size=8)

    # optional: edge weights
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in L.edges(data=True) if 'weight' in d}
    if edge_labels:
        nx.draw_networkx_edge_labels(L, pos, edge_labels=edge_labels, font_size=8)

    # --- legend ---
    plt.legend(handles=[
        mpatches.Patch(color='red',    label='Target node'),
        mpatches.Patch(color='orange', label=f'{k}-hop neighbors'),
        mpatches.Patch(color='skyblue',label='Other nodes')
    ])

    plt.title(f"Line graph – Node {node_to_idx[target_node]} with {k}-hop neighbors")
    plt.axis('off')
    plt.show()

def plot_median_error_per_horizon(all_preds, all_truepred, feature_idx=None, save_path=None, show=True):
   
    T_out = all_preds.shape[1]
    errors_per_horizon = []

    for t in range(T_out):
        # Select horizon slice
        preds_t = all_preds[:, t, :, :]
        true_t  = all_truepred[:, t, :, :]

        # If feature_idx specified, select only that feature(s)
        if feature_idx is not None:
            preds_t = preds_t[..., feature_idx]
            true_t  = true_t[..., feature_idx]

        # Flatten & compute absolute error
        mae_t = np.abs(preds_t.reshape(-1) - true_t.reshape(-1))
        errors_per_horizon.append(mae_t)

    # Compute median & variability (IQR)
    median_errors = [np.median(e) for e in errors_per_horizon]
    iqr = [np.percentile(e, 75) - np.percentile(e, 25) for e in errors_per_horizon]

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(1, T_out + 1),
                   median_errors,
                   yerr=iqr,
                   capsize=5,
                   color=plt.cm.viridis(np.linspace(0.2, 0.8, T_out)),
                   alpha=0.85,
                   edgecolor="black")

    plt.xlabel("Forecast Horizon", fontsize=12)
    plt.ylabel("Median Absolute Error", fontsize=12)
    plt.title("Median Error per Forecast Horizon", fontsize=14, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate bars with values
    for bar, val in zip(bars, median_errors):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{val:.2f}",
                 ha="center",
                 va="bottom",
                 fontsize=10)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def plot_rmse_per_node(all_preds, all_truepred, node_mask=None, title="Forecasting Error (RMSE) per Node"):
 
    all_preds = np.array(all_preds)
    all_truepred = np.array(all_truepred)
    
    # Compute RMSE per node
    squared_errors = (all_preds - all_truepred) ** 2
    mse_per_node = squared_errors.mean(axis=(0, 1, 3))  # Mean over samples, horizons, features
    rmse_per_node = np.sqrt(mse_per_node)  # shape [N]
    
    N = rmse_per_node.shape[0]
    plt.figure(figsize=(14, 5))

    if node_mask is not None:
        node_mask = np.array(node_mask).astype(bool)
        missing_nodes = ~node_mask  # True where missing
        for i in range(N):
            if missing_nodes[i]:
                plt.bar(i, rmse_per_node[i], color='#9775fa', 
                        label="Missing node" if i == np.where(missing_nodes)[0][0] else "")
            else:
                plt.bar(i, rmse_per_node[i], color='#fcc2d7', 
                        label="Observed node" if i == np.where(node_mask)[0][0] else "")
    else:
        plt.bar(range(N), rmse_per_node, color='#91a7ff')

    plt.xlabel("Node Index")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return rmse_per_node



def plot_task_losses(fixed_lambda_path, dwa_path, confidence_path=None, figsize=(16, 6)):
    """
    Plot imputation/forecast losses and their ratio across epochs for different training strategies.

    Parameters
    ----------
    fixed_lambda_path : str
        Path to CSV file containing columns: ['Epoch', 'imputation_loss', 'forecast_loss']
    dwa_path : str
        Path to CSV file containing columns: ['Epoch', 'Imputation', 'Forecast']
    confidence_path : str, optional
        Path to CSV file containing columns: ['Epoch', 'imputation', 'forecast'] (for uncertainty weighting)
    figsize : tuple
        Figure size for the plot
    """

    # Seaborn style for publication-ready plots
    sns.set(style="whitegrid", context="talk", font_scale=1.1)

    # --- Load CSVs ---
    fixed_lambda = pd.read_csv(fixed_lambda_path)
    dwa = pd.read_csv(dwa_path)
    confidence = pd.read_csv(confidence_path) if confidence_path else None

    # --- Compute loss ratios ---
    fixed_lambda['loss_ratio'] = fixed_lambda['imputation_loss'] / fixed_lambda['forecast_loss']
    dwa['loss_ratio'] = dwa['Imputation'] / dwa['Forecast']
    if confidence is not None:
        confidence['loss_ratio'] = confidence['imputation'] / confidence['forecast']

    # --- Plotting ---
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # ---- Subplot 1: Task Losses ----
    ax[0].plot(fixed_lambda['Epoch'], fixed_lambda['imputation_loss'],
               marker='o', label='Imputation - Fixed λ', color='#9775fa')
    ax[0].plot(fixed_lambda['Epoch'], fixed_lambda['forecast_loss'],
               marker='o', linestyle='--', label='Forecast - Fixed λ', color='#9775fa', alpha=0.6)

    ax[0].plot(dwa['Epoch'], dwa['Imputation'],
               marker='^', label='Imputation - DWA', color='#4dabf7')
    ax[0].plot(dwa['Epoch'], dwa['Forecast'],
               marker='^', linestyle='--', label='Forecast - DWA', color='#4dabf7', alpha=0.6)

    if confidence is not None:
        ax[0].plot(confidence['Epoch'], confidence['imputation'],
                   marker='s', label='Imputation - Confidence', color='#2fb344')
        ax[0].plot(confidence['Epoch'], confidence['forecast'],
                   marker='s', linestyle='--', label='Forecast - Confidence', color='#2fb344', alpha=0.6)

    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Task Losses Across Epochs")
    ax[0].legend()
    ax[0].grid(True)

    # ---- Subplot 2: Loss Ratios ----
    ax[1].plot(fixed_lambda['Epoch'], fixed_lambda['loss_ratio'],
               marker='o', label='Fixed λ', color='#9775fa')
    ax[1].plot(dwa['Epoch'], dwa['loss_ratio'],
               marker='^', label='DWA', color='#4dabf7')

    if confidence is not None:
        ax[1].plot(confidence['Epoch'], confidence['loss_ratio'],
                   marker='s', label='Confidence', color='#2fb344')

    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss Ratio (Imputation / Forecast)")
    ax[1].set_title("Loss Balance Across Epochs")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_smoothed_node_series(true_vals, imputed_vals, node_idx=0, feature_idx=5, figsize=(12, 6)):
  
    # --- Validate shapes ---
    if true_vals.shape != imputed_vals.shape:
        raise ValueError(f"Shape mismatch: true_vals {true_vals.shape} vs imputed_vals {imputed_vals.shape}")

    B, T, N, F = true_vals.shape
    if not (0 <= node_idx < N):
        raise ValueError(f"node_idx={node_idx} out of range (0-{N-1})")
    if not (0 <= feature_idx < F):
        raise ValueError(f"feature_idx={feature_idx} out of range (0-{F-1})")

    # --- Smooth over time ---
    true_smoothed = true_vals.mean(axis=1)      # (B, N, F)
    imputed_smoothed = imputed_vals.mean(axis=1)

    true_series = true_smoothed[:, node_idx, feature_idx]
    imputed_series = imputed_smoothed[:, node_idx, feature_idx]

    # --- Plot ---
    plt.figure(figsize=figsize)
    sns.set(style="whitegrid", font_scale=1.2)

    plt.plot(true_series, label="True (smoothed)", color="#9775fa", linewidth=2)
    plt.plot(imputed_series, label="Imputed (smoothed)", color="#4dabf7", linestyle="--", linewidth=2)


    plt.title(f"Smoothed Values for Node {node_idx}, Feature {feature_idx}", fontsize=14)
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

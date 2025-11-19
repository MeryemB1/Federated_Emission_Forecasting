import logging
logging.basicConfig(level=logging.INFO)
import os
import flwr as fl
import torch
import torch.nn as nn
from fed_utils import *
import numpy as np
from sklearn.cluster import KMeans
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import flwr as fl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from sklearn.manifold import TSNE


class FedRepClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, adj, mask, cid, index, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.adj = adj.to(device) if hasattr(adj, "to") else adj
        self.mask = mask.to(device) if hasattr(mask, "to") else mask
        self.cid = cid
        self.device = device
        self.index = index
        self.current_phase = "joint"
        self.all_keys = get_all_keys(self.model)

        # Precompute static masks for imputation
        self.static_masks = []
        mask_ratio = 0.15
        for xb, yb in self.train_loader:
            B, _, N, _ = xb.shape
            masking, _ = generate_missing_nodes(self.adj, missing_ratio=mask_ratio, exception=self.index)
            masking = masking.to(self.device).unsqueeze(0).expand(B, -1)  # (B, N)
            self.static_masks.append(masking)

    # ---------------- get/set params ----------------
    def get_parameters(self, config=None):
        # Always return full model params in joint training
        return numpy_from_state(self.model, self.all_keys)

    def set_parameters(self, parameters):
        load_into_state(self.model, self.all_keys, parameters)

    # ---------------- training ----------------
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        alpha = float(config.get("alpha", 0.7))
        epochs_joint = int(config.get("epochs_joint", 3))
        lr_joint = float(config.get("lr_joint", 1e-3))

        mse_none = nn.MSELoss(reduction="none")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_joint)

        epoch_imp_losses = []
        epoch_foc_losses = []
        epoch_total_losses = []
        # DWA setup
        T = 2.0  # temperature
        task_num = 2
        epochs=epochs_joint
        loss_history = torch.zeros((epochs, task_num))  # store per-task losses
        lambdas = torch.ones(task_num) / task_num  # start with equal weights

        for epoch in range(epochs_joint):
            last_imp, last_foc, last_loss = 0.0, 0.0, 0.0
            for (xb, yb), bmask in zip(self.train_loader, self.static_masks):
                xb, yb, bmask = xb.to(self.device), yb.to(self.device), bmask.to(self.device)
                B,T,N,Fe = xb.shape
                bmask_expanded = bmask.unsqueeze(1).unsqueeze(-1).expand(B, T, N, Fe)
                target = 1 - bmask_expanded
                xb_noisy = xb * bmask_expanded
                impute, pred = self.model(xb_noisy, self.adj, bmask_expanded)
                L_imp = (mse_none(impute, xb) * target).sum() / target.sum()
                mask_f = self.mask.unsqueeze(0).unsqueeze(1).unsqueeze(-1)
                B, T, N, F = pred.shape
                mask_f = mask_f.expand(B, T, N, F)
                L_foc = mse_none(pred, yb) * mask_f
                L_foc = L_foc.sum() / mask_f.sum()
                loss = lambdas[0] * L_imp + lambdas[1] * L_foc

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                last_imp, last_foc, last_loss = L_imp.item(), L_foc.item(), loss.item()

        num_train_examples = sum(xb.size(0) for xb, _ in self.train_loader)
        checkpoint_dir = "/content/drive/MyDrive/manhattan/checkpointsfedcls"
        os.makedirs(checkpoint_dir, exist_ok=True)

        current_round = config.get("rnd", 0)
        checkpoint_path = os.path.join(checkpoint_dir, f"client_{self.cid}_round_{current_round}.pt")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "mask": self.mask.cpu(),  # ensure it's on CPU for portability
                "round": current_round
            },
            checkpoint_path
        )
        print(f"✅ Saved checkpoint for Client {self.cid} at Round {current_round} → {checkpoint_path}")

        os.makedirs("training_logs", exist_ok=True)
        with open(f"training_logs/client_{self.cid}_train_round_{config.get('round', 0)}_joint.txt", "w") as f:
            f.write(f"Round {config.get('rnd', 0)} Phase joint\n")
            f.write(f"Imputation loss: {last_imp:.6f}\n")
            f.write(f"Forecasting loss: {last_foc:.6f}\n")
            f.write(f"Total loss: {last_loss:.6f}\n")

        return numpy_from_state(self.model, self.all_keys), int(num_train_examples), {
            "phase": "joint", "loss_alpha": float(alpha),
        }

    # ---------------- evaluation ----------------
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        mse_none = nn.MSELoss(reduction="none")
        mse_mean = nn.MSELoss(reduction="mean")
        alpha = float(config.get("alpha", 0.7))

        total_loss = total_imp = total_foc = 0.0
        total_samples = 0

        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                B,T,N,Fe = xb.shape
                masked = self.mask.unsqueeze(0).unsqueeze(1).unsqueeze(-1).expand(B, T, N, Fe)
                target = 1 - masked
                xb_noisy = xb * masked

                concat, fwd_out = self.model.encode(xb_noisy, self.adj, masked)
                impute = self.model.head_impute(concat)
                pred   = self.model.head_forecast(fwd_out)

                L_imp = (mse_none(impute, xb) * target).sum() / target.sum()
                L_foc = mse_mean(pred, yb)
                L = alpha * L_imp + (1 - alpha) * L_foc

                bs = xb.size(0)
                total_samples += bs
                total_loss += L.item() * bs
                total_imp  += L_imp.item() * bs
                total_foc  += L_foc.item() * bs

        loss_value = total_loss / max(total_samples, 1)
        imp_value  = total_imp  / max(total_samples, 1)
        foc_value  = total_foc  / max(total_samples, 1)
        print("here round",config.get('round', 0))

        os.makedirs("validation_logs", exist_ok=True)
        with open(f"validation_logs/client_{self.cid}_eval_round_{config.get('round', 0)}_joint.txt", "w") as f:
            f.write(f"Round {config.get('rnd', 0)} Phase joint\n")
            f.write(f"Imputation loss: {imp_value:.6f}\n")
            f.write(f"Forecasting loss: {foc_value:.6f}\n")
            f.write(f"Total loss: {loss_value:.6f}\n")

        return float(loss_value), int(total_samples), {
            "phase": "joint",
            "imputation_loss": float(imp_value),
            "forecasting_loss": float(foc_value),
        }
    
def client_fn(all_loaders, subgraphs_with_masks,all_tests ,context: fl.common.Context):
    # --- identify client ---
    cid = str(context.node_config["partition-id"])
    i = int(cid)

    # --- load client-specific data ---
    loader = all_loaders[i]
    X, adj, mask, index = subgraphs_with_masks[i]
    adj = torch.tensor(adj, dtype=torch.float32)
    loadertest= all_tests[i]

    # --- build FedRep model (encoder + heads) ---
    model = FedRepModel(
        in_features=loader.dataset.tensors[0].shape[-1],   # input feature dim
        hidden_dim=16,
        forecast_len=loader.dataset.tensors[1].shape[1],   # prediction horizon
        sequence_len=loader.dataset.tensors[0].shape[1],   # sequence length
        num_nodes=loader.dataset.tensors[0].shape[2],      # number of nodes
        projection_dim=32,
        k_hop=2,
    )

    # --- wrap into FedRep client ---
    return FedRepClient(
        model=model,
        train_loader=loader,
        test_loader=loadertest,
        adj=adj,
        mask=mask,
        index=index,
        cid=cid,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ).to_client()

def flatten_params(weights_list):
    """
    Flatten client weights into 1D vectors for clustering.
    weights_list: list of list of numpy arrays (from get_parameters)
    """
    flattened = []
    for w in weights_list:
        vec = np.concatenate([p.flatten() for p in w])
        flattened.append(vec)
    return np.array(flattened)

def cluster_clients(weights_list, n_clusters=2):
    """
    Cluster clients dynamically based on their model parameters.
    Returns cluster assignments (list of cluster idx per client)
    """
    flattened = flatten_params(weights_list)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_ids = kmeans.fit_predict(flattened)
    return cluster_ids

class ClusteredFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, n_clusters=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_clusters = n_clusters
        self.client_cluster_map = {}  # client_id -> cluster_id
        self.cluster_models = {}      # cluster_id -> aggregated Parameters
        self.current_round = 0

    # ---------------- Clustered aggregation ----------------
    def aggregate_fit(self, rnd, results, failures):
        self.current_round = rnd
        if not results:
            return None, {}

        weights_list = []
        num_examples = []
        client_ids = []

        # ---------------- Collect client weights ----------------
        for client_proxy, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            n_examples = fit_res.num_examples

            weights_list.append(weights)
            num_examples.append(n_examples)
            client_ids.append(client_proxy.cid)

        # ---------------- Step 1: cluster clients ----------------
        cluster_ids = cluster_clients(weights_list, n_clusters=self.n_clusters)

        self.client_cluster_map = {}  # reset mapping
        for cid, cluster_id in zip(client_ids, cluster_ids):
            self.client_cluster_map[cid] = cluster_id

        # ---------------- Step 2: aggregate weights per cluster ----------------
        self.cluster_models = {}  # reset cluster models
        for cluster_id in np.unique(cluster_ids):
            idxs = [i for i, c in enumerate(cluster_ids) if c == cluster_id]
            cluster_weights = [weights_list[i] for i in idxs]
            cluster_num_examples = [num_examples[i] for i in idxs]

            total_examples = sum(cluster_num_examples)
            avg_weights = []
            for layer in range(len(cluster_weights[0])):
                layer_sum = sum(
                    cluster_weights[i][layer] * cluster_num_examples[i] / total_examples
                    for i in range(len(idxs))
                )
                avg_weights.append(layer_sum)

            self.cluster_models[cluster_id] = ndarrays_to_parameters(avg_weights)

        # ---------------- Step 3: metrics ----------------
        metrics_aggregated = {
            "cluster_ids": cluster_ids,
            "client_cluster_map": self.client_cluster_map
        }

        # ---------------- Step 4: Plot gradient profile per client using t-SNE ----------------
        try:
            weights_matrix = np.array([
                np.concatenate([w.flatten() for w in client_w])
                for client_w in weights_list
            ])  # shape [n_clients, total_params]

            tsne = TSNE(n_components=3, init='random', random_state=42, learning_rate='auto',perplexity=3)
            projected = tsne.fit_transform(weights_matrix)  # [n_clients, 3]

            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection='3d')

            markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'X']
            colors = ['r','g','b','c','m','y','k','orange']

            for cluster_id in range(self.n_clusters):
                idxs = np.where(cluster_ids == cluster_id)[0]
                ax.scatter(
                    projected[idxs,0],
                    projected[idxs,1],
                    projected[idxs,2],
                    s=60,
                    color=colors[cluster_id % len(colors)],
                    marker=markers[cluster_id % len(markers)],
                    alpha=0.6,
                    label=f"Cluster {cluster_id}"
                )

            ax.set_title(f"Client Gradients t-SNE Projection - Round {rnd}")
            ax.set_xlabel("t-SNE Dim 1")
            ax.set_ylabel("t-SNE Dim 2")
            ax.set_zlabel("t-SNE Dim 3")
            ax.legend()
            os.makedirs("cfl_results", exist_ok=True)
            plt.savefig(f"cfl_results/gradients_round_{rnd}_tsne.png")
            plt.show()
            plt.close()

        except Exception as e:
            print(f"Gradient plotting failed for round {rnd}: {e}")

        return None, metrics_aggregated





    # ---------------- Configure Fit ----------------

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins = []

        clients = client_manager.all().values()
        for client_proxy in clients:
            cid = client_proxy.cid  # should be valid if client_proxy is ClientProxy
            cluster_id = self.client_cluster_map.get(cid, 0)
            cluster_params = self.cluster_models.get(cluster_id, parameters)
            print(f"[DEBUG] Round {server_round}: Client {cid} assigned to Cluster {cluster_id}")
            print(f"[DEBUG] Cluster {cluster_id} model parameters (first layer first 5 elements): "
                  f"{parameters_to_ndarrays(cluster_params)[0][:5]}")
            fit_config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
            fit_config["rnd"] = server_round
            fit_ins.append((client_proxy, fl.server.client_proxy.FitIns(
                parameters=cluster_params,
                config=fit_config
            )))
        return fit_ins






    # ---------------- Optional: Configure Evaluate ----------------
    def configure_evaluate(self, server_round, parameters, client_manager):
        eval_ins = []
        for client_proxy in client_manager.all().values():  # or client_manager.all()
            cid = client_proxy.cid
            cluster_id = self.client_cluster_map.get(cid, 0)
            cluster_params = self.cluster_models.get(cluster_id, parameters)
            eval_config = self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn else {}
            eval_ins.append((
                client_proxy,  # Pair the ClientProxy here
                fl.server.client_proxy.EvaluateIns(
                    parameters=cluster_params,
                    config=eval_config
                )
            ))
        return eval_ins

def save_cluster_membership_and_models(client_objs, server_strategy):
    """
    client_objs: list of client instances (FedRepClient)
    server_strategy: ClusteredFedAvg instance
    """
    os.makedirs("cfl_results", exist_ok=True)

    # Save cluster membership
    cluster_map = server_strategy.client_cluster_map
    np.save("cfl_results/client_cluster_map.npy", cluster_map)

    # Save client models
    for client in client_objs:
        cid = client.cid
        model_path = f"cfl_results/client_{cid}_final_model.pt"
        torch.save(client.model.state_dict(), model_path)

def aggregate_metrics(metrics_list):
    forecasting_values, imputation_values = [], []

    for x in metrics_list:
        if isinstance(x, dict):
            metrics = x
        elif isinstance(x, tuple):
            if len(x) == 2:   # (num_examples, metrics_dict)
                metrics = x[1]
            elif len(x) == 3: # (loss, num_examples, metrics_dict)
                metrics = x[2]
            else:
                continue
        else:
            continue

        forecasting_values.append(metrics.get("forecasting_loss", 0.0))
        imputation_values.append(metrics.get("imputation_loss", 0.0))

    return {
        "forecasting_loss": np.mean(forecasting_values) if forecasting_values else 0.0,
        "imputation_loss": np.mean(imputation_values) if imputation_values else 0.0,
    }

def fit_config(server_round=None):
    # server_round can be optionally used if needed
    return {
        "alpha": 0.7,
        "phase": "joint",
        "epochs_joint": 6,
        "epochs_heads": 0,
        "lr_joint": 1e-3,
        "lr_heads": 1e-3,
        "round": server_round,
    }

def evaluate_config(server_round):
    # Customize evaluation config per round if needed
    return {"round": server_round}



# ------------------ Start training ------------------
def start_clustered_federated_training(num_rounds=10 , n_clusters=3):
    strategy = ClusteredFedAvg(
        fraction_fit=1.0,
        min_fit_clients=9,
        min_available_clients=0,
        fit_metrics_aggregation_fn=aggregate_metrics,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        on_fit_config_fn=fit_config,       # <- no rnd argument
        on_evaluate_config_fn=evaluate_config,
        n_clusters=n_clusters,
    )

    client_objs = []  # keep references to clients for saving models

    def client_fn_wrapper(context):
        client = client_fn(context)
        client_objs.append(client)
        return client

    fl.simulation.start_simulation(
        client_fn=client_fn_wrapper,
        num_clients=9,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Save cluster membership and final models
    save_cluster_membership_and_models(client_objs, strategy)

start_clustered_federated_training()
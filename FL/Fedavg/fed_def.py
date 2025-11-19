import flwr as fl
import torch
import torch.nn as nn
import os
import logging
from fed_utils import* 

logging.basicConfig(level=logging.INFO)

class FedRepClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader,test_loader, adj, mask, cid, index, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.adj = adj.to(device) if hasattr(adj, "to") else adj
        self.mask = mask.to(device) if hasattr(mask, "to") else mask
        self.cid = cid
        self.device = device
        self.index = index

        # For FedAvg, we share all keys (entire model)
        self.all_keys = list(model.state_dict().keys())

        # precompute static masks per batch
        self.static_masks = []
        mask_ratio = 0.15
        for xb, yb in self.train_loader:
            B, _, N, _ = xb.shape
            masking, _ = generate_missing_nodes(self.adj, missing_ratio=mask_ratio, exception=self.index)
            masking = masking.to(self.device).unsqueeze(0).expand(B, -1)  # (B, N)
            self.static_masks.append(masking)

    # --- parameter exchange: all parameters ---
    def get_parameters(self, config=None):
        return numpy_from_state(self.model, self.all_keys)

    def set_parameters(self, parameters):
        load_into_state(self.model, self.all_keys, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        current_round = config.get("rnd", 0)

        epochs = int(config.get("epochs", 5))
        lr = float(config.get("lr", 1e-3))
        alpha = float(config.get("alpha", 0.6))

        mse_none = nn.MSELoss(reduction="none")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        epoch_imp_losses = []
        epoch_foc_losses = []
        epoch_total_losses = []
        # DWA setup
        T = 2.0
        task_num = 2
        loss_history = torch.zeros((epochs, task_num))
        lambdas = torch.ones(task_num) / task_num

        for epoch in range(epochs):
            total_imp, total_foc, total_all = 0.0, 0.0, 0.0
            num_batches = 0
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
                L = lambdas[0] * L_imp + lambdas[1] * L_foc

                optimizer.zero_grad()
                L.backward()
                optimizer.step()

                total_imp += L_imp.item()
                total_foc += L_foc.item()
                total_all += L.item()
                num_batches += 1

            epoch_imp_losses.append(total_imp / num_batches)
            epoch_foc_losses.append(total_foc / num_batches)
            epoch_total_losses.append(total_all / num_batches)

            # Store per-task losses for DWA
            loss_history[epoch-1, 0] = epoch_imp_losses[-1]
            loss_history[epoch-1, 1] = epoch_foc_losses[-1]

            if epoch > 2:
                w = [loss_history[epoch-1, i] / loss_history[epoch-2, i] for i in range(task_num)]
                w = torch.tensor(w)
                lambdas = torch.softmax(w / T, dim=0).detach()

        # --- Save training losses ---
        os.makedirs("training_logs", exist_ok=True)
        save_path = os.path.join("training_logs", f"client_{self.cid}_training_losses_round_{current_round}.txt")
        with open(save_path, "w") as f:
            for i, (imp, foc, total) in enumerate(zip(epoch_imp_losses, epoch_foc_losses, epoch_total_losses)):
                f.write(f"Epoch {i}: imputation_loss={imp}, forecasting_loss={foc}, total_loss={total}\n")
        num_train_examples = sum(xb.size(0) for xb, _ in self.train_loader)
        return self.get_parameters({}), int(num_train_examples), {
            "imputation_loss": epoch_imp_losses[-1],
            "forecasting_loss": epoch_foc_losses[-1],
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        mse_none = torch.nn.MSELoss(reduction="none")
        mse_mean = torch.nn.MSELoss(reduction="mean")
        alpha = float(config.get("alpha", 0.6))
        current_round = config.get("rnd", None)
        total_loss = total_imp = total_foc = 0.0
        total_samples = 0

        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                B,T,N,Fe = xb.shape

                masked = self.mask.unsqueeze(0).unsqueeze(1).unsqueeze(-1).expand(B, T, N, Fe)
                target = 1 - masked
                xb_noisy = xb * masked

                impute, pred = self.model(xb_noisy, self.adj, masked)
                L_imp = (mse_none(impute, xb) * target).sum() / target.sum()
                L_foc = mse_mean(pred, yb)
                L = alpha * L_imp + (1 - alpha) * L_foc

                bs = xb.size(0)
                total_samples += bs
                total_loss += L.item() * bs
                total_imp  += L_imp.item() * bs
                total_foc  += L_foc.item() * bs

        loss_value = total_loss / total_samples if total_samples > 0 else 0.0
        imp_value  = total_imp  / total_samples if total_samples > 0 else 0.0
        foc_value  = total_foc  / total_samples if total_samples > 0 else 0.0
        save_path = os.path.join("validation_logs", f"client_{self.cid}_training_losses_round_{current_round}.txt")
        with open(save_path, "w") as f:
            f.write(f"Round {current_round} | Total: {loss_value:.4f}, "
                  f"Imputation: {imp_value:.4f}, "
                  f"Forecasting: {foc_value:.4f}\n")


        return float(loss_value), int(total_samples), {
            "imputation_loss": float(imp_value),
            "forecasting_loss": float(foc_value),
        }


def make_client_fn(all_loaders, subgraphs_with_masks, all_tests):
    """
    Factory that creates a client_fn with access to all_loaders, subgraphs_with_masks, and all_tests.
    """

    def client_fn(context: fl.common.Context):
        # --- identify client ---
        cid = str(context.node_config["partition-id"])
        i = int(cid)

        # --- load client-specific data ---
        loader = all_loaders[i]
        X, adj, mask, index = subgraphs_with_masks[i]
        loadertest = all_tests[i]

        adj = torch.tensor(adj, dtype=torch.float32)

        # --- build FedRep model ---
        model = FedRepModel(
            in_features=loader.dataset.tensors[0].shape[-1],   # input feature dim
            hidden_dim=16,
            forecast_len=loader.dataset.tensors[1].shape[1],   # prediction horizon
            sequence_len=loader.dataset.tensors[0].shape[1],   # sequence length
            num_nodes=loader.dataset.tensors[0].shape[2],      # number of nodes
            projection_dim=32,
            k_hop=3,
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

    return client_fn

def aggregate_metrics(m):
    foc_vals, imp_vals = [], []
    for x in m:
        metrics = x if isinstance(x, dict) else (x[2] if isinstance(x, tuple) and len(x)==3 else None)
        if metrics is None:
            continue
        foc_vals.append(float(metrics.get("forecasting_loss", 0.0)))
        imp_vals.append(float(metrics.get("imputation_loss", 0.0)))
    return {
        "forecasting_loss": float(np.mean(foc_vals)) if foc_vals else 0.0,
        "imputation_loss": float(np.mean(imp_vals)) if imp_vals else 0.0,
    }

class FedAvgWithLogging(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, rnd, results, failures):
        aggregated = super().aggregate_evaluate(rnd, results, failures)
        if aggregated is not None:
            loss, metrics = aggregated
            print(f"[Round {rnd}] loss={loss:.3f} foc={metrics.get('forecasting_loss',0):.3f} imp={metrics.get('imputation_loss',0):.3f}")
        return aggregated

import numpy as np
import flwr as fl

# Global tracking
global_metrics = {
    "round": [],
    "forecasting_loss": [],
    "imputation_loss": []
}

# --- Custom FedRep strategy with logging ---
class FedRepWithLogging(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_eval = super().aggregate_evaluate(rnd, results, failures)

        if aggregated_eval is not None:
            loss, metrics = aggregated_eval
            global_metrics["round"].append(rnd)
            global_metrics["forecasting_loss"].append(metrics.get("forecasting_loss", 0.0))
            global_metrics["imputation_loss"].append(metrics.get("imputation_loss", 0.0))

        return aggregated_eval


# --- Metrics Aggregation ---
def aggregate_metrics(metrics_list):
    forecasting_values = []
    imputation_values = []

    for x in metrics_list:
        if isinstance(x, dict):
            metrics = x
        elif isinstance(x, tuple):
            if len(x) == 2:  # (num_examples, metrics_dict)
                metrics = x[1]
            elif len(x) == 3:  # (loss, num_examples, metrics_dict)
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


# --- Config to send to clients each round ---
def fit_config(server_round: int):
    return {"rnd": server_round}


# --- Main training launcher ---
def start_federated_training(all_loaders, subgraphs_with_masks, all_tests):
    # ✅ Create client_fn with access to data
    from training_utils.client_utils import make_client_fn  # adjust path if needed
    client_fn = make_client_fn(all_loaders, subgraphs_with_masks, all_tests)

    # ✅ Define strategy
    strategy = FedRepWithLogging(
        fraction_fit=1.0,
        min_fit_clients=len(all_loaders),
        min_available_clients=len(all_loaders),
        fit_metrics_aggregation_fn=aggregate_metrics,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
    )

    # ✅ Start Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(all_loaders),
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


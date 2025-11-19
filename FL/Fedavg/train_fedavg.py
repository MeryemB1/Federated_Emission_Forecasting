# main.py
import os
import torch
from fed_utils import *
from train_fedavg import *


def main():
    # ─────────── Hyperparameters ───────────
    load_dir = r"C:\Users\km_ka\Desktop\Demo\Dataset\Graphs_Fed"
    load_test_dir = r"C:\Users\km_ka\Desktop\Demo\Dataset\Graphs_Fed\subgraphstest"
    seq_len   = 8
    pred_len  = 6
    batch_sz  = 16
    missing_ratio = 0.15

    # ─────────── Step 1: Load Subgraphs ───────────
    subgraphs_loaded = load_subgraphs(load_dir)
    subgraphs_loaded_test = load_subgraphs(load_test_dir)
    # ─────────── Step 2: Generate Masks ───────────
    
    subgraphs_with_masks = generate_masks_for_subgraphs(subgraphs_loaded, missing_ratio=missing_ratio)

    # ─────────── Step 3: Build Train Loaders ───────────
    all_loaders = build_dataloaders(subgraphs_with_masks, seq_len=seq_len, pred_len=pred_len, batch_sz=batch_sz)

    # ─────────── Step 4: Build Test Loaders ───────────
    # (Optional: you can split from train loaders or load a separate test set)
    all_tests = build_test_loaders(subgraphs_with_masks, seq_len=seq_len, pred_len=pred_len, batch_sz=batch_sz)

    # ─────────── Step 5: Launch Federated Training ───────────
    start_federated_training(all_loaders, subgraphs_with_masks, all_tests)


if __name__ == "__main__":
    main()

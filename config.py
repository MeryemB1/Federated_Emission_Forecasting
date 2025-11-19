from types import SimpleNamespace


model_config = SimpleNamespace(
    T = 2.0 ,
    task_num = 2  ,                     
    hidden_dim=16,
    prejection=32,           
    k_hops=4,              
   
)

training_config = SimpleNamespace(
    missing_ratio=0.15,
    seq_len   = 8,
    pred_len  = 6,
    batch_sz  = 10,
    epochs    = 50,
    lr        = 0.001,
    eps       = 1e-8,
    mask_ratio=0.20,
    weight_decay=1e-4,
    optimizer="adam",       
    scheduler="cosine",     
    early_stopping_patience=10,
    loss_fn="MSE"            
)

federated_config = SimpleNamespace(
    num_rounds=50,
    local_epochs=3,
    aggregation_method="FedAvg",  # or custom "FedCustom"
    adaptive_local_epochs=True,
    noise_proportion_monitor=True,
    global_convergence_threshold=0.01
)


data_config = SimpleNamespace(
    dataset_path="/content/drive/MyDrive/manhattan/datasets",
    adjacency_matrix_path="Meta_Data/G_custom.pkl",
  
)

logging_config = SimpleNamespace(
    save_model=True,
    model_save_path="./checkpoints/model_best.pth",
    save_training_logs=True,
    log_dir="./logs",
    plot_results=True
)


config = SimpleNamespace(
    model=model_config,
    training=training_config,
    federated=federated_config,
    data=data_config,
    logging=logging_config
)

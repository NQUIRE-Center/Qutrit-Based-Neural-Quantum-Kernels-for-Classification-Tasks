"""
Author: Camila Cristiano-Romero
Contact: ccristiano@bcamath.org / ccristiano001@ikasle.ehu.eus
License: MIT License
"""

#%%
import torch
import numpy as np
import pandas as pd
import os
import time
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from training.train_loop import run_training
from training.losses import (
    cost_Lz_Binary,
)
from utils.predict import (
    predict_Lz_Binary,
)
from utils.split_features_and_labels import split_features_and_labels
from utils.report_results import summarize_cv_results
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)



''' =========================================
                   MODELS
    =========================================
'''

from models.qnn_1qutrit_givens_Lz import QNN_0_Givens_Lz


MODEL = {
        "name": "QNN_1qutrit_Givens_Lz",
        "model_class": QNN_0_Givens_Lz,
    }


EXPERIMENT = {
    "name": "QNN_QUTRIT_LZ_BINARY",
    "loss_fn": cost_Lz_Binary,
    "predict_fn": predict_Lz_Binary,
}


def make_optimizer(model, LR, WD):
    return torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WD,
    )


''' =========================================
                  RUN TRAINING
    =========================================
'''

model_name = MODEL["name"]
model_class = MODEL["model_class"]
print("================= Model =================")
print(f"Model: {model_name}")   

# Hyperparameters
EPOCHS       = 70
BATCH_SIZE   = 64
LAYERS       = 6
LR           = 0.006
WD           = 1e-5
NUM_PARAMS   = 8
NUM_FEATURES = 8
SEED         = 99

# Dataset paths
print("============== Load dataset =============")
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = BASE_DIR / "datasets/binary/FashionMNIST_5-7_2000_5folds"
print(f"Looking for dataset in: {DATASET_DIR}")
    
# Look for available folds
FOLDS_DIR  = DATASET_DIR
folds = sorted([
        f for f in os.listdir(FOLDS_DIR)
        if f.startswith("fold") and f.endswith(f"train_{NUM_FEATURES}f.csv")
        ])
n_folds = len(folds)
print(f"Found {n_folds} folds")

# Run training for each fold
print("=============== Training ================")
results = []
for fold_idx in range(1, n_folds + 1):
    print(f"Training fold {fold_idx}/{n_folds}...")

    train_path = f"{FOLDS_DIR}/fold{fold_idx}_train_{NUM_FEATURES}f.csv"
    test_path   = f"{FOLDS_DIR}/fold{fold_idx}_test_{NUM_FEATURES}f.csv"
    
    # Load data
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    
    X_train, y_train = split_features_and_labels(df_train)
    X_test, y_test     = split_features_and_labels(df_test)

    # Run training
    optimizer_fn = partial(make_optimizer, LR=LR, WD=WD)
                           
    results_fold = run_training(
        seed_init=SEED,
        model_class=model_class,
        model_kwargs={                  
            "num_layers": LAYERS,
            "num_features": NUM_FEATURES,
            "num_params": NUM_PARAMS,
            },
        X_train=X_train, 
        y_train=y_train,
        X_test=X_test, 
        y_test=y_test,
        loss_fn=EXPERIMENT["loss_fn"],
        predict_fn=EXPERIMENT["predict_fn"],
        optimizer_fn=optimizer_fn,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        device="cpu")
    results.append(results_fold)


summary = summarize_cv_results(results)
print("=============== Results =================")
print(f"Train Accuracy: {summary['train_acc_mean']:.4f} ± {summary['train_acc_se']:.4f}")
print(f"Test Accuracy:  {summary['test_acc_mean']:.4f} ± {summary['test_acc_se']:.4f}")
print("=========================================")





#%%


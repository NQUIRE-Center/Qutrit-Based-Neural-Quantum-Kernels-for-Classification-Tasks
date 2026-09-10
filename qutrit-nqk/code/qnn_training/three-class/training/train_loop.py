"""
Author: Camila Cristiano-Romero
Contact: ccristiano@bcamath.org / ccristiano001@ikasle.ehu.eus
License: MIT License
"""

import time
import torch
from copy import deepcopy
from training.metrics import evaluate_metrics
from torch.utils.data import DataLoader
from utils.Custom_Dataset import CustomDataset
import random
import numpy as np


def run_training(
    seed_init: int,
    model_class,
    model_kwargs,
    X_train, 
    y_train,
    X_test, 
    y_test,
    loss_fn,
    predict_fn,
    optimizer_fn,
    epochs=50,
    batch_size=64,
    device="cpu",
):

    t0 = time.perf_counter()
    
    
    # ----------------------------
    # Data
    # ----------------------------
    
    seed = seed_init
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dl = DataLoader(
        CustomDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_dl  = DataLoader(
        CustomDataset(X_test, y_test),
        batch_size=batch_size, 
        shuffle=False)

    # ----------------------------
    # Inicialización
    # ----------------------------
    init_weights = [
        torch.rand(model_kwargs["num_params"]) * 2 - 1
        for _ in range(model_kwargs["num_layers"])
    ]
    
    model = model_class(
        num_layers=model_kwargs["num_layers"],
        num_features=model_kwargs["num_features"],
        init_weights=init_weights,
    ).to(device)

    optimizer = optimizer_fn(model)

    # ----------------------------
    # Training
    # ----------------------------
    for epoch in range(epochs):
        model.train()

        for data, target in train_dl:
            data   = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss   = loss_fn(output, target)
            loss.backward()
            optimizer.step()


    # ----------------------------
    # Final evaluation
    # ----------------------------
    train_loss, train_acc, train_f1, train_prec, train_recall = evaluate_metrics(
        train_dl, model, loss_fn, predict_fn, device
    )
    test_loss, test_acc, test_f1, test_prec, test_recall  = evaluate_metrics(
        test_dl, model, loss_fn, predict_fn, device
    )

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
        "train_prec": train_prec,
        "train_recall": train_recall,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_prec": test_prec,
        "test_recall": test_recall,
        "params_opt": deepcopy(model.state_dict()),
        "init_weights": [w.detach().clone() for w in init_weights],
        "train_time": time.perf_counter() - t0,
    }

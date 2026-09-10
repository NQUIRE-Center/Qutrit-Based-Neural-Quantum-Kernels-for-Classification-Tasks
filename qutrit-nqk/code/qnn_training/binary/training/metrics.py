"""
Author: Camila Cristiano-Romero
Contact: ccristiano@bcamath.org / ccristiano001@ikasle.ehu.eus
License: MIT License
"""

import torch
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score
)

def evaluate_metrics(dataloader, model, loss_fn, predict_fn, device="cpu"):
    model.eval()

    y_true, y_pred = [], []
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for data, target in dataloader:
            data   = data.to(device)
            target = target.to(device)

            output = model(data)
            loss   = loss_fn(output, target)

            preds = predict_fn(output)

            bs = data.size(0)
            total_loss += float(loss) * bs
            n += bs

            y_true.extend(target.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return (
        total_loss / max(n, 1),
        accuracy_score(y_true, y_pred) * 100,
        f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        precision_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
        )



        

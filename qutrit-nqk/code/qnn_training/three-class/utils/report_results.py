"""
Author: Camila Cristiano-Romero
Contact: ccristiano@bcamath.org / ccristiano001@ikasle.ehu.eus
License: MIT License
"""

import numpy as np
import pandas as pd

def summarize_cv_results(results):
    """
    Takes a list of fold result dictionaries and returns mean accuracy
    and standard error across folds.
    """

    train_accs = np.array([r["train_acc"] for r in results], dtype=float)
    test_accs  = np.array([r["test_acc"] for r in results], dtype=float)

    n_folds = len(results)

    summary = {
        "n_folds": n_folds,

        "train_acc_mean": train_accs.mean(),
        "train_acc_se": train_accs.std(ddof=1) / np.sqrt(n_folds),

        "test_acc_mean": test_accs.mean(),
        "test_acc_se": test_accs.std(ddof=1) / np.sqrt(n_folds),
    }

    return summary

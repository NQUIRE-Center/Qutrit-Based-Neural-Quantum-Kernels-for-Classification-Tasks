import numpy as np
import pandas as pd

def split_features_and_labels(df):
    if "label" in df.columns:
        X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
        y = df["label"].to_numpy(dtype=np.int64)
    else:
        X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
        y = df.iloc[:, -1].to_numpy(dtype=np.int64)
    return X, y


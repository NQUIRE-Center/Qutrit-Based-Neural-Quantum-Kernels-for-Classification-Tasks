#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Camila Cristiano-Romero
Contact: ccristiano@bcamath.org / ccristiano001@ikasle.ehu.eus
License: MIT License
"""


import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from sklearn import svm
from sklearn import datasets
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def gram_from_states(psiA: torch.Tensor, psiB: torch.Tensor) -> torch.Tensor:
    """
    psiA: [N,D] complex
    psiB: [M,D] complex
    returns K: [N,M] real, K_ij = |<psiB_j | psiA_i>|^2
    """
    overlap = torch.einsum("id,jd->ij", psiA, psiB.conj())
    return (overlap.abs() ** 2).real


def zero_state_n_qutrits(num_qutrits: int, device=None, dtype=torch.complex64):
    device = device or torch.device("cpu")
    ket0 = torch.tensor([[1], [0], [0]], dtype=dtype, device=device)

    state = ket0
    for _ in range(num_qutrits - 1):
        state = torch.kron(state, ket0)

    return state   # [3**n, 1]



def givens_3x3(i, j, theta, phi):
    """
    SU(2) embebida en (i,j) dentro de SU(3).
    theta, phi: [B] (float o complex; internamente usamos complex64)
    return: [B,3,3] complex64
    """
    device = theta.device
    B = theta.shape[0]
    ctype = torch.complex64

    U = torch.eye(3, dtype=ctype, device=device).unsqueeze(0).expand(B, -1, -1).clone()

    theta = theta.to(device=device)
    phi   = phi.to(device=device)

    ct = torch.cos(theta / 2)
    st = torch.sin(theta / 2)

    eiphi  = torch.exp(1j * phi.to(dtype=ctype))
    eimphi = torch.conj(eiphi)

    U[:, i, i] = ct.to(dtype=ctype)
    U[:, j, j] = ct.to(dtype=ctype)
    U[:, i, j] = -(eiphi * st.to(dtype=ctype))
    U[:, j, i] =  (eimphi * st.to(dtype=ctype))
    return U

def diag_phase(alpha1, alpha2):
    """
    D = diag(e^{i a1}, e^{i a2}, e^{-i(a1+a2)}), batch.
    alpha1, alpha2: [B]
    return: [B,3,3] complex64
    """
    device = alpha1.device
    B = alpha1.shape[0]
    ctype = torch.complex64

    a1 = alpha1.to(dtype=ctype, device=device)
    a2 = alpha2.to(dtype=ctype, device=device)
    a3 = -(a1 + a2)

    d0 = torch.exp(1j * a1)
    d1 = torch.exp(1j * a2)
    d2 = torch.exp(1j * a3)

    D = torch.zeros(B, 3, 3, dtype=ctype, device=device)
    D[:, 0, 0] = d0
    D[:, 1, 1] = d1
    D[:, 2, 2] = d2
    return D

def U_su3_from_8_enc(p8: torch.Tensor) -> torch.Tensor:
    """
    p8: [B,8] (se asume float/complex; internamente usamos complex64)
    orden (como tu encoding):
      x1,x2 -> diag_phase
      x3,x4 -> givens(01)
      x5,x6 -> givens(12)
      x7,x8 -> givens(01)
    return: [B,3,3]
    """
    # en tu código original batch_c es complex, pero trig/exp funcionan ok.
    x1, x2 = p8[:,0], p8[:,1]
    x3, x4 = p8[:,2], p8[:,3]
    x5, x6 = p8[:,4], p8[:,5]
    x7, x8 = p8[:,6], p8[:,7]

    D    = diag_phase(x1, x2)
    U01a = givens_3x3(0,1, x3, x4)
    U12  = givens_3x3(1,2, x5, x6)
    U01b = givens_3x3(0,1, x7, x8)

    return U01b @ U12 @ U01a @ D

def U_su3_from_8_var(w8: torch.Tensor) -> torch.Tensor:
    """
    w8: [B,8] o [8] (si es [8], expandimos a [B,8])
    orden (como tu variational):
      t01,p01,t12,p12,t02,p02,a1,a2
    return: [B,3,3]
    """
    if w8.ndim == 1:
        # [8] -> [1,8] (el caller suele expandir, pero por las dudas)
        w8 = w8.unsqueeze(0)

    t01, p01 = w8[:,0], w8[:,1]
    t12, p12 = w8[:,2], w8[:,3]
    t02, p02 = w8[:,4], w8[:,5]
    a1,  a2  = w8[:,6], w8[:,7]

    U01a = givens_3x3(0,1, t01, p01)
    U12  = givens_3x3(1,2, t12, p12)
    U01b = givens_3x3(0,1, t02, p02)
    D    = diag_phase(a1, a2)

    return U01b @ U12 @ U01a @ D

def local_encoding_unitary(
    x: torch.Tensor,
    num_features: int,
) -> torch.Tensor:
    device = x.device
    ctype = torch.complex64

    x = x.to(device=device, dtype=ctype).flatten()

    if num_features < 8:
        x_eff = x[:num_features]
        pad = torch.zeros(8 - num_features, dtype=ctype, device=device)
        x8 = torch.cat([x_eff, pad], dim=0)
    else:
        if x.shape[0] < 8:
            pad = torch.zeros(8 - x.shape[0], dtype=ctype, device=device)
            x8 = torch.cat([x, pad], dim=0)
        else:
            x8 = x[:8]

    return U_su3_from_8_enc(x8.unsqueeze(0))[0]


def local_variational_unitary(
    w: torch.Tensor,
) -> torch.Tensor:
    device = w.device
    ctype = torch.complex64

    w = w.to(device=device, dtype=ctype).flatten()
    return U_su3_from_8_var(w.unsqueeze(0))[0]


def cx_qutrit_full(num_qutrits: int, control: int, target: int,
                   device=None, dtype=torch.complex64):
    """
    Construye la CX_3 global de tamaño [3^n, 3^n]
    que actúa sobre los qutrits (control, target).
    Convención:
        CX_3 |a,b> = |a, b+a mod 3>
    """
    device = device or torch.device("cpu")
    d = 3
    dim = d ** num_qutrits

    CX = torch.zeros((dim, dim), dtype=dtype, device=device)

    for in_idx in range(dim):
        # convertir índice a dígitos base-3
        digits = []
        x = in_idx
        for _ in range(num_qutrits):
            digits.append(x % d)
            x //= d
        digits = digits[::-1]   # qutrit 0 = más significativo

        out_digits = digits.copy()
        out_digits[target] = (digits[target] + digits[control]) % d

        # volver a índice
        out_idx = 0
        for val in out_digits:
            out_idx = out_idx * d + val

        CX[out_idx, in_idx] = 1.0

    return CX

def kron_n(U: torch.Tensor, n: int) -> torch.Tensor:
    """
    Devuelve U^{⊗n}
    """
    out = U
    for _ in range(n - 1):
        out = torch.kron(out, U)
    return out

def entangler_chain_cx(num_qutrits: int, device=None, dtype=torch.complex64):
    """
    Devuelve el unitario global E correspondiente a una cadena:
        CX(0->1), CX(1->2), ..., CX(n-2 -> n-1)
    """
    device = device or torch.device("cpu")
    dim = 3 ** num_qutrits
    E = torch.eye(dim, dtype=dtype, device=device)

    for c in range(num_qutrits - 1):
        CX_ct = cx_qutrit_full(
            num_qutrits=num_qutrits,
            control=c,
            target=c + 1,
            device=device,
            dtype=dtype,
        )
        E = CX_ct @ E

    return E

def get_layer_weights(weights, layer: int, device=None, dtype=None):
    w = weights[layer] if isinstance(weights, (list, tuple)) else weights[layer]
    if not torch.is_tensor(w):
        w = torch.tensor(w, device=device)
    if device is not None:
        w = w.to(device=device)
    if dtype is not None:
        w = w.to(dtype=dtype)
    return w

def qnn_nqutrit_cx_state(
    x: torch.Tensor,          # [F]
    weights,
    num_layers: int,
    num_features: int,
    num_qutrits: int,
) -> torch.Tensor:
    """
    Estado final de n qutrits usando:
      U_enc(x)^{⊗n}
      U_var(w_l)^{⊗n}
      entangler CX nearest-neighbours
    """
    device = x.device
    dtype = torch.complex64

    state = zero_state_n_qutrits(
        num_qutrits=num_qutrits,
        device=device,
        dtype=dtype
    )  # [3^n, 1]

    E = entangler_chain_cx(
        num_qutrits=num_qutrits,
        device=device,
        dtype=dtype
    )

    for layer in range(num_layers):
        w = get_layer_weights(weights, layer, device=device, dtype=dtype)

        U_enc = local_encoding_unitary(x, num_features)   # [3,3]
        U_var = local_variational_unitary(w)              # [3,3]

        U_enc_global = kron_n(U_enc, num_qutrits)         # [3^n,3^n]
        U_var_global = kron_n(U_var, num_qutrits)         # [3^n,3^n]

        state = U_enc_global @ state
        state = U_var_global @ state
        state = E @ state

    return state.squeeze(-1)   # [3^n]

def compute_states_nqutrit_cx(
    X: torch.Tensor,          # [B,F]
    weights,
    num_layers: int,
    num_features: int,
    num_qutrits: int,
) -> torch.Tensor:
    psis = []

    with torch.no_grad():
        for i in range(X.shape[0]):
            psi = qnn_nqutrit_cx_state(
                x=X[i],
                weights=weights,
                num_layers=num_layers,
                num_features=num_features,
                num_qutrits=num_qutrits,
            )
            psis.append(psi)

    return torch.stack(psis, dim=0)   # [B,3^n]


def load_fold(
    data_dir,
    fold,
    n_features,
):
    """
    Carga los datos procesados (train + val) de un fold específico del dataset HAR binario.

    Soporta dos formatos:
      - CSV con columna 'label'
      - CSV sin header, donde la última columna es la etiqueta
    """
    data_dir = Path(data_dir)

    train_path = data_dir / f"fold{fold}_train_{n_features}f.csv"
    val_path   = data_dir / f"fold{fold}_val_{n_features}f.csv"

    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)

    # --- detectar columna de label ---
    if "label" in df_train.columns:
        # Caso: CSV con header y columna 'label'
        X_train = df_train.drop(columns=["label"]).to_numpy(dtype=np.float32)
        y_train = df_train["label"].to_numpy(dtype=np.int64)

        X_val = df_val.drop(columns=["label"]).to_numpy(dtype=np.float32)
        y_val = df_val["label"].to_numpy(dtype=np.int64)
    else:
        # Caso: CSV sin header → última columna es la etiqueta
        X_train = df_train.iloc[:, :-1].to_numpy(dtype=np.float32)
        y_train = df_train.iloc[:, -1].to_numpy(dtype=np.int64)

        X_val = df_val.iloc[:, :-1].to_numpy(dtype=np.float32)
        y_val = df_val.iloc[:, -1].to_numpy(dtype=np.int64)

    print(f"✔ Fold {fold} cargado ({n_features} features):")
    print(f"  Train: {X_train.shape[0]} muestras")
    print(f"  Val  : {X_val.shape[0]} muestras")
    return X_train, y_train, X_val, y_val


def load_best_trial(path, metric="train_loss", mode="min"):
    """
    path: ruta al .pkl
    metric: métrica a optimizar
    mode: "min" o "max"
    """

    with open(path, "rb") as f:
        results = pickle.load(f)

    assert len(results) > 0, "Archivo vacío"

    if mode == "min":
        best = min(results, key=lambda x: x[metric])
    elif mode == "max":
        best = max(results, key=lambda x: x[metric])
    else:
        raise ValueError("mode debe ser 'min' o 'max'")

    print(
        f"🏆 Best trial: {best['trial']} (fold {best['fold']}) | "
        f"{metric} = {best[metric]:.6f} |  " 
        f"train_acc = {best['train_acc']:.2f}% | "
        f"val_acc = {best['val_acc']:.2f}% "
    )

    return best

from collections import OrderedDict

def extract_qnn_params_from_state_dict(state_dict, prefix="weights."):
    """
    Extrae todos los tensores weights.i del state_dict y los devuelve ordenados.
    """

    params = []

    i = 0
    while f"{prefix}{i}" in state_dict:
        params.append(state_dict[f"{prefix}{i}"].detach().clone())
        i += 1

    if len(params) == 0:
        raise ValueError("No se encontraron parámetros con prefijo:", prefix)

    print(f"✅ {len(params)} layers cargadas")

    return params

#%%

LAYERS       = 6
NUM_PARAMS   = 8

DATASET_DIR = "/Users/camilacristiano/Documents/Proyects/QML/Qudits_2/FashionMNIST_2classes/dataset/FashionMNIST_5-7_2000_5folds"
SAVE_DIR    = "/Users/camilacristiano/Documents/Proyects/QML/Qudits_2/FashionMNIST_2classes/data/FashionMNIST_5-7_2000_5folds_StateDict_corregidoSz"
MODEL_PARAMETRIZATION       = "QNN_1qutrit_Givens_M2_Lz" 
MODEL   = "1-to-n_Kernel_Givens_M2_nn"

device = "cpu"


for NUM_FEATURES in [4, 6, 8]:

    all_results = []
        
    for num_qutrits in [1, 2, 3, 4, 5, 6]:

        print("\n" + "#" * 70)
        print(f"RUNNING EXPERIMENT FOR {num_qutrits} QUTRITS")
        print("#" * 70)

        results = []

        for fold_idx in [1, 2, 3, 4, 5]:

            print(f"\nQutrits = {num_qutrits} | Fold = {fold_idx}")

            X_train, y_train, X_test, y_test = load_fold(
                data_dir=DATASET_DIR,
                fold=fold_idx,
                n_features=NUM_FEATURES
            )

            path_data_q0 = os.path.join(SAVE_DIR, MODEL_PARAMETRIZATION, f"results_1qutrit_NUM_FEATURES_{NUM_FEATURES}_NUM_PARAMS_{NUM_PARAMS}_LAYERS_{LAYERS}_fold{fold_idx}.pkl")

            best_trial      = load_best_trial(path_data_q0, metric="train_loss", mode="min")
            best_state_dict = best_trial["params_opt"]
            best_params     = extract_qnn_params_from_state_dict(best_state_dict)

            device = "cpu"

            Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
            Xte = torch.tensor(X_test,  dtype=torch.float32, device=device)

            psi_tr = compute_states_nqutrit_cx(
                Xtr,
                weights=best_params,
                num_layers=LAYERS,
                num_features=NUM_FEATURES,
                num_qutrits=num_qutrits,
            )

            psi_te = compute_states_nqutrit_cx(
                Xte,
                weights=best_params,
                num_layers=LAYERS,
                num_features=NUM_FEATURES,
                num_qutrits=num_qutrits,
            )

            K_train = gram_from_states(psi_tr, psi_tr).numpy()
            K_test  = gram_from_states(psi_te, psi_tr).numpy()

            clf = svm.SVC(kernel="precomputed")
            clf.fit(K_train, y_train)

            pred_train = clf.predict(K_train)
            pred_test  = clf.predict(K_test)

            train_acc = accuracy_score(y_train, pred_train)
            test_acc  = accuracy_score(y_test, pred_test)

            print(f"Kernel train acc: {train_acc*100:.2f}%")
            print(f"Kernel test  acc: {test_acc*100:.2f}%")
            print(f"QNN    train acc: {best_trial['train_acc']:.2f}%")
            print(f"QNN    test  acc: {best_trial['val_acc']:.2f}%")

            results_fold = {
                "num_qutrits": num_qutrits,
                "fold": fold_idx,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "qnn_train_acc": best_trial["train_acc"] / 100.0,
                "qnn_test_acc": best_trial["val_acc"] / 100.0,
            }

            results.append(results_fold)
            all_results.append(results_fold)
            

    df_results = pd.DataFrame(all_results)
    print(df_results.head())

    out_file = os.path.join(SAVE_DIR, MODEL, f"results_NUM_FEATURES_{NUM_FEATURES}_LAYERS_{LAYERS}.pkl")

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, "wb") as f:
        pickle.dump(all_results, f)

    print(f"✅ Resultados guardados en {out_file}")


    df_summary = (
        df_results
        .groupby("num_qutrits", as_index=False)
        .agg({
            "train_acc": ["mean", "std"],
            "test_acc": ["mean", "std"],
            "qnn_train_acc": ["mean"],
            "qnn_test_acc": ["mean"],
        })
    )

    df_summary.columns = [
        "num_qutrits",
        "train_acc_mean", "train_acc_std",
        "test_acc_mean", "test_acc_std",
        "qnn_train_acc_mean",
        "qnn_test_acc_mean",
    ]

    print(df_summary)

    import matplotlib.pyplot as plt

    x = df_summary["num_qutrits"]

    plt.figure(figsize=(8,5))

    # Kernel accuracies
    plt.errorbar(
        x,
        df_summary["train_acc_mean"],
        yerr=df_summary["train_acc_std"],
        marker="o",
        capsize=4,
        color='green',
        label="Kernel train acc"
    )

    plt.errorbar(
        x,
        df_summary["test_acc_mean"],
        yerr=df_summary["test_acc_std"],
        marker="s",
        capsize=4,
        color='orange',
        label="Kernel test acc"
    )

    # Horizontal lines for 1-qutrit QNN
    plt.axhline(
        y=df_summary["qnn_train_acc_mean"].iloc[0],
        linestyle="--",
        color='green',
        label="1-qutrit QNN train acc"
    )

    plt.axhline(
        y=df_summary["qnn_test_acc_mean"].iloc[0],
        linestyle="--",
        color='orange',
        label="1-qutrit QNN test acc"
    )
    plt.xlabel("Number of qutrits")
    plt.ylabel("Accuracy")
    plt.title(f"Kernel accuracy vs number of qutrits | NUM_FEATURES={NUM_FEATURES} | LAYERS={LAYERS}")
    plt.xticks(x)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, MODEL, f"accuracy_vs_qutrits_NUM_FEATURES_{NUM_FEATURES}_LAYERS_{LAYERS}.png"))



# Neural-Quantum-Kernels

## Repository structure

This repository contains the datasets and example code used for the qutrit Neural Quantum Kernel (NQK) experiments.

The main folder is organized as follows:

```text
qutrit-qnk/
├── datasets/
│   ├── binary/
│   └── three-class/
└── code/
    ├── qnn_training/
    │   ├── binary/
    │   └── three-class/
    └── nqk/
```

The `datasets/` directory contains the exact train/test splits used in the experiments. The data are organized by classification setting: `binary/` for binary classification and `three-class/` for three-class classification. Each dataset is provided in stratified folds. File names follow the convention

```text
fold{fold_number}_{split}_{num_features}f.csv
```

For example, `fold1_train_8f.csv` corresponds to the training split of fold 1 using 8 input features. The suffix `{num_features}f` indicates the number of encoded features used in that experiment.

The `code/` directory contains example scripts for reproducing the main experiments. The `qnn_training/` folder includes the code used to train the qutrit QNN models, separated into `binary/` and `three-class/` settings. The main difference between these settings is the cost function and the corresponding circuit readout: binary classification uses an (S_z)-based readout with a mean-squared error cost function, while three-class classification uses computational-basis probabilities with a multiclass cross-entropy cost function.

The `nqk/` folder contains the code used to construct the (1)-to-(n) Neural Quantum Kernel from a trained single-qutrit QNN embedding. This includes replicating the learned single-qutrit feature map across a larger register, applying the fixed entangling layer, constructing the kernel matrix, and training the final classical kernel classifier.

The code includes the three single-qutrit (\mathrm{SU}(3)) parametrizations studied in the paper: the geometric Lie-algebra exponential parametrization, the Euler-angle decomposition, and the Givens-rotation decomposition.

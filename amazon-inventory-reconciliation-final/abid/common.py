"""Shared constants and numpy-only helpers (no torch import here)."""
import json
import os
import random

import numpy as np

NUM_CLASSES = 6            # bins with 0..5 items
CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]
RANDOM_BASELINE = 1.0 / NUM_CLASSES

# Reference numbers reported by Bertorello, Sripada & Dendumrongsup (Stanford ML project,
# "Amazon Inventory Reconciliation Using AI") on a 150K / 324K subset with <=5 items per bin.
PAPER_REF = {
    "cnn_resnet34_adam_324k": {"acc": 0.562, "rmse": 0.90},
    "cnn_resnet34_sgdr_150k": {"acc": 0.538, "rmse": 0.94},
    "svm_raw": {"acc": 0.26},
    "svm_pca": {"acc": 0.32},
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_metrics(y_true, y_pred):
    """Accuracy, RMSE/MAE on counts, within-1 accuracy, per-class accuracy, confusion matrix."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    err = y_pred - y_true
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class = {}
    for c in range(NUM_CLASSES):
        n = cm[c].sum()
        per_class[str(c)] = float(cm[c, c] / n) if n else None
    return {
        "n": int(len(y_true)),
        "acc": float((err == 0).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "within1": float((np.abs(err) <= 1).mean()),
        "per_class_acc": per_class,
        "confusion": cm.tolist(),
    }


def load_splits(data_dir):
    import pandas as pd
    df = pd.read_csv(os.path.join(data_dir, "splits.csv"))
    return {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)

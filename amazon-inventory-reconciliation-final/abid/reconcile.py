"""Inventory-variance detection built on top of the count predictor.

The ABID labels are the *true* counts, so recorded-inventory errors are SIMULATED: with
probability `rate` a bin's recorded count is off by +-1 or +-2 (clipped to 0..5). A bin is flagged
when the model's predicted count differs from the recorded count. We also score every bin with
1 - P(recorded count) and report the AUROC of that score as a detector (threshold-free).
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from abid.common import NUM_CLASSES


def simulate_recorded(y_true, rate, rng):
    rec = y_true.copy()
    bad = rng.rand(len(y_true)) < rate
    for i in np.where(bad)[0]:
        opts = [d for d in (-2, -1, 1, 2) if 0 <= y_true[i] + d < NUM_CLASSES]
        rec[i] = y_true[i] + rng.choice(opts)
    return rec


def detect(y_true, pred, probs, rate=0.15, seed=0):
    rng = np.random.RandomState(seed)
    rec = simulate_recorded(np.asarray(y_true), rate, rng)
    truth = rec != y_true                                   # a real variance exists
    flag = pred != rec
    tp, fp, fn = (flag & truth).sum(), (flag & ~truth).sum(), (~flag & truth).sum()
    prec = tp / max(tp + fp, 1)
    rec_ = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec_ / max(prec + rec_, 1e-9)
    score = 1.0 - probs[np.arange(len(rec)), rec]
    auroc = roc_auc_score(truth, score) if 0 < truth.sum() < len(truth) else float("nan")
    return {"precision": float(prec), "recall": float(rec_), "f1": float(f1), "flag_rate": float(flag.mean()),
            "auroc": float(auroc), "true_variance_rate": float(truth.mean())}


def detect_multi(y_true, pred, probs, rate=0.15, seeds=20):
    runs = [detect(y_true, pred, probs, rate, s) for s in range(seeds)]
    return {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
            for k in runs[0]} | {"rate": rate, "seeds": seeds}

"""Aggregate baselines + CNN runs into results/results.json and publication figures.

Model selection uses the VALIDATION split (CNN: val accuracy with TTA); test numbers are reported.
Usage: python make_results.py --data data --runs runs --out results
"""
import argparse
import glob
import json
import os
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from abid.common import CLASS_NAMES, PAPER_REF, RANDOM_BASELINE, load_json, save_json
from abid.reconcile import detect_multi

NAVY, TEAL, ORANGE, GREY = "#1F3A5F", "#2A9D8F", "#E76F51", "#9AA5B1"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

BASE_LABELS = {"logreg_pca": "LogReg\n(PCA)", "linear_svc_pca": "Linear SVM\n(PCA)", "decision_tree_pca": "Dec. tree\n(PCA)",
               "rbf_svm_pca": "RBF-SVM\n(PCA)", "rbf_svm_hog": "RBF-SVM\n(HOG)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--baselines", default="results/baselines.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--var-rate", type=float, default=0.15)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    res = {"generated": str(date.today()), "paper_ref": PAPER_REF}
    sp = os.path.join(a.data, "stats.json")
    if os.path.exists(sp):
        res["data"] = load_json(sp)

    base = load_json(a.baselines) if os.path.exists(a.baselines) else {}
    res["baselines"] = {k: v for k, v in base.items() if not k.startswith("_")}
    res["best_baseline"] = base.get("_best_by_val")

    runs = {}
    for p in sorted(glob.glob(os.path.join(a.runs, "*", "metrics.json"))):
        m = load_json(p)
        m["_dir"] = os.path.dirname(p)
        runs[m["name"]] = m
    res["cnn"] = {k: {kk: vv for kk, vv in v.items() if kk != "_dir"} for k, v in runs.items()}
    best = max(runs, key=lambda k: runs[k]["val_tta"]["acc"]) if runs else None
    res["best_cnn"] = best

    if best:
        b = runs[best]
        t = b["test_tta"]
        bb = res["baselines"].get(res["best_baseline"]) if res["best_baseline"] else None
        summ = {"cnn_name": best, "cnn_acc": t["acc"], "cnn_rmse": t["rmse"], "cnn_mae": t["mae"], "cnn_within1": t["within1"],
                "cnn_acc_no_tta": b["test"]["acc"], "x_over_random": t["acc"] / RANDOM_BASELINE}
        if bb:
            summ.update(baseline_name=res["best_baseline"], baseline_acc=bb["test"]["acc"], baseline_rmse=bb["test"]["rmse"],
                        rel_improvement=t["acc"] / bb["test"]["acc"] - 1)
        res["summary"] = summ

        # variance-detection simulation on the best run's test predictions
        pr = os.path.join(b["_dir"], "test_probs.npy")
        pp = os.path.join(b["_dir"], "test_preds.csv")
        if os.path.exists(pr) and os.path.exists(pp):
            df = pd.read_csv(pp)
            res["reconciliation"] = detect_multi(df.y.values, df.pred.values, np.load(pr), rate=a.var_rate)

    save_json(res, os.path.join(a.out, "results.json"))
    make_figures(res, runs, best, a)
    print(json.dumps(res.get("summary", {}), indent=2))
    print(f"wrote {a.out}/results.json and figures")


def make_figures(res, runs, best, a):
    out = a.out
    # class distribution
    if "data" in res:
        cc = res["data"]["class_counts"]
        fig, ax = plt.subplots(figsize=(4.2, 2.6))
        ax.bar(list(cc), list(cc.values()), color=NAVY)
        ax.set_xlabel("items in bin"); ax.set_ylabel("images"); ax.set_title("Class distribution (subset)")
        fig.tight_layout(); fig.savefig(f"{out}/fig_class_dist.png"); plt.close(fig)

    # model comparison (test accuracy)
    labels, vals, cols = [], [], []
    for k, v in res["baselines"].items():
        if k in BASE_LABELS:
            labels.append(BASE_LABELS[k]); vals.append(v["test"]["acc"] * 100); cols.append(GREY)
    for k, v in runs.items():
        labels.append(k.replace("_", "\n", 1)); vals.append(v["test_tta"]["acc"] * 100); cols.append(TEAL if k == best else NAVY)
    if vals:
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        bars = ax.bar(range(len(vals)), vals, color=cols)
        ax.axhline(RANDOM_BASELINE * 100, ls="--", c=ORANGE, lw=1); ax.text(len(vals) - 0.5, RANDOM_BASELINE * 100 + 1, "random 16.7%", ha="right", color=ORANGE, fontsize=8)
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
        ax.set_ylabel("test accuracy (%)"); ax.set_ylim(0, max(vals) * 1.18)
        fig.tight_layout(); fig.savefig(f"{out}/fig_model_comparison.png"); plt.close(fig)

    if not best:
        return
    b = runs[best]
    # confusion matrix
    cm = np.array(b["test_tta"]["confusion"], dtype=float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > .5 else "black")
    ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.set_xlabel("predicted count"); ax.set_ylabel("true count")
    ax.set_title(f"Test confusion ({best})", fontsize=9)
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(f"{out}/fig_confusion.png"); plt.close(fig)

    # per-class accuracy: best baseline vs best CNN
    bb = res["baselines"].get(res.get("best_baseline") or "")
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    w = 0.38
    x = np.arange(6)
    if bb:
        ax.bar(x - w / 2, [(bb["test"]["per_class_acc"][c] or 0) * 100 for c in CLASS_NAMES], w, color=GREY, label=BASE_LABELS.get(res["best_baseline"], res["best_baseline"]).replace("\n", " "))
    ax.bar(x + (w / 2 if bb else 0), [(b["test_tta"]["per_class_acc"][c] or 0) * 100 for c in CLASS_NAMES], w, color=TEAL, label=best)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES); ax.set_xlabel("items in bin"); ax.set_ylabel("test accuracy (%)"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(f"{out}/fig_per_class.png"); plt.close(fig)

    # training curves
    h = os.path.join(b["_dir"], "history.csv")
    if os.path.exists(h):
        d = pd.read_csv(h); d["step"] = range(1, len(d) + 1)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
        a1.plot(d.step, d.train_loss, c=NAVY); a1.set_xlabel("epoch"); a1.set_ylabel("train loss")
        a2.plot(d.step, d.train_acc * 100, c=NAVY, label="train"); a2.plot(d.step, d.val_acc * 100, c=TEAL, label="val")
        a2.set_xlabel("epoch"); a2.set_ylabel("accuracy (%)"); a2.legend(frameon=False, fontsize=8)
        n_head = int((d.stage == "head").sum())
        for ax in (a1, a2):
            if n_head:
                ax.axvline(n_head + 0.5, ls=":", c=GREY)
        fig.tight_layout(); fig.savefig(f"{out}/fig_training_curves.png"); plt.close(fig)

    # LR finder
    lf = os.path.join(b["_dir"], "lr_find.json")
    if os.path.exists(lf):
        d = load_json(lf)
        fig, ax = plt.subplots(figsize=(3.6, 2.4))
        ax.semilogx(d["lrs"], d["losses"], c=NAVY); ax.axvline(d["used"], c=ORANGE, ls="--", lw=1)
        ax.set_xlabel("learning rate"); ax.set_ylabel("smoothed loss"); ax.set_title(f"LR finder (used {d['used']:.0e})", fontsize=9)
        fig.tight_layout(); fig.savefig(f"{out}/fig_lrfinder.png"); plt.close(fig)

    # qualitative examples
    pp = os.path.join(b["_dir"], "test_preds.csv")
    if os.path.exists(pp):
        d = pd.read_csv(pp)
        ok, bad = d[d.y == d.pred].head(4), d[d.y != d.pred].head(4)
        sel = pd.concat([ok, bad])
        fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.9))
        for ax, (_, r) in zip(axs.ravel(), sel.iterrows()):
            ax.imshow(Image.open(os.path.join(a.data, "images", f"{int(r.id)}.jpg"))); ax.axis("off")
            ax.set_title(f"true {int(r.y)} / pred {int(r.pred)}", fontsize=8, color=TEAL if r.y == r.pred else ORANGE)
        for ax in axs.ravel()[len(sel):]:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(f"{out}/fig_examples.png"); plt.close(fig)


if __name__ == "__main__":
    main()

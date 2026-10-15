"""Classical baselines: random, majority class, LogReg / LinearSVC / Decision Tree / RBF-SVM on
PCA-compressed raw pixels, and RBF-SVM on HOG features. Hyper-parameters are chosen on the
validation split; the test split is touched once per model.

Usage: python baselines.py --data data --out results/baselines.json
"""
import argparse
import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from abid.common import NUM_CLASSES, compute_metrics, load_splits, save_json, set_seed
from abid.features import hog_features, load_images


def evaluate(name, model, Xtr, ytr, Xva, yva, Xte, yte, results, extra=None):
    t0 = time.time()
    model.fit(Xtr, ytr)
    val = compute_metrics(yva, model.predict(Xva))
    test = compute_metrics(yte, model.predict(Xte))
    results[name] = {"val_acc": val["acc"], "test": test, "fit_seconds": round(time.time() - t0, 1)}
    if extra:
        results[name].update(extra)
    print(f"{name:34s} val acc {val['acc']:.3f} | test acc {test['acc']:.3f} rmse {test['rmse']:.2f}")


def tune_svc(Xtr, ytr, Xva, yva, grid):
    best, best_acc = None, -1
    for C in grid:
        m = SVC(kernel="rbf", C=C, gamma="scale").fit(Xtr, ytr)
        acc = (m.predict(Xva) == yva).mean()
        if acc > best_acc:
            best, best_acc = C, acc
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/baselines.json")
    ap.add_argument("--size", type=int, default=64, help="downsampled image size for raw pixels")
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--svm-train", type=int, default=8000, help="train subsample for RBF-SVM (O(n^2))")
    ap.add_argument("--no-hog", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    sp = load_splits(args.data)
    img_dir = os.path.join(args.data, "images")
    y = {s: sp[s].qty.values for s in sp}
    results = {}

    # trivial references
    maj = np.bincount(y["train"], minlength=NUM_CLASSES).argmax()
    results["random_guess"] = {"test": {"acc": 1 / NUM_CLASSES, "rmse": None}}
    results["majority_class"] = {"test": compute_metrics(y["test"], np.full_like(y["test"], maj))}

    print("loading raw pixels ...")
    raw = {s: load_images(sp[s].id.values, img_dir, args.size).reshape(len(sp[s]), -1) for s in sp}

    # standardise + PCA (fit on a train subsample for speed)
    sub = rng.choice(len(raw["train"]), size=min(10000, len(raw["train"])), replace=False)
    scaler = StandardScaler().fit(raw["train"][sub].astype(np.float32))
    pca = PCA(n_components=args.pca, svd_solver="randomized", random_state=args.seed)
    pca.fit(scaler.transform(raw["train"][sub].astype(np.float32)))
    P = {s: pca.transform(scaler.transform(raw[s].astype(np.float32))) for s in sp}
    evr = float(pca.explained_variance_ratio_.sum())
    print(f"PCA({args.pca}) keeps {evr:.1%} variance")
    del raw

    args_eval = (P["train"], y["train"], P["val"], y["val"], P["test"], y["test"], results)
    evaluate("logreg_pca", LogisticRegression(max_iter=300, C=1.0), *args_eval, extra={"pca_var": evr})
    evaluate("linear_svc_pca", LinearSVC(C=0.1, dual=False), *args_eval)
    evaluate("decision_tree_pca", DecisionTreeClassifier(max_depth=12, random_state=args.seed), *args_eval)

    # RBF-SVM: tune C on val using a train subsample
    idx = rng.choice(len(P["train"]), size=min(args.svm_train, len(P["train"])), replace=False)
    Xs, ys = P["train"][idx], y["train"][idx]
    C = tune_svc(Xs, ys, P["val"][:3000], y["val"][:3000], [1, 10, 100])
    evaluate("rbf_svm_pca", SVC(kernel="rbf", C=C, gamma="scale"), Xs, ys, P["val"], y["val"],
             P["test"], y["test"], results, extra={"C": C, "train_subsample": int(len(idx))})

    if not args.no_hog:
        print("computing HOG ...")
        G = {s: hog_features(load_images(sp[s].id.values, img_dir, args.size, gray=True)) for s in sp}
        Gs = G["train"][idx]
        C = tune_svc(Gs, ys, G["val"][:3000], y["val"][:3000], [1, 10])
        evaluate("rbf_svm_hog", SVC(kernel="rbf", C=C, gamma="scale"), Gs, ys, G["val"], y["val"],
                 G["test"], y["test"], results, extra={"C": C, "train_subsample": int(len(idx))})

    best = max((k for k in results if "val_acc" in results[k]), key=lambda k: results[k]["val_acc"])
    results["_best_by_val"] = best
    save_json(results, args.out)
    print(f"best baseline by val acc: {best} -> {args.out}")


if __name__ == "__main__":
    main()

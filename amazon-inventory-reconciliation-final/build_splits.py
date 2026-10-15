"""Create a stratified 70/20/10 train/val/test split from data/labels.csv -> data/splits.csv."""
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from abid.common import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.data, "labels.csv"))
    train, rest = train_test_split(df, test_size=0.30, stratify=df.qty, random_state=args.seed)
    val, test = train_test_split(rest, test_size=1 / 3, stratify=rest.qty, random_state=args.seed)
    df = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    df = df.sort_values("id").reset_index(drop=True)
    df.to_csv(os.path.join(args.data, "splits.csv"), index=False)

    stats = {
        "n_images": int(len(df)),
        "split_sizes": {s: int((df.split == s).sum()) for s in ("train", "val", "test")},
        "class_counts": {str(q): int((df.qty == q).sum()) for q in range(6)},
        "majority_class_share": float(df.qty.value_counts(normalize=True).max()),
    }
    save_json(stats, os.path.join(args.data, "stats.json"))
    print(stats)


if __name__ == "__main__":
    main()

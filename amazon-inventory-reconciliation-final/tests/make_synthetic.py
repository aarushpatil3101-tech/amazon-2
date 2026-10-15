"""Generate a tiny SYNTHETIC bin-counting dataset (k random boxes on a noisy shelf) so the whole
pipeline can be smoke-tested without downloading ABID.  Numbers from this data are meaningless.

Usage: python tests/make_synthetic.py --out data_smoke --n 600
"""
import argparse
import csv
import os

import numpy as np
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_smoke")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    rows = []
    for i in range(1, a.n + 1):
        k = int(rng.randint(0, 6))
        bg = np.clip(rng.normal(170, 12, (a.size, a.size, 3)) + np.array([10, 0, -15]), 0, 255).astype(np.uint8)
        im = Image.fromarray(bg)
        d = ImageDraw.Draw(im)
        for _ in range(k):
            w, h = rng.randint(30, 70, 2)
            x, y = rng.randint(0, a.size - w), rng.randint(0, a.size - h)
            col = tuple(int(c) for c in rng.randint(40, 120, 3))
            d.rectangle([x, y, x + w, y + h], fill=col)
        im.save(os.path.join(a.out, "images", f"{i}.jpg"), quality=90)
        rows.append((i, k))
    with open(os.path.join(a.out, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "qty"])
        w.writerows(rows)
    print(f"wrote {a.n} synthetic images to {a.out}")


if __name__ == "__main__":
    main()

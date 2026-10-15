"""Download a random subset of the Amazon Bin Image Dataset (ABID) over plain HTTPS.

Public bucket (no AWS account needed): https://aft-vbi-pds.s3.amazonaws.com
  bin-images/<id>.jpg   photo of a bin
  metadata/<id>.json    JSON with EXPECTED_QUANTITY (the label = number of items in the bin)

Strategy: shuffle the ids 1..535234 with a fixed seed, read each metadata file, keep bins with
0..MAX_ITEMS items, download + resize the image to SIZE x SIZE, stop at N images.
Resumable: re-running continues from data/labels.csv.

Usage: python download_data.py --n 30000 --out data
"""
import argparse
import csv
import io
import os
import random
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

BASE = "https://aft-vbi-pds.s3.amazonaws.com"
MAX_ID = 535234


def make_session(pool):
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=retry))
    return s


def process(i, sess, img_dir, size, max_items):
    """Return (id, qty) if the bin qualifies and its image was saved, else None."""
    try:
        r = sess.get(f"{BASE}/metadata/{i}.json", timeout=20)
        if r.status_code != 200:
            return None
        qty = int(r.json()["EXPECTED_QUANTITY"])
        if not 0 <= qty <= max_items:
            return None
        out = os.path.join(img_dir, f"{i}.jpg")
        if not os.path.exists(out):
            r = sess.get(f"{BASE}/bin-images/{i}.jpg", timeout=30)
            if r.status_code != 200:
                return None
            im = Image.open(io.BytesIO(r.content)).convert("RGB").resize((size, size), Image.BILINEAR)
            im.save(out, quality=92)
        return i, qty
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000, help="number of qualifying images to keep")
    ap.add_argument("--out", default="data")
    ap.add_argument("--size", type=int, default=224, help="stored square image size")
    ap.add_argument("--max-items", type=int, default=5)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    label_path = os.path.join(args.out, "labels.csv")

    kept = {}
    if os.path.exists(label_path):
        with open(label_path) as f:
            for row in csv.DictReader(f):
                kept[int(row["id"])] = int(row["qty"])
        print(f"resuming: {len(kept)} images already labelled")

    ids = list(range(1, MAX_ID + 1))
    random.Random(args.seed).shuffle(ids)
    sess = make_session(args.workers)

    pos = 0
    pbar = tqdm(total=args.n, initial=min(len(kept), args.n), desc="images kept")
    with ThreadPoolExecutor(args.workers) as ex:
        while len(kept) < args.n and pos < len(ids):
            chunk = [i for i in ids[pos:pos + args.chunk] if i not in kept]
            pos += args.chunk
            for res in ex.map(lambda i: process(i, sess, img_dir, args.size, args.max_items), chunk):
                if res and len(kept) < args.n:
                    kept[res[0]] = res[1]
                    pbar.update(1)
            with open(label_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "qty"])
                for k, v in kept.items():
                    w.writerow([k, v])
    pbar.close()

    counts = {q: sum(1 for v in kept.values() if v == q) for q in range(args.max_items + 1)}
    print(f"done: {len(kept)} images -> {img_dir}\nclass counts (items: images): {counts}")


if __name__ == "__main__":
    main()

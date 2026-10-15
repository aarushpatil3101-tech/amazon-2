"""Transfer-learning ResNet for bin item counting (0..5).

Heads
  cls      : 6-way softmax + cross-entropy (the paper's formulation)
  ordinal  : 5 cumulative binary outputs P(count > k), BCE loss (Frank & Hall style); count = #(P>0.5)

Schedule
  stage 1  : backbone frozen, train the new head (cosine)
  stage 2  : unfreeze all, discriminative LRs (early=lr/9, middle=lr/3, head=lr), AdamW,
             cosine annealing with warm restarts (SGDR), optional LR-finder for `lr`
  eval     : best-val checkpoint, hflip test-time augmentation, accuracy / RMSE / confusion matrix

Usage:
  python train_cnn.py --data data --arch resnet34 --head cls --name r34_cls
  python train_cnn.py --data data --arch resnet34 --head ordinal --name r34_ord
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from abid.common import NUM_CLASSES, compute_metrics, load_splits, save_json, set_seed
from abid.lrfinder import lr_find

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class BinDataset(Dataset):
    def __init__(self, df, img_dir, tf):
        self.ids, self.y, self.dir, self.tf = df.id.values, df.qty.values, img_dir, tf

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        im = Image.open(os.path.join(self.dir, f"{self.ids[i]}.jpg")).convert("RGB")
        return self.tf(im), int(self.y[i])


def build_model(arch, head, pretrained):
    m = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1" if pretrained else None)
    m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, NUM_CLASSES if head == "cls" else NUM_CLASSES - 1))
    return m


def param_groups(model, lr, wd):
    early = [p for n, p in model.named_parameters() if n.split(".")[0] in ("conv1", "bn1", "layer1", "layer2")]
    mid = [p for n, p in model.named_parameters() if n.split(".")[0] in ("layer3", "layer4")]
    head = list(model.fc.parameters())
    return [{"params": early, "lr": lr / 9, "weight_decay": wd},
            {"params": mid, "lr": lr / 3, "weight_decay": wd},
            {"params": head, "lr": lr, "weight_decay": wd}]


def make_loss(head):
    if head == "cls":
        return lambda out, y: F.cross_entropy(out, y)

    def ordinal_loss(out, y):
        t = (y.unsqueeze(1) > torch.arange(NUM_CLASSES - 1, device=y.device).unsqueeze(0)).float()
        return F.binary_cross_entropy_with_logits(out.float(), t)
    return ordinal_loss


# ---- native outputs (softmax / sigmoid) -> predictions & class probabilities -----------------
def to_native(logits, head):
    return logits.softmax(1) if head == "cls" else logits.sigmoid()


def native_to_pred(native, head):
    if head == "cls":
        return native.argmax(1)
    mono = torch.cummin(native, dim=1).values            # enforce P(>k) non-increasing in k
    return (mono > 0.5).sum(1)


def native_to_classprobs(native, head):
    if head == "cls":
        return native
    mono = torch.cummin(native, dim=1).values
    z = torch.ones(mono.size(0), 1, device=mono.device)
    zero = torch.zeros(mono.size(0), 1, device=mono.device)
    p = torch.cat([z, mono], 1) - torch.cat([mono, zero], 1)
    p = p.clamp_min(1e-6)
    return p / p.sum(1, keepdim=True)


@torch.no_grad()
def predict(model, loader, head, device, tta=False, amp=False):
    model.eval()
    outs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            nat = to_native(model(x).float(), head)
            if tta:
                nat = (nat + to_native(model(torch.flip(x, dims=[3])).float(), head)) / 2
        outs.append(nat.cpu())
        ys.append(y)
    return torch.cat(outs), torch.cat(ys)


def run_epoch(model, loader, loss_fn, opt, scaler, device, amp, sched=None, epoch=0, head="cls", train=True):
    model.train(train)
    tot, n, correct = 0.0, 0, 0
    n_it = len(loader)
    for it, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(x)
            loss = loss_fn(out, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if sched is not None:
            sched(epoch + (it + 1) / n_it)
        tot += loss.item() * len(y)
        n += len(y)
        correct += (native_to_pred(to_native(out.detach().float(), head), head) == y).sum().item()
    return tot / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--arch", default="resnet34", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--head", default="cls", choices=["cls", "ordinal"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--head-epochs", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=9, help="stage-2 epochs (SGDR cycles 3+6 by default)")
    ap.add_argument("--t0", type=int, default=3, help="SGDR first cycle length in epochs")
    ap.add_argument("--lr", type=float, default=1e-3, help="head LR (backbone gets lr/3 and lr/9)")
    ap.add_argument("--lr-find", action="store_true")
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true", help="smoke tests only")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap images per split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    name = args.name or f"{args.arch}_{args.head}"
    run_dir = os.path.join("runs", name)
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    print(f"run {name} | device {device} | amp {amp}")

    sp = load_splits(args.data)
    if args.limit:
        sp = {k: v.head(args.limit) for k, v in sp.items()}
    img_dir = os.path.join(args.data, "images")
    resize = [T.Resize((args.img, args.img))] if args.img != 224 else []
    tf_train = T.Compose(resize + [T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2), T.ToTensor(), T.Normalize(MEAN, STD)])
    tf_eval = T.Compose(resize + [T.ToTensor(), T.Normalize(MEAN, STD)])
    mk = lambda s, tf, shuf: DataLoader(BinDataset(sp[s], img_dir, tf), batch_size=args.bs, shuffle=shuf,
                                        num_workers=args.workers, pin_memory=device.type == "cuda",
                                        drop_last=shuf, persistent_workers=args.workers > 0)
    dl = {"train": mk("train", tf_train, True), "val": mk("val", tf_eval, False), "test": mk("test", tf_eval, False)}

    model = build_model(args.arch, args.head, not args.no_pretrained).to(device)
    loss_fn = make_loss(args.head)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, best_val, t_start = [], -1.0, time.time()
    ckpt = os.path.join(run_dir, "best.pt")

    def log(stage, ep, tr_loss, tr_acc, lr):
        nat, y = predict(model, dl["val"], args.head, device, amp=amp)
        vm = compute_metrics(y.numpy(), native_to_pred(nat, args.head).numpy())
        history.append({"stage": stage, "epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_acc": vm["acc"], "val_rmse": vm["rmse"], "lr": lr})
        print(f"[{stage}] ep {ep:2d} | loss {tr_loss:.3f} train acc {tr_acc:.3f} | val acc {vm['acc']:.3f} rmse {vm['rmse']:.3f}")
        return vm["acc"]

    # ---------------- stage 1: head only ----------------
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.fc.parameters(), lr=args.lr, weight_decay=args.wd)
    n1 = max(args.head_epochs, 1)
    set_lr = lambda e: [g.update(lr=0.5 * args.lr * (1 + np.cos(np.pi * min(e / n1, 1)))) for g in opt.param_groups]
    for ep in range(args.head_epochs):
        l, a = run_epoch(model, dl["train"], loss_fn, opt, scaler, device, amp, set_lr, ep, args.head)
        v = log("head", ep + 1, l, a, opt.param_groups[0]["lr"])
        if v > best_val:
            best_val = v
            torch.save(model.state_dict(), ckpt)

    # ---------------- stage 2: full fine-tune ----------------
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(param_groups(model, args.lr, args.wd))
    lrfind_info = None
    if args.lr_find:
        lrs, losses, sug = lr_find(model, opt, dl["train"], loss_fn, device, amp=amp)
        args.lr = float(np.clip(sug, 1e-5, 1e-3))   # cap: backbone gets lr/3 and lr/9 under AdamW
        opt = torch.optim.AdamW(param_groups(model, args.lr, args.wd))
        lrfind_info = {"lrs": lrs, "losses": losses, "suggested": sug, "used": args.lr}
        save_json(lrfind_info, os.path.join(run_dir, "lr_find.json"))
        print(f"LR finder: suggested {sug:.2e} -> using head lr {args.lr:.2e}")
    base = [g["lr"] for g in opt.param_groups]

    def sgdr(e):                                           # e = fractional epoch
        t0, T_i, start = args.t0, args.t0, 0.0
        while e >= start + T_i:
            start += T_i
            T_i *= 2
        f = 0.5 * (1 + np.cos(np.pi * (e - start) / T_i))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * f

    for ep in range(args.epochs):
        l, a = run_epoch(model, dl["train"], loss_fn, opt, scaler, device, amp, sgdr, ep, args.head)
        v = log("full", ep + 1, l, a, opt.param_groups[-1]["lr"])
        if v > best_val:
            best_val = v
            torch.save(model.state_dict(), ckpt)
    train_seconds = time.time() - t_start

    # ---------------- final evaluation ----------------
    model.load_state_dict(torch.load(ckpt, map_location=device))
    out = {"name": name, "arch": args.arch, "head": args.head, "img": args.img, "epochs_head": args.head_epochs,
           "epochs_full": args.epochs, "lr": args.lr, "lr_find": bool(args.lr_find), "train_seconds": round(train_seconds, 1),
           "n_train": len(sp["train"]), "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}
    for split in ("val", "test"):
        for tta in (False, True):
            nat, y = predict(model, dl[split], args.head, device, tta=tta, amp=amp)
            key = f"{split}{'_tta' if tta else ''}"
            out[key] = compute_metrics(y.numpy(), native_to_pred(nat, args.head).numpy())
            if split == "test" and tta:
                np.save(os.path.join(run_dir, "test_probs.npy"), native_to_classprobs(nat, args.head).numpy())
                pd.DataFrame({"id": sp["test"].id.values, "y": y.numpy(),
                              "pred": native_to_pred(nat, args.head).numpy()}).to_csv(os.path.join(run_dir, "test_preds.csv"), index=False)
            print(f"{key:9s} acc {out[key]['acc']:.4f} rmse {out[key]['rmse']:.3f} mae {out[key]['mae']:.3f} within1 {out[key]['within1']:.3f}")

    # single-image latency (batch 1)
    x = torch.randn(1, 3, args.img, args.img, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(30):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    out["latency_ms_bs1"] = round((time.time() - t) / 30 * 1000, 2)

    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    save_json(out, os.path.join(run_dir, "metrics.json"))
    print(f"saved {run_dir}/metrics.json")


if __name__ == "__main__":
    main()

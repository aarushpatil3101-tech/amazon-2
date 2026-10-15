"""Learning-rate range test (Smith, 2017): sweep LR exponentially for one short pass, pick the
LR with the steepest loss decrease. Model/optimizer state is restored afterwards."""
import copy

import numpy as np
import torch


def lr_find(model, optimizer, loader, loss_fn, device, start=1e-6, end=1e-1, steps=100, beta=0.98,
            amp=False):
    model_state = copy.deepcopy(model.state_dict())
    opt_state = copy.deepcopy(optimizer.state_dict())
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    ratios = [lr / base_lrs[-1] for lr in base_lrs]       # keep discriminative ratios
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    model.train()

    lrs, losses, avg, best = [], [], 0.0, float("inf")
    mult = (end / start) ** (1 / max(steps - 1, 1))
    lr = start
    def forever():                      # itertools.cycle would cache every batch in RAM
        while True:
            for b in loader:
                yield b
    batches = forever()
    for it in range(steps):
        x, y = next(batches)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        for g, r in zip(optimizer.param_groups, ratios):
            g["lr"] = lr * r
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        avg = beta * avg + (1 - beta) * loss.item()
        smooth = avg / (1 - beta ** (it + 1))
        lrs.append(lr)
        losses.append(smooth)
        best = min(best, smooth)
        if smooth > 4 * best and it > 10:
            break
        lr *= mult

    model.load_state_dict(model_state)
    optimizer.load_state_dict(opt_state)
    for g, lr0 in zip(optimizer.param_groups, base_lrs):
        g["lr"] = lr0

    lrs, losses = np.array(lrs), np.array(losses)
    # steepest negative slope of loss vs log10(lr), ignoring the noisy first/last points
    k = max(3, len(lrs) // 20)
    grad = np.gradient(losses[k:-k] if len(lrs) > 2 * k + 3 else losses, np.log10(lrs[k:-k] if len(lrs) > 2 * k + 3 else lrs))
    off = k if len(lrs) > 2 * k + 3 else 0
    suggestion = float(lrs[off + int(np.argmin(grad))])
    return lrs.tolist(), losses.tolist(), suggestion

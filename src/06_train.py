"""Step 6 - Train BeltPulse-Net and the baselines on the cached ResNet features.

Each training round ("epoch") shows the model every training example once:
  model guesses a score per square -> compare with the 0/1 answer grid (loss) -> nudge weights to do better.
After each epoch we check on the validation set and keep the best version.

Loss = BCE (per-square right/wrong, object squares weighted x5 because they are rare)
     + Dice (overlap between predicted and true object area; insensitive to the 94% easy ore squares)

Usage:
  .venv\\Scripts\\python src\\06_train.py --limit 100                       # smoke test (all 6 modes), 3 views
  .venv\\Scripts\\python src\\06_train.py --limit 100 --variant T8w7        # smoke test, 8 views + 7x7 window
  .venv\\Scripts\\python src\\06_train.py --modes beltpulse single           # full data, selected modes

Output:
  data/models/<tag>/<mode>.pt
  outputs/06_train/<tag>/curves.png, log.json
"""
import os
import json
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT, run_tag, VARIANTS
from dataset import FeatureSamples
from model import build_model, MODES

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--variant", default="base", choices=list(VARIANTS))
ap.add_argument("--modes", nargs="+", default=MODES)
ap.add_argument("--epochs", type=int, default=0, help="0 = 25 for smoke test, 30 for full")
ap.add_argument("--batch", type=int, default=0, help="0 = 8 for base, 4 for bigger variants (GPU memory)")
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--seed", type=int, default=0, help="random start; seed N>0 saves to <run>_sN so seed 0 is kept")
args = ap.parse_args()
data_tag = run_tag(args.limit, args.variant)                   # which samples (subset file)
tag = data_tag + (f"_s{args.seed}" if args.seed else "")       # where models / curves go
cfg = VARIANTS[args.variant]
args.batch = args.batch or (8 if cfg["T"] * cfg["win"] ** 2 <= 27 else 4)
epochs = args.epochs or (25 if args.limit else 30)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)


def loss_fn(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(5.0, device=logits.device))
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
    return bce + dice


def run_epoch(model, loader, opt=None):
    model.train(opt is not None)
    tot, n, inter, union = 0.0, 0, 0.0, 0.0
    with torch.set_grad_enabled(opt is not None):
        for feats, shift, grid, _, white in loader:
            feats, shift, grid = feats.to(dev), shift.to(dev), grid.to(dev)
            logits = model(feats, shift, white=white.to(dev))
            loss = loss_fn(logits, grid)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += loss.item() * len(feats)
            n += len(feats)
            pred = (logits > 0).float()
            inter += (pred * grid).sum().item()
            union += ((pred + grid) > 0).float().sum().item()
    return tot / n, inter / max(union, 1)


if __name__ == "__main__":
    subset = json.load(open(os.path.join(ROOT, "data", "samples", f"subset_{data_tag}.json")))
    train = DataLoader(FeatureSamples([s for s in subset if s["split"] == "train"], with_white=True),
                       batch_size=args.batch, shuffle=True, num_workers=0)
    val = DataLoader(FeatureSamples([s for s in subset if s["split"] == "val"], with_white=True), batch_size=args.batch)
    MDIR = os.path.join(ROOT, "data", "models", tag)
    OUT = os.path.join(ROOT, "outputs", "06_train", tag)
    os.makedirs(MDIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    print(f"run '{tag}': {len(train.dataset)} train / {len(val.dataset)} val samples, {cfg['T']} views, "
          f"{cfg['win']}x{cfg['win']} window, {epochs} epochs, batch {args.batch}, device {dev}")

    log = {}
    for mode in args.modes:
        model = build_model(mode, cfg).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        hist, best, t0 = [], -1, time.time()
        for ep in range(epochs):
            tr_loss, tr_iou = run_epoch(model, train, opt)
            va_loss, va_iou = run_epoch(model, val)
            sched.step()
            hist.append({"epoch": ep + 1, "train_loss": tr_loss, "val_loss": va_loss, "train_iou": tr_iou, "val_iou": va_iou})
            if va_iou > best:
                best = va_iou
                torch.save(model.state_dict(), os.path.join(MDIR, f"{mode}.pt"))
            el = time.time() - t0
            print(f"  [{mode} {args.modes.index(mode) + 1}/{len(args.modes)}] epoch {ep + 1:2d}/{epochs}  "
                  f"train loss {tr_loss:.3f}  val loss {va_loss:.3f}  val IoU {va_iou:.3f} (best {best:.3f})  "
                  f"{el / 60:.1f} min, ~{el / (ep + 1) * (epochs - ep - 1) / 60:.0f} min left for this model", flush=True)
        log[mode] = hist
        print(f"{mode:10s} best val object-IoU {best:.3f}   last train loss {tr_loss:.3f}   ({time.time() - t0:.0f}s)")

    prev = os.path.join(OUT, "log.json")
    if os.path.exists(prev):
        log = {**json.load(open(prev)), **log}
    json.dump(log, open(prev, "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for mode, hist in log.items():
        e = [h["epoch"] for h in hist]
        l, = ax[0].plot(e, [h["train_loss"] for h in hist], label=f"{mode} train")
        ax[0].plot(e, [h["val_loss"] for h in hist], "--", color=l.get_color(), label=f"{mode} val")
        ax[1].plot(e, [h["val_iou"] for h in hist], color=l.get_color(), label=mode)
    ax[0].set_title("loss (lower = better)"); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=7)
    ax[1].set_title("validation object IoU (higher = better)"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "curves.png"), dpi=90)
    print("curves in", OUT)

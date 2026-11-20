"""Step 7 - Final exam on the TEST videos (never seen during training) + pictures.

Metrics (computed over all grid squares of all test samples):
  object IoU  = overlap / union of predicted vs true object squares       (1.0 = perfect)
  mIoU        = mean of object IoU and background IoU                    (the proposal's metric)
  F1          = balance of precision (how many flagged squares are real) and recall (how many real ones were found)
  recall@glare= of the object squares hidden under glare in the target view t, how many were still found
Reported for: all test samples / clean target view / target view with glare, and per test video.
  AP          = average precision: area under the precision-recall curve (all thresholds, not only 0.5)

Voting baselines (E9, no extra training): the trained `single` model is run on EVERY view separately,
its probability grids are moved into view t's coordinates with the belt shift, then combined per square:
  OR       = max over the views          ("any view says impurity": fewest misses, false alarms add up)
  2-of-N   = 2nd-highest probability     (at least 2 views agree; only with more than 3 views)
  majority = (N//2+1)-th highest         (more than half of the views agree)
  mean     = average probability
Views where the square is outside the image are left out of the vote.

Attention check (BeltPulse): average attention weight a view receives on object squares
  - where that view is GLARED there   vs.   - where that view is clean there.
  If the transformer learned to distrust glare, the first is clearly smaller.

Usage:
  .venv\\Scripts\\python src\\07_evaluate.py --limit 100 [--variant T8w7]

Output: outputs/07_eval/<tag>/metrics.md, examples_XX.jpg, attention.png, pr_curves.png
"""
import os
import json
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (ROOT, run_tag, load_model_frame, put_label, MODEL_W, MODEL_H, VARIANTS, GRID_W, GRID_H,
                    frame_glare, target_glare, t_index, glare_grid, CELL_W_ORIG)
from dataset import FeatureSamples
from model import build_model, MODES, kinematic_align

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--variant", default="base", choices=list(VARIANTS))
ap.add_argument("--seed", type=int, default=0, help="evaluate the models trained with 06 --seed N (folder <run>_sN)")
args = ap.parse_args()
data_tag = run_tag(args.limit, args.variant)
tag = data_tag + (f"_s{args.seed}" if args.seed else "")
cfg = VARIANTS[args.variant]
dev = "cuda" if torch.cuda.is_available() else "cpu"

subset = json.load(open(os.path.join(ROOT, "data", "samples", f"subset_{data_tag}.json")))
test = [s for s in subset if s["split"] == "test"]
loader = DataLoader(FeatureSamples(test, with_white=True), batch_size=4)
OUT = os.path.join(ROOT, "outputs", "07_eval", tag)
os.makedirs(OUT, exist_ok=True)
T, TI = cfg["T"], (cfg["T"] - 1) // 2
VIEW_NAMES = [f"{j - TI:+d}k" if j != TI else "t" for j in range(T)]


def load(mode):
    path = os.path.join(ROOT, "data", "models", tag, f"{mode}.pt")
    if not os.path.exists(path):
        print(f"  (no trained '{mode}' model in run '{tag}', skipped)")
        return None
    model = build_model(mode, cfg).to(dev)
    model.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    return model.eval()


@torch.no_grad()
def predict(model):
    probs, attns = [], []
    for feats, shift, _, _, white in loader:
        logits, attn = model(feats.to(dev), shift.to(dev), return_attn=True, white=white.to(dev))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        attns.append(attn.cpu().numpy())
    return np.concatenate(probs), np.concatenate(attns)


VOTE_RULES = ["OR", "majority", "mean"] if T <= 3 else ["OR", "2-of-N", "majority", "mean"]


@torch.no_grad()
def predict_votes(single):
    """Run `single` on each view alone, align the T probability grids to view t, combine with each rule."""
    out = {r: [] for r in VOTE_RULES}
    for feats, shift, _, _, _ in loader:
        feats, shift = feats.to(dev), shift.to(dev)
        per_view = []
        for j in range(T):
            fj = feats.clone()
            fj[:, TI] = feats[:, j]                       # `single` only looks at slot TI: put view j there
            per_view.append(torch.sigmoid(single(fj, shift)))
        p = torch.stack(per_view, 1).unsqueeze(2)         # (B, T, 1, H, W), each in its own view's coordinates
        p, valid = kinematic_align(p, shift)              # -> view t's coordinates
        p = p[:, :, 0]                                    # (B, T, H, W)
        valid = valid.unsqueeze(2).expand_as(p)           # (B, T, H, W)
        n = valid.sum(1)                                  # number of views that see this square
        ranked = torch.where(valid, p, torch.full_like(p, -1.0)).sort(1, descending=True).values

        def kth(k):                                       # k-th highest probability among the valid views
            return ranked.gather(1, (torch.minimum(k, n) - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
        rules = {"OR": ranked[:, 0], "2-of-N": kth(torch.full_like(n, 2)), "majority": kth(n // 2 + 1),
                 "mean": (p * valid).sum(1) / n.clamp(min=1)}
        for r in VOTE_RULES:
            out[r].append(rules[r].cpu().numpy())
    return {r: np.concatenate(v) for r, v in out.items()}


def average_precision(score, true):
    """Area under the precision-recall curve over all squares (sorted by score)."""
    order = np.argsort(-score.ravel(), kind="stable")
    t = true.ravel()[order]
    tp = np.cumsum(t); fp = np.cumsum(~t)
    prec, rec = tp / np.maximum(tp + fp, 1), tp / max(t.sum(), 1)
    return float(np.sum(np.diff(np.concatenate([[0], rec])) * prec)), prec, rec


def scores(pred, true):
    tp = (pred & true).sum(); fp = (pred & ~true).sum(); fn = (~pred & true).sum(); tn = (~pred & ~true).sum()
    iou_obj = tp / max(tp + fp + fn, 1)
    iou_bg = tn / max(tn + fp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return iou_obj, (iou_obj + iou_bg) / 2, f1


def aligned_glare(s, slot):
    """Glare mask of view `slot`, moved into the target view's coordinates (where the attention looks)."""
    g = frame_glare(s, slot)
    if not g:
        return np.zeros((GRID_H, GRID_W), bool)
    gg = glare_grid(g).astype(bool)
    sh = int(round(s["shift_px"][slot] / CELL_W_ORIG))
    out = np.zeros_like(gg)
    for c in range(GRID_W):
        if 0 <= c + sh < GRID_W:
            out[:, c] = gg[:, c + sh]
    return out


truth, gmask = [], []
for _, _, grid, gm, _ in loader:
    truth.append(grid.numpy() > 0.5)
    gmask.append(gm.numpy() > 0)
truth, gmask = np.concatenate(truth), np.concatenate(gmask)
glared = np.array([target_glare(s) is not None for s in test])

preds, attn_bp = {}, None
for mode in MODES:
    model = load(mode)
    if model is None:
        continue
    preds[mode], a = predict(model)
    if mode == "beltpulse":
        attn_bp = a                                         # (N, H, W, T)
    if mode == "single":
        for rule, p in predict_votes(model).items():
            preds[f"single + {rule} vote"] = p

videos = sorted({s["video"] for s in test})
vid = np.array([s["video"] for s in test])
results = {}
for name, p in preds.items():
    pb = p > 0.5
    row = {}
    for part, sel in [("all", np.ones(len(test), bool)), ("clean", ~glared), ("glare on t", glared)]:
        row[part] = scores(pb[sel], truth[sel])
    hidden = truth & gmask                                  # object squares covered by glare in view t
    row["recall@glare"] = (pb & hidden).sum() / max(hidden.sum(), 1)
    row["AP"] = average_precision(p, truth)[0]
    row["video"] = {v: scores(pb[vid == v], truth[vid == v])[0] for v in videos}
    row["false alarm squares"] = {v: int((pb[vid == v] & ~truth[vid == v]).sum()) for v in videos}
    results[name] = row

# attention a view gets where it is glared vs clean (object squares only)
if attn_bp is None:
    attn_bp = np.zeros((len(test), GRID_H, GRID_W, T))     # beltpulse not trained in this run
w_glared, w_clean = [], []
per_view = {v: ([], []) for v in range(T)}
for i, s in enumerate(test):
    for v in range(T):
        ag = aligned_glare(s, v)
        a = attn_bp[i, :, :, v]
        w_glared += list(a[truth[i] & ag]); w_clean += list(a[truth[i] & ~ag])
        per_view[v][0].extend(a[truth[i] & ag]); per_view[v][1].extend(a[truth[i] & ~ag])
uniform = 1 / T

# ---- table
lines = [f"# Test results — run `{tag}` ({T} views, {cfg['win']}x{cfg['win']} window)", "",
         f"{len(test)} test samples ({glared.sum()} with glare on the target view), "
         f"{int(truth.sum())} object squares, {int((truth & gmask).sum())} of them under glare in the target view.", "",
         "Models are trained; `single + ... vote` rows reuse the trained `single` model on every view (no training).", "",
         "| model | object IoU (all) | mIoU (all) | F1 (all) | AP (all) | object IoU clean | object IoU glare-on-t | recall under glare |",
         "|---|---|---|---|---|---|---|---|"]
for mode, r in results.items():
    lines.append(f"| {mode} | {r['all'][0]:.3f} | {r['all'][1]:.3f} | {r['all'][2]:.3f} | {r['AP']:.3f} | "
                 f"{r['clean'][0]:.3f} | {r['glare on t'][0]:.3f} | {r['recall@glare']:.3f} |")
short = [v.replace("boardcamera_", "") for v in videos]
lines += ["", "## Per test video",
          "`iron_ore_jaspilito_4` has real glare and motion blur. `iron_ore_4` is ore only, so its object IoU is 0 by definition:",
          "look at its false-alarm squares instead.", "",
          "| model (object IoU / false-alarm squares) | " + " | ".join(short) + " |", "|---|" + "---|" * len(videos)]
for mode, r in results.items():
    lines.append(f"| {mode} | " + " | ".join(f"{r['video'][v]:.3f} / {r['false alarm squares'][v]}" for v in videos) + " |")
lines += ["", "## Does BeltPulse distrust glared views?",
          f"Average attention weight a view receives on object squares (uniform = 1/{T} = {uniform:.3f}):", "",
          "| where the view is | avg weight | n |", "|---|---|---|",
          f"| glared | {np.mean(w_glared) if w_glared else float('nan'):.3f} | {len(w_glared)} |",
          f"| clean | {np.mean(w_clean):.3f} | {len(w_clean)} |"]
open(os.path.join(OUT, "metrics.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines).replace("—", "-"))


# ---- example pictures: first / target / last view | truth | single-frame | BeltPulse
def heat(img, prob, color=(0, 0, 255)):
    m = cv2.resize(prob.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    over = img.copy()
    over[m > 0.5] = color
    return cv2.addWeighted(over, 0.5, img, 0.5, 0)


order = [i for i in range(len(test)) if glared[i] and truth[i].any()] + [i for i in range(len(test)) if not glared[i] and truth[i].any()]
for n, i in enumerate(order[:6] if {"single", "beltpulse"} <= set(preds) else []):
    s = test[i]
    ims = []
    for slot in sorted({0, t_index(s), len(s["frames"]) - 1}):
        g = frame_glare(s, slot)
        im = load_model_frame(s["video"], s["frames"][slot], g)
        put_label(im, f"view {VIEW_NAMES[slot]}" + (" GLARE" if g else ""), (8, 30), 0.8)
        ims.append(cv2.resize(im, (MODEL_W // 2, MODEL_H // 2)))
    base = load_model_frame(s["video"], s["t"], target_glare(s))
    panels = [heat(base, truth[i].astype(float), (0, 200, 0)), heat(base, preds["single"][i]), heat(base, preds["beltpulse"][i])]
    for p_, name in zip(panels, ["truth", "single-frame", "BeltPulse"]):
        put_label(p_, name, (8, 30), 0.8)
    row = np.hstack(ims + [cv2.resize(p_, (MODEL_W // 2, MODEL_H // 2)) for p_ in panels])
    cv2.imwrite(os.path.join(OUT, f"examples_{n:02d}.jpg"), row)

# ---- attention per view: glared vs clean
fig, ax = plt.subplots(figsize=(max(6, T * 0.9), 4))
x = np.arange(T)
ax.bar(x - 0.2, [np.mean(per_view[v][1]) if per_view[v][1] else 0 for v in range(T)], 0.4, label="view is clean at this square")
ax.bar(x + 0.2, [np.mean(per_view[v][0]) if per_view[v][0] else 0 for v in range(T)], 0.4, label="view is GLARED at this square")
ax.axhline(uniform, color="gray", ls="--", label=f"uniform (1/{T})")
ax.set_xticks(x, [f"view {n}" for n in VIEW_NAMES])
ax.set_ylabel("average attention weight (object squares)")
ax.set_title("Which views does BeltPulse trust?")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "attention.png"), dpi=90)

# ---- precision-recall curves (all thresholds): a sorter picks its own operating point
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for k, (part, sel) in enumerate([("all test samples", np.ones(len(test), bool)), ("glare on view t", glared)]):
    for name, p in preds.items():
        ap_, prec, rec = average_precision(p[sel], truth[sel])
        ax[k].plot(rec, prec, "--" if "vote" in name else "-", lw=1.5, label=f"{name} (AP {ap_:.2f})")
    ax[k].set_xlabel("recall (share of impurity squares found)"); ax[k].set_ylabel("precision (flagged squares that are impurity)")
    ax[k].set_title(part); ax[k].set_xlim(0, 1); ax[k].set_ylim(0, 1.02); ax[k].legend(fontsize=7); ax[k].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "pr_curves.png"), dpi=90)
print("pictures in", OUT)

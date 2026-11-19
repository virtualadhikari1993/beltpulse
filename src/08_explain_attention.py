"""Step 8 - Look inside the transformer: for ONE square, which patches did it listen to?

For a chosen square of the target view t, BeltPulse's cross-frame attention compares it with the
win x win neighbourhood of that square in every one of the T aligned views (T * win² patches).
This script draws those patches, zoomed in, with the attention weight (in %) on each.

Extra for wide windows (E10): per view, the attention-weighted CENTRE of the window is drawn as a cyan dot
and printed as an offset in squares. If an object rolled/slid away from where physics put it, attention
should move towards where it really is, i.e. the offset shows the object's motion relative to the belt.

Squares shown:
  A) object square UNDER GLARE in view t   -> hope: low weight on view t, high on other views
  B) object square, view t clean
  C) ore square (no object)

Usage:  .venv\\Scripts\\python src\\08_explain_attention.py --limit 100 [--variant T8w7]
Output: outputs/08_attention/<tag>/square_XX_<kind>.jpg
"""
import os
import json
import argparse
import numpy as np
import cv2
import torch

from common import (ROOT, run_tag, frame_path, apply_glare, put_label, IMG_W, IMG_H, GRID_W, GRID_H, CELL_W_ORIG,
                    VARIANTS, frame_glare, target_glare, t_index)
from dataset import FeatureSamples
from model import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--variant", default="base", choices=list(VARIANTS))
ap.add_argument("--n", type=int, default=3, help="examples per kind")
args = ap.parse_args()
tag = run_tag(args.limit, args.variant)
cfg = VARIANTS[args.variant]
T, WIN = cfg["T"], cfg["win"]
R = WIN // 2
dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(ROOT, "outputs", "08_attention", tag)
os.makedirs(OUT, exist_ok=True)

CELL_H_ORIG = IMG_H / GRID_H
PANEL = 330                                   # drawn size of one view's window, pixels
ZOOM = PANEL / (WIN * CELL_W_ORIG)
TI = (T - 1) // 2
NAMES = [("view t" if v == TI else f"view {v - TI:+d}k") for v in range(T)]

subset = json.load(open(os.path.join(ROOT, "data", "samples", f"subset_{tag}.json")))
test = [s for s in subset if s["split"] == "test"]
ds = FeatureSamples(test)
model = build_model("beltpulse", cfg).to(dev)
model.load_state_dict(torch.load(os.path.join(ROOT, "data", "models", tag, "beltpulse.pt"), map_location=dev, weights_only=True))
model.eval()


def photo(s, slot):
    img = cv2.imread(frame_path(s["video"], s["frames"][slot]))
    g = frame_glare(s, slot)
    if g:
        img, _ = apply_glare(img, g)
    return img


def crop_window(img, r, c_float):
    """WIN x WIN squares around (r, c) in original pixels, c may be fractional (aligned position), zoomed."""
    x0, y0 = (c_float - R) * CELL_W_ORIG, (r - R) * CELL_H_ORIG
    w, h = WIN * CELL_W_ORIG, WIN * CELL_H_ORIG
    M = np.float32([[1, 0, -x0], [0, 1, -y0]])                     # sub-pixel shift, black outside image
    patch = cv2.warpAffine(img, M, (int(round(w)), int(round(h))), borderValue=(0, 0, 0))
    return cv2.resize(patch, (int(w * ZOOM), int(h * ZOOM)), interpolation=cv2.INTER_LINEAR)


def draw_weights(patch, w):
    """Grid lines + attention weight per patch; brighter yellow border = more attention; cyan dot = attention centre."""
    ph, pw = patch.shape[:2]
    cw, ch = pw / WIN, ph / WIN
    full = 3.0 / (T * WIN * WIN) * 4                                # this weight or more = full yellow
    for i in range(WIN * WIN):
        rr, cc = divmod(i, WIN)
        p1 = (int(cc * cw) + 1, int(rr * ch) + 1)
        p2 = (int((cc + 1) * cw) - 1, int((rr + 1) * ch) - 1)
        strength = min(1.0, w[i] / full)
        cv2.rectangle(patch, p1, p2, (0, int(255 * strength), int(255 * strength)), 1 + int(3 * strength))
        if w[i] * 100 >= 0.5:
            put_label(patch, f"{w[i] * 100:.0f}", (p1[0] + 3, p1[1] + int(ch * 0.55)), 0.4 if WIN > 3 else 0.7)
    if w.sum() > 0:
        rr, cc = np.divmod(np.arange(WIN * WIN), WIN)
        cy, cx = (w * rr).sum() / w.sum(), (w * cc).sum() / w.sum()
        cv2.circle(patch, (int((cx + 0.5) * cw), int((cy + 0.5) * ch)), 7, (255, 255, 0), -1)
        return patch, (cx - R, cy - R)
    return patch, (0.0, 0.0)


def explain(idx, r, c, kind, n):
    s = test[idx]
    feats, shift, grid, gmask = ds[idx]
    with torch.no_grad():
        logits, per_frame = model(feats[None].to(dev), shift[None].to(dev), return_attn=True)
    tok = model.last_token_attn[0, r, c].cpu().numpy()            # (T, WIN²)
    prob = torch.sigmoid(logits[0, r, c]).item()

    panels, offsets = [], []
    for slot in range(T):
        c_src = c + shift[slot].item()                              # where this square is in that view
        p, off = draw_weights(crop_window(photo(s, slot), r, c_src), tok[slot])
        offsets.append(off)
        head = np.full((56, p.shape[1], 3), 40, np.uint8)
        g = " GLARE" if frame_glare(s, slot) else ""
        put_label(head, f"{NAMES[slot]}{g}: {tok[slot].sum() * 100:.0f}%", (6, 24), 0.6)
        put_label(head, f"centre offset dx {off[0]:+.1f} dy {off[1]:+.1f}", (6, 48), 0.5)
        panels.append(np.vstack([head, p]))
    gap_v = np.full((panels[0].shape[0], 8, 3), 255, np.uint8)
    per_row = 4 if T > 4 else T
    rows = []
    for i in range(0, T, per_row):
        row = panels[i:i + per_row]
        while len(row) < per_row:
            row.append(np.full_like(panels[0], 255))
        rows.append(np.hstack(sum([[p, gap_v] for p in row], [])[:-1]))
    zoom = np.vstack(sum([[rw, np.full((8, rows[0].shape[1], 3), 255, np.uint8)] for rw in rows], [])[:-1])

    over = photo(s, t_index(s))
    x0, y0 = int(c * CELL_W_ORIG), int(r * CELL_H_ORIG)
    cv2.rectangle(over, (x0, y0), (int(x0 + CELL_W_ORIG), int(y0 + CELL_H_ORIG)), (0, 0, 255), 4)
    cv2.rectangle(over, (int(x0 - R * CELL_W_ORIG), int(y0 - R * CELL_H_ORIG)),
                  (int(x0 + (R + 1) * CELL_W_ORIG), int(y0 + (R + 1) * CELL_H_ORIG)), (0, 255, 255), 2)
    truth = "impurity" if grid[r, c] > 0 else "ore"
    put_label(over, f"square row {r}, col {c} | truth: {truth} | model: {prob * 100:.0f}% impurity", (10, 40), 1.0)
    over = cv2.resize(over, (zoom.shape[1], int(IMG_H * zoom.shape[1] / IMG_W)))
    cv2.imwrite(os.path.join(OUT, f"square_{n:02d}_{kind}.jpg"), np.vstack([over, zoom]))
    return tok.sum(1), offsets


# ---- pick squares
rng = np.random.default_rng(0)
picks = {"A_glare": [], "B_clean": [], "C_ore": []}
for i, s in enumerate(test):
    _, _, grid, gmask = ds[i]
    grid, gmask = grid.numpy() > 0, gmask.numpy() > 0
    shift = np.array(s["shift_px"]) / CELL_W_ORIG
    ok = np.zeros_like(grid)                     # keep away from borders so the whole window is inside every view
    lo, hi = R + int(np.ceil(max(0, -shift.min()))), GRID_W - 1 - R - int(np.ceil(max(0, shift.max())))
    if hi > lo:
        ok[R:GRID_H - R, lo:hi] = True
    for kind, sel in [("A_glare", grid & gmask & ok), ("B_clean", grid & ~gmask & ok),
                      ("C_ore", ~grid & ok & (target_glare(s) is None))]:
        rc = np.argwhere(sel)
        if len(rc):
            picks[kind].append((i, *rc[rng.integers(len(rc))]))

n = 0
for kind, lst in picks.items():
    for idx, r, c in [lst[j] for j in rng.permutation(len(lst))[:args.n]]:
        w, offs = explain(idx, int(r), int(c), kind, n)
        print(f"square_{n:02d}_{kind}: attention per view " + " ".join(f"{x:.2f}" for x in w))
        n += 1
print("pictures in", OUT)

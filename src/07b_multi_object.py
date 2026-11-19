"""Step 7b - Several objects in one photo: what does the model do?

The model labels every square "impurity" or "ore", so several objects simply give several red regions.
It does NOT know which red squares belong to which object. Here we split the predicted grid into
separate blobs (connected components: touching red squares = one blob) and compare the blob count
with the true number of objects.

Usage:  .venv\\Scripts\\python src\\07b_multi_object.py --limit 100 [--mode beltpulse]
Output: outputs/07_eval/<tag>/multi_object_XX.jpg, multi_object.md
"""
import os
import json
import argparse
import numpy as np
import cv2
import torch

from common import ROOT, run_tag, frame_path, apply_glare, put_label, IMG_W, IMG_H, GRID_W, GRID_H, VARIANTS, target_glare
from dataset import FeatureSamples
from model import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--variant", default="base", choices=list(VARIANTS))
ap.add_argument("--mode", default="beltpulse")
args = ap.parse_args()
tag = run_tag(args.limit, args.variant)
dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(ROOT, "outputs", "07_eval", tag)
os.makedirs(OUT, exist_ok=True)

subset = json.load(open(os.path.join(ROOT, "data", "samples", f"subset_{tag}.json")))
test = [s for s in subset if s["split"] == "test" and s["boxes"]]
ds = FeatureSamples(test)
model = build_model(args.mode, VARIANTS[args.variant]).to(dev)
model.load_state_dict(torch.load(os.path.join(ROOT, "data", "models", tag, f"{args.mode}.pt"), map_location=dev, weights_only=True))
model.eval()

BLOB_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
MIN_BLOB = 2  # ignore single stray squares


def blobs(grid01):
    n, lab = cv2.connectedComponents(grid01.astype(np.uint8), connectivity=8)
    keep = [k for k in range(1, n) if (lab == k).sum() >= MIN_BLOB]
    return lab, keep


rows, n_img = [], 0
for i, s in enumerate(test):
    feats, shift, grid, _ = ds[i]
    with torch.no_grad():
        prob = torch.sigmoid(model(feats[None].to(dev), shift[None].to(dev)))[0].cpu().numpy()
    lab, keep = blobs(prob > 0.5)
    rows.append((len(s["boxes"]), len(keep)))
    if len(s["boxes"]) < 2 or n_img >= 6:
        continue
    img = cv2.imread(frame_path(s["video"], s["t"]))
    if target_glare(s):
        img, _ = apply_glare(img, target_glare(s))
    truth = img.copy()
    for x1, y1, x2, y2, c in s["boxes"]:
        cv2.rectangle(truth, (x1, y1), (x2, y2), (0, 200, 0), 5)
        put_label(truth, c, (x1 + 6, y1 + 32), 0.9)
    put_label(truth, f"truth: {len(s['boxes'])} objects", (10, 45), 1.3)
    pred = img.copy()
    over = img.copy()
    cw, ch = IMG_W / GRID_W, IMG_H / GRID_H
    for j, k in enumerate(keep):
        for r, c in zip(*np.nonzero(lab == k)):
            cv2.rectangle(over, (int(c * cw), int(r * ch)), (int((c + 1) * cw), int((r + 1) * ch)), BLOB_COLORS[j % 6], -1)
    pred = cv2.addWeighted(over, 0.5, pred, 0.5, 0)
    put_label(pred, f"{args.mode}: {len(keep)} separate regions (1 colour each)", (10, 45), 1.3)
    out = np.hstack([cv2.resize(truth, (640, 360)), cv2.resize(pred, (640, 360))])
    cv2.imwrite(os.path.join(OUT, f"multi_object_{n_img:02d}.jpg"), out)
    n_img += 1

r = np.array(rows)
lines = [f"# Several objects per photo — run `{tag}`, model `{args.mode}`", "",
         "| true objects | test samples | avg. predicted regions | exact count correct |", "|---|---|---|---|"]
for n in sorted(set(r[:, 0])):
    sel = r[r[:, 0] == n]
    lines.append(f"| {n} | {len(sel)} | {sel[:, 1].mean():.2f} | {(sel[:, 1] == n).mean() * 100:.0f}% |")
open(os.path.join(OUT, "multi_object.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"{n_img} pictures in {OUT}")

"""Step 2 - Measure how far the belt moves between every pair of consecutive frames.

This is the "kinematic" part of BeltPulse-Net. Instead of tracking objects with a neural
network, we use the fact that everything on the belt moves together: if we know the belt
moved dx pixels between frame t and t+1, an object at x in frame t is at x+dx in frame t+1.

How: phase correlation (cv2.phaseCorrelate) finds the single translation that best aligns
two images. We only use the belt area (rows BELT_Y0..BELT_Y1) because the steel frame at the
top/bottom of the image does not move and would pull the estimate towards 0.

Output:
  data/meta/motion/<video>.csv      frame_id, dx, dy, confidence  (dx of frame i -> i+1)
  outputs/02_belt_motion/dx_<video>.png   dx over time (is the belt speed constant?)
  outputs/02_belt_motion/summary.txt

Run:  .venv\\Scripts\\python src\\02_belt_motion.py
"""
import os
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from common import DATA, ROOT, videos, frames_of, BELT_Y0, BELT_Y1

OUT = os.path.join(ROOT, "outputs", "02_belt_motion")
META = os.path.join(ROOT, "data", "meta", "motion")
os.makedirs(OUT, exist_ok=True)
os.makedirs(META, exist_ok=True)

SCALE = 0.5  # measure on half-resolution images (4x faster, dx is multiplied back)


def load(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)[BELT_Y0:BELT_Y1]
    return cv2.resize(g, None, fx=SCALE, fy=SCALE).astype(np.float32)


summary = []
for v in videos():
    fs = frames_of(v)
    rows = []
    prev = load(fs[0][1])
    win = cv2.createHanningWindow(prev.shape[::-1], cv2.CV_32F)
    for (fid, path), (fid_next, path_next) in tqdm(list(zip(fs, fs[1:])), desc=v, leave=False):
        cur = load(path_next)
        if fid_next != fid + 1:          # a gap in the frame numbering: skip
            prev = cur
            continue
        (dx, dy), conf = cv2.phaseCorrelate(prev, cur, win)
        rows.append((fid, dx / SCALE, dy / SCALE, conf))
        prev = cur

    with open(os.path.join(META, f"{v}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "dx", "dy", "conf"])
        w.writerows(rows)

    a = np.array(rows)
    med, std = np.median(a[:, 1]), a[:, 1].std()
    summary.append(f"{v:36s} median dx {med:7.2f} px/frame   std {std:5.2f}   median dy {np.median(a[:, 2]):5.2f}")

    plt.figure(figsize=(10, 3))
    plt.plot(a[:, 0], a[:, 1], lw=0.6)
    plt.axhline(med, color="r", ls="--", label=f"median {med:.1f}")
    plt.xlabel("frame id"); plt.ylabel("dx (px / frame)"); plt.title(v); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"dx_{v}.png"), dpi=80); plt.close()

text = "\n".join(summary)
print(text)
open(os.path.join(OUT, "summary.txt"), "w").write(text + "\n")

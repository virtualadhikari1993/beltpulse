"""Step 4b - Training examples with MANY views (variant T8w7, extensions E10 + E11).

Like step 04, but each example has T=8 views spread along the object's path instead of 3:
    frames = t + j*k  for j = -3 .. +4         (target t is view index 3)
k is chosen so the belt moves ~STEP_PX between neighbouring views (fast belt -> small k).
Over 8 views the object travels ~7 x 64 = 450 px, so in the outer views it is often partly
out of the picture: some views are simply less useful, which is what attention should learn to handle.

Glare: every view independently gets a glare blob with probability P_GLARE (so 0, 1, 2 ... views can be
blinded). In the test split, half of the examples additionally always have glare on the target view t.

Output: data/samples/samples_T8.json  (same format as samples.json, plus "glares": one entry per view, "t_idx")
        outputs/04_samples/T8_sample_XX.jpg

Run:  .venv\\Scripts\\python src\\04b_build_samples_multi.py
"""
import os
import json
import random
import cv2
import numpy as np

from common import (ROOT, frames_of, load_motion, cumulative_shift, boxes_to_grid, load_model_frame,
                    split_of, NORMAL_VIDEOS, IMG_W, BELT_Y0, BELT_Y1, put_label, MODEL_W, MODEL_H, GRID_W, GRID_H)

random.seed(0)
T = 8
OFFSETS = list(range(-((T - 1) // 2), T - (T - 1) // 2))   # -3 .. 4 (8 views)
T_IDX = OFFSETS.index(0)
STEP_PX = 64
MAX_K = 12
POS_STRIDE, NEG_STRIDE = 2, 6
P_GLARE = 0.3
P_GLARE_ON_OBJECT = 0.7

SAMP = os.path.join(ROOT, "data", "samples")
OUT = os.path.join(ROOT, "outputs", "04_samples")
boxes = json.load(open(os.path.join(ROOT, "data", "labels", "boxes.json")))


def glare_at(obj_boxes):
    if obj_boxes and random.random() < P_GLARE_ON_OBJECT:
        x1, y1, x2, y2, _ = random.choice(obj_boxes)
        cx, cy = random.uniform(max(0, x1), min(IMG_W, x2)), random.uniform(y1, y2)
    else:
        cx, cy = random.uniform(100, IMG_W - 100), random.uniform(BELT_Y0 + 80, BELT_Y1 - 80)
    return {"cx": round(cx), "cy": round(cy), "r": random.randint(90, 220), "a": round(random.uniform(0.7, 1.0), 2)}


def choose_k(motion, fid):
    for k in range(1, MAX_K + 1):
        if abs(cumulative_shift(motion, fid, fid + k)) >= STEP_PX:
            return k
    return MAX_K


def make_sample(video, fid, motion, available, bxs):
    k = choose_k(motion, fid)
    frames = [fid + j * k for j in OFFSETS]
    if any(f not in available for f in frames):
        return None
    shifts = [round(cumulative_shift(motion, fid, f), 1) for f in frames]
    split = split_of(video)
    glares = []
    for i, sh in enumerate(shifts):
        # where the object is in this view (belt shift), so glare can be placed on it
        moved = [[x1 + sh, y1, x2 + sh, y2, c] for x1, y1, x2, y2, c in bxs if x2 + sh > 0 and x1 + sh < IMG_W]
        forced = split == "test" and i == T_IDX and (fid // POS_STRIDE) % 2 == 0
        glares.append(glare_at(moved) if (forced or random.random() < P_GLARE) else None)
    grid = boxes_to_grid(bxs)
    return {"split": split, "video": video, "t": fid, "k": k, "t_idx": T_IDX, "frames": frames,
            "shift_px": shifts, "glares": glares, "boxes": bxs,
            "grid_ones": [[int(r), int(c)] for r, c in zip(*np.nonzero(grid))]}


samples = []
for video in sorted({k.split("/")[0] for k in boxes}):
    motion, available = load_motion(video), {f for f, _ in frames_of(video)}
    fids = sorted(int(k.split("/")[1]) for k in boxes if k.startswith(video + "/"))
    for fid in fids[::POS_STRIDE]:
        s = make_sample(video, fid, motion, available, boxes[f"{video}/{fid}"])
        if s and s["grid_ones"]:
            samples.append(s)
for video in NORMAL_VIDEOS:
    motion, fr = load_motion(video), frames_of(video)
    available = {f for f, _ in fr}
    for fid, _ in fr[4 * MAX_K:-4 * MAX_K:NEG_STRIDE]:
        s = make_sample(video, fid, motion, available, [])
        if s:
            samples.append(s)
json.dump(samples, open(os.path.join(SAMP, "samples_T8.json"), "w"))

print(f"{'split':6s} {'samples':>8s} {'with object':>12s} {'glare on t':>11s} {'glared views/sample':>20s}")
for split in ["train", "val", "test"]:
    ss = [s for s in samples if s["split"] == split]
    print(f"{split:6s} {len(ss):8d} {sum(1 for s in ss if s['grid_ones']):12d} "
          f"{sum(1 for s in ss if s['glares'][T_IDX]):11d} {np.mean([sum(g is not None for g in s['glares']) for s in ss]):20.2f}")
uniq = {(s["video"], f) for s in samples for f in s["frames"]}
print(f"unique frames: {len(uniq)}   k range {min(s['k'] for s in samples)}..{max(s['k'] for s in samples)}")

# pictures: 2 examples, 8 views each, target view with the answer grid
random.seed(1)
for n, s in enumerate(random.sample([x for x in samples if x["grid_ones"] and sum(g is not None for g in x["glares"]) >= 2], 2)):
    cells = []
    for i, (f, sh, g) in enumerate(zip(s["frames"], s["shift_px"], s["glares"])):
        im = load_model_frame(s["video"], f, g)
        if i == T_IDX:
            over = im.copy()
            for r, c in s["grid_ones"]:
                cv2.rectangle(over, (c * MODEL_W // GRID_W, r * MODEL_H // GRID_H),
                              ((c + 1) * MODEL_W // GRID_W, (r + 1) * MODEL_H // GRID_H), (0, 0, 255), -1)
            im = cv2.addWeighted(over, 0.4, im, 0.6, 0)
        put_label(im, f"view {OFFSETS[i]:+d}k  frame {f}" + ("  = TARGET t" if i == T_IDX else ""), (8, 30), 0.8)
        put_label(im, f"shift {sh:+.0f}px" + ("  GLARE" if g else ""), (8, 62), 0.8)
        cells.append(cv2.resize(im, (480, 264)))
    cv2.imwrite(os.path.join(OUT, f"T8_sample_{n:02d}.jpg"), np.vstack([np.hstack(cells[:4]), np.hstack(cells[4:])]))
print("pictures:", OUT)

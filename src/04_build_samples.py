"""Step 4 - Build the training examples: 3 frames (t-k, t, t+k) + fake glare + the 0/1 answer grid.

One SAMPLE =
  video, target frame t, neighbour gap k
  shift_px  : how far the belt content moved from t to t-k and from t to t+k (original pixels, x)
  glare     : None, or {frame: 0|1|2, cx, cy, r, a} (which of the 3 frames gets a fake glare blob)
  grid      : 22x40 0/1 answer for frame t (computed from the boxes; stored as a list of cells that are 1)

Positive samples come from labeled frames (data/labels/boxes.json).
Negative samples come from ore-only videos (answer grid = all zeros).

Choosing k: we want the object to move ~3 grid cells (~96 px) between neighbours, so the 3 views
are clearly different (helps against glare) but the object is still in view. Fast belt -> small k.

Output:
  data/samples/samples.json
  outputs/04_samples/box_to_grid.jpg      how a box becomes a 0/1 grid
  outputs/04_samples/sample_XX.jpg        what one training example looks like
  outputs/04_samples/glare_examples.jpg   a few glare blobs

Run:  .venv\\Scripts\\python src\\04_build_samples.py
"""
import os
import json
import random
import cv2
import numpy as np

from common import (ROOT, frames_of, load_motion, cumulative_shift, boxes_to_grid, apply_glare,
                    load_model_frame, split_of, NORMAL_VIDEOS, GRID_W, GRID_H, MODEL_W, MODEL_H,
                    IMG_W, IMG_H, BELT_Y0, BELT_Y1, put_label, frame_path)

random.seed(0)
TARGET_SHIFT_PX = 96      # desired object movement between neighbouring views
MAX_K = 12
POS_STRIDE = 2            # use every 2nd labeled frame (neighbours are ~identical anyway)
NEG_STRIDE = 6            # ore-only frames: every 6th frame
P_NO_GLARE_TRAIN = 0.25   # training: 25% of samples without glare
P_GLARE_ON_OBJECT = 0.7   # place glare over the object most of the time (that is the hard case)

OUT = os.path.join(ROOT, "outputs", "04_samples")
SAMP = os.path.join(ROOT, "data", "samples")
os.makedirs(OUT, exist_ok=True)
os.makedirs(SAMP, exist_ok=True)

boxes = json.load(open(os.path.join(ROOT, "data", "labels", "boxes.json")))


def random_glare(boxes_t, frame_slot):
    if boxes_t and random.random() < P_GLARE_ON_OBJECT:
        x1, y1, x2, y2, _ = random.choice(boxes_t)
        cx, cy = random.uniform(x1, x2), random.uniform(y1, y2)
    else:
        cx, cy = random.uniform(100, IMG_W - 100), random.uniform(BELT_Y0 + 80, BELT_Y1 - 80)
    return {"frame": frame_slot, "cx": round(cx), "cy": round(cy),
            "r": random.randint(90, 220), "a": round(random.uniform(0.7, 1.0), 2)}


def choose_k(motion, fid):
    """Smallest k (1..MAX_K) such that the belt moved >= TARGET_SHIFT_PX between t and t+k."""
    for k in range(1, MAX_K + 1):
        if abs(cumulative_shift(motion, fid, fid + k)) >= TARGET_SHIFT_PX:
            return k
    return MAX_K  # belt nearly stopped: take the largest gap


def make_sample(video, fid, motion, available, bxs):
    k = choose_k(motion, fid)
    if fid - k not in available or fid + k not in available:
        return None
    split = split_of(video)
    if split == "test":
        # test: half the samples clean, half with glare on the TARGET frame (hardest, cleanest comparison)
        glare = random_glare(bxs, 1) if random.random() < 0.5 else None
    else:
        glare = None if random.random() < P_NO_GLARE_TRAIN else random_glare(bxs, random.randint(0, 2))
    grid = boxes_to_grid(bxs)
    return {
        "split": split, "video": video, "t": fid, "k": k,
        "frames": [fid - k, fid, fid + k],
        # where frame t's content is in the neighbour frames (negative = moved left)
        "shift_px": [round(cumulative_shift(motion, fid, fid - k), 1), 0.0,
                     round(cumulative_shift(motion, fid, fid + k), 1)],
        "glare": glare,
        "boxes": bxs,
        "grid_ones": [[int(r), int(c)] for r, c in zip(*np.nonzero(grid))],
    }


samples = []
# positives
pos_videos = sorted({k.split("/")[0] for k in boxes})
for video in pos_videos:
    motion = load_motion(video)
    available = {f for f, _ in frames_of(video)}
    fids = sorted(int(k.split("/")[1]) for k in boxes if k.startswith(video + "/"))
    for fid in fids[::POS_STRIDE]:
        s = make_sample(video, fid, motion, available, boxes[f"{video}/{fid}"])
        if s and s["grid_ones"]:
            samples.append(s)
# negatives
for video in NORMAL_VIDEOS:
    motion = load_motion(video)
    fr = frames_of(video)
    available = {f for f, _ in fr}
    for fid, _ in fr[MAX_K:-MAX_K:NEG_STRIDE]:
        s = make_sample(video, fid, motion, available, [])
        if s:
            samples.append(s)

json.dump(samples, open(os.path.join(SAMP, "samples.json"), "w"))

# ---- statistics
print(f"{'split':6s} {'samples':>8s} {'with object':>12s} {'ore only':>9s} {'with glare':>11s} {'object cells %':>15s}")
for split in ["train", "val", "test"]:
    ss = [s for s in samples if s["split"] == split]
    pos = sum(1 for s in ss if s["grid_ones"])
    gl = sum(1 for s in ss if s["glare"])
    cells = sum(len(s["grid_ones"]) for s in ss) / max(1, len(ss) * GRID_W * GRID_H) * 100
    print(f"{split:6s} {len(ss):8d} {pos:12d} {len(ss) - pos:9d} {gl:11d} {cells:14.1f}%")
uniq = {(s["video"], f) for s in samples for f in s["frames"]}
print(f"unique frames used: {len(uniq)}   k range: {min(s['k'] for s in samples)}..{max(s['k'] for s in samples)}")


# ---- pictures
def grid_overlay(img, grid, color=(0, 0, 255)):
    """Draw grid lines and fill cells that are 1."""
    h, w = img.shape[:2]
    cw, ch = w / GRID_W, h / GRID_H
    over = img.copy()
    for r, c in zip(*np.nonzero(grid)):
        cv2.rectangle(over, (int(c * cw), int(r * ch)), (int((c + 1) * cw), int((r + 1) * ch)), color, -1)
    img = cv2.addWeighted(over, 0.45, img, 0.55, 0)
    for c in range(GRID_W + 1):
        cv2.line(img, (int(c * cw), 0), (int(c * cw), h), (200, 200, 200), 1)
    for r in range(GRID_H + 1):
        cv2.line(img, (0, int(r * ch)), (w, int(r * ch)), (200, 200, 200), 1)
    return img


# 1) box -> grid explanation
s = next(x for x in samples if x["video"].endswith("canga2") and len(x["grid_ones"]) > 20)
img = cv2.resize(cv2.imread(frame_path(s["video"], s["t"])), (MODEL_W * 2, MODEL_H * 2))
a = img.copy()
for x1, y1, x2, y2, c in s["boxes"]:
    f = MODEL_W * 2 / IMG_W
    cv2.rectangle(a, (int(x1 * f), int(y1 * f * IMG_W / IMG_H * MODEL_H / MODEL_W)),
                  (int(x2 * f), int(y2 * f * IMG_W / IMG_H * MODEL_H / MODEL_W)), (0, 120, 255), 4)
put_label(a, "1) label = box (4 numbers)", (10, 40))
grid = boxes_to_grid(s["boxes"])
b = grid_overlay(img, grid)
put_label(b, f"2) grid {GRID_W}x{GRID_H}: red cells = 1", (10, 40))
small = np.kron(grid, np.ones((1, 1), np.uint8))
txt = np.full((MODEL_H * 2, MODEL_W * 2, 3), 255, np.uint8)
for r in range(GRID_H):
    line = "".join("1" if v else "." for v in small[r])
    cv2.putText(txt, line, (20, 40 + r * 30), cv2.FONT_HERSHEY_PLAIN, 1.6, (0, 0, 0), 2)
put_label(txt, "3) what the model must output", (10, MODEL_H * 2 - 20))
cv2.imwrite(os.path.join(OUT, "box_to_grid.jpg"), np.hstack([a, b, txt]))

# 2) a few complete training samples
picked = [x for x in samples if x["glare"] and x["grid_ones"]]
picked = random.sample(picked, 4) + random.sample([x for x in samples if not x["grid_ones"]], 1)
for i, s in enumerate(picked):
    row = []
    for slot, (fid, sh) in enumerate(zip(s["frames"], s["shift_px"])):
        gl = s["glare"] if (s["glare"] and s["glare"]["frame"] == slot) else None
        im = load_model_frame(s["video"], fid, gl)
        if slot == 1:
            im = grid_overlay(im, boxes_to_grid(s["boxes"]))
        name = ["t-k", "t (target)", "t+k"][slot]
        put_label(im, f"{name}: frame {fid}", (8, 30), 0.8)
        put_label(im, f"belt shift {sh:+.0f}px" + ("  GLARE" if gl else ""), (8, 60), 0.7)
        row.append(im)
    strip = np.hstack(row)
    put_label(strip, f"{s['split']} | {s['video'].replace('boardcamera_', '')} | k={s['k']}", (8, MODEL_H - 12), 0.8)
    cv2.imwrite(os.path.join(OUT, f"sample_{i:02d}.jpg"), strip)

# 3) glare examples
cells = []
for s in random.sample([x for x in samples if x["glare"]], 6):
    im = load_model_frame(s["video"], s["frames"][s["glare"]["frame"]], s["glare"])
    cells.append(im)
cv2.imwrite(os.path.join(OUT, "glare_examples.jpg"), np.vstack([np.hstack(cells[:3]), np.hstack(cells[3:])]))
print("pictures in", OUT)

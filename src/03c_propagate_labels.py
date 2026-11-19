"""Step 3c - Spread keyframe boxes to all frames of a pass using belt motion, with two-keyframe correction.

PLAIN PHYSICS (old method, saved for comparison):
  every frame copies the boxes of its nearest keyframe, shifted by the belt displacement.
  Problem: objects that roll/slide drift away from their box, and the box jumps at the hand-over
  to the next keyframe.

TWO-KEYFRAME CORRECTION (extension E8, method 1, used for training):
  a frame between keyframes A and B gets, for each object seen in both,
      position = (1-a) * (A's box moved by physics) + a * (B's box moved by physics),   a = (f-A)/(B-A)
  At A it equals A's label, at B it equals B's label, and in between the error is spread linearly:
  no jump, and rolling/sliding is followed (approximately, assuming it is steady).
  Box sides cut off by the image border in a keyframe are taken from the other keyframe
  (or extended to the border if cut in both).

ROLLING FLAGS: for every keyframe pair, how far is B's real box from A's physics prediction?
  More than FLAG_PX (larger than the ~30-50 px label noise) -> the object moved relative to the belt.

Input : data/labels/keyframes.csv (+ keyframes_extra.csv if present), data/labels/<BOX_FILE>, data/meta/motion/*.csv
Output: data/labels/boxes.json          corrected labels (used by step 04)
        data/labels/boxes_plain.json    plain physics labels (for comparison only)
        data/labels/rolling_flags.csv   keyframe pairs where the object moved relative to the belt
        outputs/03_labeling/correction_<video>.jpg  plain vs corrected, side by side

Run:  .venv\\Scripts\\python src\\03c_propagate_labels.py  [box_file.json]
"""
import os
import sys
import csv
import json
import cv2
import numpy as np

from common import ROOT, IMG_W, load_motion, cumulative_shift, frame_path, put_label

BOX_FILE = sys.argv[1] if len(sys.argv) > 1 else "keyframe_boxes_llm.json"
MAX_TRAVEL_PX = 500   # outside the keyframe range: do not trust a box further than this from its keyframe
MIN_VISIBLE_PX = 30   # drop a box when less than this many pixels of it are inside the image
EDGE = 5              # a keyframe box this close to the border is "cut off" (object continues outside)
FLAG_PX = 60          # residual above this = rolling/sliding (LLM box noise is ~30-50 px)
LAB = os.path.join(ROOT, "data", "labels")
OUT = os.path.join(ROOT, "outputs", "03_labeling")
COLORS = {"wood": (0, 200, 255), "plastic": (255, 255, 255), "canga": (0, 120, 255),
          "jaspilito": (200, 120, 255), "tool": (0, 255, 0)}

kf_boxes = json.load(open(os.path.join(LAB, BOX_FILE)))["keyframes"]
kf_rows = list(csv.DictReader(open(os.path.join(LAB, "keyframes.csv"))))
extra_path = os.path.join(LAB, "keyframes_extra.csv")
if os.path.exists(extra_path):
    kf_rows += [r for r in csv.DictReader(open(extra_path)) if f"{r['video']}/{r['keyframe_id']}" in kf_boxes]


def cut_l(b):
    return b[0] <= EDGE


def cut_r(b):
    return b[2] >= IMG_W - EDGE


def finish(x1, y1, x2, y2, c):
    """Clip to the image; None if (almost) nothing is visible."""
    x1, x2 = max(0.0, x1), min(float(IMG_W), x2)
    if x2 - x1 < MIN_VISIBLE_PX:
        return None
    return [round(x1), round(y1), round(x2), round(y2), c]


def from_one(box, dx):
    """Plain physics from one keyframe; cut-off sides extend to the border."""
    x1 = 0 if cut_l(box) else box[0] + dx
    x2 = IMG_W if cut_r(box) else box[2] + dx
    return finish(x1, box[1], x2, box[3], box[4])


def from_two(bA, sA, bB, sB, a):
    """Two-keyframe blend. sA/sB = belt shift from keyframe A/B to this frame, a = 0 at A .. 1 at B."""
    def side(i, cA, cB, border):
        va, vb = bA[i] + sA, bB[i] + sB
        if not cA and not cB:
            return (1 - a) * va + a * vb
        if not cA:
            return va
        if not cB:
            return vb
        return border
    x1 = side(0, cut_l(bA), cut_l(bB), 0)
    x2 = side(2, cut_r(bA), cut_r(bB), IMG_W)
    y1 = (1 - a) * bA[1] + a * bB[1]
    y2 = (1 - a) * bA[3] + a * bB[3]
    return finish(x1, y1, x2, y2, bA[4])


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    union = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
    return inter / union if union > 0 else 0


def match(boxesA, boxesB, shift_AB):
    """Pair up the same object in keyframes A and B: same class, A's physics prediction overlaps B (greedy by IoU)."""
    cands = []
    for i, a in enumerate(boxesA):
        pa = [max(0, a[0] + shift_AB), a[1], min(IMG_W, a[2] + shift_AB), a[3]]
        for j, b in enumerate(boxesB):
            if a[4] == b[4]:
                ov = iou(pa, b)
                dist = abs((pa[0] + pa[2]) / 2 - (b[0] + b[2]) / 2) + abs((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2)
                if ov > 0.05 or dist < 200:
                    cands.append((-ov, dist, i, j))
    pairs, used_a, used_b = [], set(), set()
    for _, _, i, j in sorted(cands):
        if i not in used_a and j not in used_b:
            pairs.append((i, j))
            used_a.add(i)
            used_b.add(j)
    return pairs


def residual(a, b, shift_AB):
    """How far B's real box is from A's physics prediction (only sides visible in both)."""
    rx = [b[i] - (a[i] + shift_AB) for i, cut in [(0, cut_l), (2, cut_r)] if not cut(a) and not cut(b)]
    ry = ((b[1] - a[1]) + (b[3] - a[3])) / 2
    return (float(np.mean(rx)) if rx else 0.0), ry


# ---- group keyframes into passes
passes = {}
for r in kf_rows:
    key = (r["video"], int(r["pass_start"]), int(r["pass_end"]))
    passes.setdefault(key, set()).add(int(r["keyframe_id"]))

plain, corrected, flags = {}, {}, []
for (video, start, end), kfs in sorted(passes.items()):
    kfs = sorted(kfs)
    motion = load_motion(video)
    pair_info = {}
    for A, B in zip(kfs, kfs[1:]):
        bA, bB = kf_boxes[f"{video}/{A}"], kf_boxes[f"{video}/{B}"]
        sAB = cumulative_shift(motion, A, B)
        pairs = match(bA, bB, sAB)
        pair_info[(A, B)] = pairs
        for i, j in pairs:
            rx, ry = residual(bA[i], bB[j], sAB)
            if max(abs(rx), abs(ry)) > FLAG_PX:
                flags.append({"video": video, "A": A, "B": B, "class": bA[i][4],
                              "residual_x": round(rx), "residual_y": round(ry), "belt_travel": round(abs(sAB))})

    for fid in range(start, end + 1):
        # plain physics: nearest keyframe in belt travel
        d, k = min((abs(cumulative_shift(motion, k, fid)), k) for k in kfs)
        if d <= MAX_TRAVEL_PX:
            plain[f"{video}/{fid}"] = [b for b in (from_one(x, cumulative_shift(motion, k, fid)) for x in kf_boxes[f"{video}/{k}"]) if b]

        # corrected
        if fid in kfs:
            boxes = [b for b in (finish(*x) for x in kf_boxes[f"{video}/{fid}"]) if b]
        elif fid < kfs[0] or fid > kfs[-1]:
            k = kfs[0] if fid < kfs[0] else kfs[-1]
            s = cumulative_shift(motion, k, fid)
            if abs(s) > MAX_TRAVEL_PX:
                continue
            boxes = [b for b in (from_one(x, s) for x in kf_boxes[f"{video}/{k}"]) if b]
        else:
            A = max(k for k in kfs if k < fid)
            B = min(k for k in kfs if k > fid)
            bA, bB = kf_boxes[f"{video}/{A}"], kf_boxes[f"{video}/{B}"]
            sA, sB = cumulative_shift(motion, A, fid), cumulative_shift(motion, B, fid)
            a = (fid - A) / (B - A)
            pairs = pair_info[(A, B)]
            boxes = [from_two(bA[i], sA, bB[j], sB, a) for i, j in pairs]
            # objects seen only in A (leaving) or only in B (entering): plain physics from that keyframe
            boxes += [from_one(x, sA) for i, x in enumerate(bA) if i not in {p[0] for p in pairs}]
            boxes += [from_one(x, sB) for j, x in enumerate(bB) if j not in {p[1] for p in pairs}]
            boxes = [b for b in boxes if b]
        corrected[f"{video}/{fid}"] = boxes

json.dump(corrected, open(os.path.join(LAB, "boxes.json"), "w"))
json.dump(plain, open(os.path.join(LAB, "boxes_plain.json"), "w"))
with open(os.path.join(LAB, "rolling_flags.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, ["video", "A", "B", "class", "residual_x", "residual_y", "belt_travel"])
    w.writeheader()
    w.writerows(flags)

print(f"{len(kf_boxes)} keyframes -> {len(corrected)} labeled frames, {sum(len(b) for b in corrected.values())} boxes")
print(f"rolling/sliding flagged on {len(flags)} keyframe pairs (residual > {FLAG_PX} px):")
for fl in flags:
    print(f"  {fl['video'].replace('boardcamera_', ''):22s} {fl['A']}->{fl['B']}  {fl['class']:9s} "
          f"off by x {fl['residual_x']:+5d}  y {fl['residual_y']:+5d} px  (belt moved {fl['belt_travel']} px)")


# ---- picture: plain vs corrected on the first flagged interval of a few videos
def draw(img, boxes, color):
    for x1, y1, x2, y2, c in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)
    return img


shown = set()
for fl in flags:
    if fl["video"] in shown or len(shown) >= 4:
        continue
    shown.add(fl["video"])
    v, A, B = fl["video"], fl["A"], fl["B"]
    fids = [int(x) for x in np.linspace(A, B, 6)]
    cells = []
    for fid in fids:
        img = cv2.imread(frame_path(v, fid))
        draw(img, plain.get(f"{v}/{fid}", []), (0, 0, 255))
        draw(img, corrected.get(f"{v}/{fid}", []), (0, 255, 0))
        put_label(img, f"frame {fid}" + ("  KEYFRAME" if fid in (A, B) else ""), (10, 45), 1.3)
        cells.append(cv2.resize(img, (640, 360)))
    sheet = np.vstack([np.hstack(cells[:3]), np.hstack(cells[3:])])
    put_label(sheet, "red = plain physics   green = two-keyframe correction", (10, 700), 1.0)
    cv2.imwrite(os.path.join(OUT, f"correction_{v.replace('boardcamera_', '')}.jpg"), sheet)
print("comparison pictures:", ", ".join(f"correction_{v.replace('boardcamera_', '')}.jpg" for v in shown))

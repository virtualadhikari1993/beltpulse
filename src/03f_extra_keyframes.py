"""Step 3f - Add keyframes where objects roll/slide (flagged by 03c), so they get a real label instead of physics.

For every flagged keyframe pair (A, B) in data/labels/rolling_flags.csv:
  residual <= 100 px -> 1 new keyframe in the middle of A..B
  residual  > 100 px -> 3 new keyframes at 1/4, 1/2, 3/4
These are labeled directly (LLM in the dry run, you later) and serve as reviewed ground truth in rolling regions.

Output:
  data/labels/keyframes_extra.csv                       video, keyframe_id, pass_start, pass_end, reason
  outputs/03_labeling/keyframes_extra/<video>__<id>.jpg  frame with coordinate grid (for labeling)
  outputs/03_labeling/sheets_extra/sheet_XX.jpg          2x2 sheets + index.txt

Then: label them (add to data/labels/keyframe_boxes_llm.json, key "<video>/<id>",
and list the keys under "extra_keyframes"), and re-run 03c -> 03d -> 03e -> 04.

Run:  .venv\\Scripts\\python src\\03f_extra_keyframes.py
"""
import os
import csv
import cv2
import numpy as np

from common import ROOT, frame_path, put_label

LAB = os.path.join(ROOT, "data", "labels")
OUT_K = os.path.join(ROOT, "outputs", "03_labeling", "keyframes_extra")
OUT_S = os.path.join(ROOT, "outputs", "03_labeling", "sheets_extra")
os.makedirs(OUT_K, exist_ok=True)
os.makedirs(OUT_S, exist_ok=True)


def draw_grid(img):
    h, w = img.shape[:2]
    for x in range(0, w, 100):
        cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
        put_label(img, str(x), (x + 2, h - 8), 0.5)
    for y in range(0, h, 100):
        cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)
        put_label(img, str(y), (2, y - 4), 0.5)
    return img


kf_rows = list(csv.DictReader(open(os.path.join(LAB, "keyframes.csv"))))
flags = list(csv.DictReader(open(os.path.join(LAB, "rolling_flags.csv"))))
existing = {(r["video"], int(r["keyframe_id"])) for r in kf_rows}

new = {}
for fl in flags:
    v, A, B = fl["video"], int(fl["A"]), int(fl["B"])
    res = max(abs(int(fl["residual_x"])), abs(int(fl["residual_y"])))
    fracs = [0.25, 0.5, 0.75] if res > 100 else [0.5]
    p = next(r for r in kf_rows if r["video"] == v and int(r["pass_start"]) <= A <= int(r["pass_end"]))
    for fr in fracs:
        fid = int(round(A + fr * (B - A)))
        if (v, fid) not in existing:
            new[(v, fid)] = (p["pass_start"], p["pass_end"], f"rolling {A}->{B} off by {res}px")

rows = sorted(new.items())
with open(os.path.join(LAB, "keyframes_extra.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["video", "keyframe_id", "pass_start", "pass_end", "reason"])
    for (v, fid), (ps, pe, why) in rows:
        w.writerow([v, fid, ps, pe, why])

names = []
for (v, fid), _ in rows:
    img = draw_grid(cv2.imread(frame_path(v, fid)))
    put_label(img, f"{v.replace('boardcamera_', '')}  frame {fid}  (extra)", (110, 40), 1.0)
    n = f"{v}__{fid}.jpg"
    cv2.imwrite(os.path.join(OUT_K, n), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    names.append(n)

with open(os.path.join(OUT_S, "index.txt"), "w") as idx:
    for s in range(0, len(names), 4):
        cells = []
        for k, n in enumerate(names[s:s + 4]):
            im = cv2.resize(cv2.imread(os.path.join(OUT_K, n)), (960, 540))
            put_label(im, f"[{s + k}]", (850, 40), 1.0)
            cells.append(im)
            idx.write(f"{s + k}\t{n}\n")
        while len(cells) < 4:
            cells.append(np.zeros_like(cells[0]))
        cv2.imwrite(os.path.join(OUT_S, f"sheet_{s // 4:02d}.jpg"),
                    np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(f"{len(rows)} extra keyframes from {len(flags)} flagged intervals -> {OUT_K}")
for (v, fid), (_, _, why) in rows:
    print(f"  {v.replace('boardcamera_', ''):22s} {fid}   {why}")

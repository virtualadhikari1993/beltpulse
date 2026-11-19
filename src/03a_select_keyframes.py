"""Step 3a - Choose the keyframes that will be labeled by hand (or, for the first dry run, by an LLM).

Idea: every object passing under the camera ("pass" = run of consecutive anomalous frames)
only needs a few labeled frames. Every time the belt has moved ~KEY_SPACING_PX pixels we
take a keyframe. Step 3c later copies each keyframe's boxes to the frames around it
by shifting them with the measured belt motion.

Output:
  data/labels/keyframes.csv                     video, keyframe_id, pass_start, pass_end
  outputs/03_labeling/keyframes/<video>__<id>.jpg   keyframe with a coordinate grid
     (grid lines every 100 px in ORIGINAL 1280x720 pixel coordinates, to read box corners)

Run:  .venv\\Scripts\\python src\\03a_select_keyframes.py
"""
import os
import csv
import cv2

from common import ROOT, anomaly_frames, runs, load_motion, cumulative_shift, frame_path, put_label

KEY_SPACING_PX = 700
OUT = os.path.join(ROOT, "outputs", "03_labeling", "keyframes")
LAB = os.path.join(ROOT, "data", "labels")
os.makedirs(OUT, exist_ok=True)
os.makedirs(LAB, exist_ok=True)


def draw_grid(img):
    h, w = img.shape[:2]
    for x in range(0, w, 100):
        cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
        put_label(img, str(x), (x + 2, h - 8), 0.5)
    for y in range(0, h, 100):
        cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)
        put_label(img, str(y), (2, y - 4), 0.5)
    return img


rows = []
for video, ids in sorted(anomaly_frames().items()):
    motion = load_motion(video)
    for run in runs(ids):
        start, end = run[0], run[-1]
        # walk through the pass and drop a keyframe every KEY_SPACING_PX of belt travel,
        # starting half a spacing in so the object is well inside the view
        last = None
        for fid in run:
            travelled = abs(cumulative_shift(motion, start if last is None else last, fid))
            if (last is None and travelled >= KEY_SPACING_PX / 2) or (last is not None and travelled >= KEY_SPACING_PX):
                rows.append((video, fid, start, end))
                last = fid
        if last is None:  # very slow / short pass: take its middle frame
            rows.append((video, run[len(run) // 2], start, end))

with open(os.path.join(LAB, "keyframes.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["video", "keyframe_id", "pass_start", "pass_end"])
    w.writerows(rows)

for video, fid, _, _ in rows:
    img = cv2.imread(frame_path(video, fid))
    img = draw_grid(img)
    put_label(img, f"{video.replace('boardcamera_', '')}  frame {fid}", (110, 40), 1.0)
    cv2.imwrite(os.path.join(OUT, f"{video}__{fid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

print(f"{len(rows)} keyframes written to {OUT}")

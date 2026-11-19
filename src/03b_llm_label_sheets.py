"""Step 3b (dry-run only) - Tile keyframes 2x2 into "sheets" so an LLM (Claude) can look at them
and write approximate boxes into data/labels/keyframe_boxes_llm.json.

In the real project you replace this step by labeling the keyframes in Roboflow/CVAT
(see docs/03_labeling.md) and exporting YOLO txt files.

Output: outputs/03_labeling/sheets/sheet_XX.jpg  (panel index in the top-left corner)
"""
import os
import csv
import cv2
import numpy as np

from common import ROOT, put_label

KF = os.path.join(ROOT, "outputs", "03_labeling", "keyframes")
OUT = os.path.join(ROOT, "outputs", "03_labeling", "sheets")
os.makedirs(OUT, exist_ok=True)

rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "labels", "keyframes.csv"))))
names = [f"{r['video']}__{r['keyframe_id']}.jpg" for r in rows]
index = []
for s in range(0, len(names), 4):
    cells = []
    for k, n in enumerate(names[s:s + 4]):
        im = cv2.resize(cv2.imread(os.path.join(KF, n)), (960, 540))
        put_label(im, f"[{s + k}]", (850, 40), 1.0)
        cells.append(im)
        index.append((s + k, n))
    while len(cells) < 4:
        cells.append(np.zeros_like(cells[0]))
    sheet = np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])
    cv2.imwrite(os.path.join(OUT, f"sheet_{s // 4:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

with open(os.path.join(OUT, "index.txt"), "w") as f:
    f.writelines(f"{i}\t{n}\n" for i, n in index)
print(len(index), "panels in", OUT)

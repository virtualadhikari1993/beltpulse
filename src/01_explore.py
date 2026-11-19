"""Step 1a - Look at the data.

Produces pictures in outputs/01_explore/:
  overview.jpg      one frame from every video (what does each recording contain?)
  sequence_<v>.jpg  6 consecutive frames from one video (how much does the belt move?)
  anomaly_samples.jpg  samples from the 'anomalous' folder (what counts as an impurity?)
And prints, for every video, the belt shift per frame (dy) measured with phase correlation.

Run:  .venv\\Scripts\\python src\\01_explore.py
"""
import os
import random
import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "raw", "ironore")
OUT = os.path.join(ROOT, "outputs", "01_explore")
os.makedirs(OUT, exist_ok=True)


def frames_of(video):
    """Sorted list of frame paths for one video (frame id is the number before '-')."""
    d = os.path.join(DATA, "image-files", video)
    files = sorted(os.listdir(d), key=lambda f: int(f.split("-")[0]))
    return [os.path.join(d, f) for f in files]


def label(img, text):
    cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
    cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return img


def grid(images, cols, cell_w=480):
    cells = []
    for im in images:
        h = int(im.shape[0] * cell_w / im.shape[1])
        cells.append(cv2.resize(im, (cell_w, h)))
    cell_h = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, cell_h - c.shape[0], 0, 0, cv2.BORDER_CONSTANT) for c in cells]
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    return np.vstack(rows)


def belt_shift(path_a, path_b):
    """How many pixels did the image content move from frame a to frame b?
    Phase correlation finds the single (dx, dy) translation that best aligns the two images.
    On a conveyor almost everything moves together, so this is the belt displacement."""
    a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, win)
    return dx, dy, response


videos = sorted(os.listdir(os.path.join(DATA, "image-files")))

# 1) one middle frame per video
overview = []
for v in videos:
    fs = frames_of(v)
    overview.append(label(cv2.imread(fs[len(fs) // 2]), v.replace("boardcamera_", "")))
cv2.imwrite(os.path.join(OUT, "overview.jpg"), grid(overview, cols=4))

# 2) consecutive frames + belt motion statistics
print(f"{'video':32s} {'dx/frame':>9s} {'dy/frame':>9s} {'std dy':>7s} {'conf':>5s}")
random.seed(0)
for v in videos:
    fs = frames_of(v)
    shifts = []
    for _ in range(40):
        i = random.randrange(len(fs) - 1)
        shifts.append(belt_shift(fs[i], fs[i + 1]))
    s = np.array(shifts)
    print(f"{v.replace('boardcamera_', ''):32s} {np.median(s[:, 0]):9.2f} {np.median(s[:, 1]):9.2f} "
          f"{s[:, 1].std():7.2f} {np.median(s[:, 2]):5.2f}")

for v in ["boardcamera_iron_ore_7_wood", "boardcamera_iron_ore_things"]:
    fs = frames_of(v)
    start = len(fs) // 2
    seq = [label(cv2.imread(fs[start + k]), f"t+{k}") for k in range(6)]
    cv2.imwrite(os.path.join(OUT, f"sequence_{v.replace('boardcamera_', '')}.jpg"), grid(seq, cols=3))

# 3) what does the 'anomalous' folder contain?
an_dir = os.path.join(DATA, "split-ds-normal-filtered", "anomalous", "anomaly")
an = random.sample(sorted(os.listdir(an_dir)), 8)
cv2.imwrite(os.path.join(OUT, "anomaly_samples.jpg"),
            grid([label(cv2.imread(os.path.join(an_dir, f)), f.split("-")[0]) for f in an], cols=4, cell_w=320))
print("pictures written to", os.path.abspath(OUT))

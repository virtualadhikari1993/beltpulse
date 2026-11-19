"""Shared paths and helpers used by every step."""
import os
import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data", "raw", "ironore")
IMG_W, IMG_H = 1280, 720

# Rows of the image that show the moving belt (the steel frame above/below does not move).
BELT_Y0, BELT_Y1 = 60, 640


def videos():
    return sorted(os.listdir(os.path.join(DATA, "image-files")))


def frames_of(video):
    """[(frame_id, path), ...] sorted by frame id."""
    d = os.path.join(DATA, "image-files", video)
    out = [(int(f.split("-")[0]), os.path.join(d, f)) for f in os.listdir(d)]
    return sorted(out)


_FRAME_INDEX = {}


def frame_path(video, fid):
    """Path of frame `fid` of `video`. Looked up from the folder, because file names are not uniform
    (e.g. iron_ore_5 frames are called '<id>-boardcamera_iron_ore_5_fastandfurious.jpeg')."""
    if video not in _FRAME_INDEX:
        _FRAME_INDEX[video] = dict(frames_of(video))
    return _FRAME_INDEX[video][fid]


def load_motion(video):
    """dict frame_id -> dx (pixels the belt moves from frame_id to frame_id+1).
    A running median over 5 frames removes single-frame measurement glitches (the spikes
    seen in outputs/02_belt_motion/*.png) while keeping real speed changes (belt stops)."""
    import csv
    p = os.path.join(ROOT, "data", "meta", "motion", f"{video}.csv")
    with open(p) as f:
        rows = [(int(r["frame_id"]), float(r["dx"])) for r in csv.DictReader(f)]
    ids = [r[0] for r in rows]
    dx = np.array([r[1] for r in rows])
    pad = np.pad(dx, 2, mode="edge")
    smooth = np.array([np.median(pad[i:i + 5]) for i in range(len(dx))])
    return dict(zip(ids, smooth.tolist()))


def anomaly_frames():
    """dict video -> sorted list of frame ids that the dataset authors marked as containing
    a foreign object (from split-ds-normal-filtered/anomalous/anomaly/<id>-<video>_detected_k.jpg)."""
    d = os.path.join(DATA, "split-ds-normal-filtered", "anomalous", "anomaly")
    out = {}
    for f in os.listdir(d):
        fid, rest = f.split("-", 1)
        video = rest.rsplit("_detected", 1)[0]
        out.setdefault(video, set()).add(int(fid))
    return {v: sorted(s) for v, s in out.items()}


def runs(ids, max_gap=3):
    """Group sorted frame ids into consecutive runs: [[a, a+1, ...], [b, ...]]."""
    groups = [[ids[0]]]
    for i in ids[1:]:
        if i - groups[-1][-1] <= max_gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def cumulative_shift(motion, f_from, f_to):
    """Total x displacement of belt content from frame f_from to frame f_to (can be negative)."""
    if f_to == f_from:
        return 0.0
    med = float(np.median(list(motion.values())))
    if f_to > f_from:
        return sum(motion.get(i, med) for i in range(f_from, f_to))
    return -sum(motion.get(i, med) for i in range(f_to, f_from))


# ---------------------------------------------------------------- model input / grid (step 04+)
# Frames are shrunk to 640x352 before going into the CNN. ResNet's layer3 reduces size by 16,
# so the feature grid is 40 x 22 cells. One cell = 16x16 px of the model input
# = 32 x 32.7 px of the original 1280x720 frame.
MODEL_W, MODEL_H = 640, 352
STRIDE = 16
GRID_W, GRID_H = MODEL_W // STRIDE, MODEL_H // STRIDE   # 40, 22
CELL_W_ORIG = IMG_W / GRID_W                             # 32 px of the original frame per cell (x)

# Video split: whole videos go to one split, so near-identical frames never leak from train into test.
# jaspilito_4 (real glare + motion blur, 2 rolling intervals) is in test instead of jaspilito_3 (2026-11-19),
# so the real difficult frames are judged on a video the models never trained on.
SPLITS = {
    "test": ["boardcamera_iron_ore_7_wood_3", "boardcamera_iron_ore_canga3",
             "boardcamera_iron_ore_jaspilito_4", "boardcamera_iron_ore_things_3", "boardcamera_iron_ore_4"],
    "val": ["boardcamera_iron_ore_canga", "boardcamera_iron_ore_jaspilito_2", "boardcamera_iron_ore_3"],
}


def split_of(video):
    for s, vs in SPLITS.items():
        if video in vs:
            return s
    return "train"


NORMAL_VIDEOS = ["boardcamera_iron_ore", "boardcamera_iron_ore_2", "boardcamera_iron_ore_3",
                 "boardcamera_iron_ore_4", "boardcamera_iron_ore_5", "boardcamera_iron_ore_6"]


def boxes_to_grid(boxes, min_cover=0.25):
    """Boxes in original pixels -> GRID_H x GRID_W array of 0/1.
    A cell is 1 when at least `min_cover` of its area lies inside some box."""
    sub = 4  # draw at 4x grid resolution, then average -> fraction of each cell covered
    m = np.zeros((GRID_H * sub, GRID_W * sub), np.float32)
    sx, sy = GRID_W * sub / IMG_W, GRID_H * sub / IMG_H
    for x1, y1, x2, y2, *_ in boxes:
        m[int(round(y1 * sy)):int(round(y2 * sy)), int(round(x1 * sx)):int(round(x2 * sx))] = 1
    cover = m.reshape(GRID_H, sub, GRID_W, sub).mean(axis=(1, 3))
    return (cover >= min_cover).astype(np.uint8)


def apply_glare(img, glare):
    """Paint a bright, soft white blob (fake overhead-light reflection) onto a BGR image.
    glare = {"cx", "cy", "r", "a"} in original pixels; a = strength 0..1.
    Returns the new image and the per-pixel glare strength map (0..1)."""
    h, w = img.shape[:2]
    sx, sy = w / IMG_W, h / IMG_H
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = ((xx - glare["cx"] * sx) / (glare["r"] * sx)) ** 2 + ((yy - glare["cy"] * sy) / (glare["r"] * sy * 0.7)) ** 2
    g = glare["a"] * np.exp(-d2)           # Gaussian, slightly flattened (wider than tall)
    g = np.clip(g * 1.6, 0, 1)             # flat saturated core, like a blown-out highlight
    out = img.astype(np.float32) * (1 - g[..., None]) + 255.0 * g[..., None]
    return out.astype(np.uint8), g


def glare_grid(glare, thresh=0.5):
    """Which grid cells are hit by the glare (strength >= thresh)? Used for the 'glare-region' metric."""
    _, g = apply_glare(np.zeros((GRID_H, GRID_W, 1), np.uint8), glare)
    return (g >= thresh).astype(np.uint8)


# "Glare-skip" rule (E9 baseline): a square counts as washed out when at least half of its pixels are
# near-white in every colour channel. Measured: synthetic glare squares 100%, real frames ~3-4% of squares
# (real lamp glare), white plastic ~11-15% of its squares.
WHITE_THR, WHITE_FRAC = 235, 0.5


def white_grid(img_model):
    """Fraction of near-white pixels per grid square (GRID_H x GRID_W, float) of a model-size BGR frame."""
    s = MODEL_W // GRID_W
    m = (img_model.min(-1) >= WHITE_THR).astype(np.float32)
    return m.reshape(GRID_H, s, GRID_W, s).mean((1, 3))


def load_model_frame(video, fid, glare=None):
    """Read a frame, optionally add glare, resize to the model input size (BGR uint8)."""
    img = cv2.imread(frame_path(video, fid))
    if glare is not None:
        img, _ = apply_glare(img, glare)
    return cv2.resize(img, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- variants (step 04+)
# base : 3 views (t-k, t, t+k), 3x3 attention window, glare on at most one view       (steps 04, 05-08)
# T8w7 : 8 views spread along the object's path, 7x7 window (+-3 squares around the
#        physics prediction, finds rolling objects), glare independently on any view   (step 04b; E10 + E11)
VARIANTS = {
    "base": dict(samples="samples.json", T=3, win=3),
    "T8w7": dict(samples="samples_T8.json", T=8, win=7),
}

# One shared cache for ResNet features of all runs/variants (file name = video, frame, glare recipe)
FEAT_DIR = os.path.join(ROOT, "data", "features", "all")


def run_tag(limit, variant="base"):
    """Name of a run: 'smoke100' / 'full', with '_<variant>' appended for non-base variants."""
    t = f"smoke{limit}" if limit else "full"
    return t if variant == "base" else f"{t}_{variant}"


def t_index(s):
    """Position of the target frame t inside s['frames']."""
    return s.get("t_idx", 1)


def frame_glare(s, slot):
    """Glare recipe painted on view `slot` of sample s (None = clean).
    base samples: one optional glare {'frame': slot, ...}; T8 samples: a list 'glares' with one entry per view."""
    if "glares" in s:
        return s["glares"][slot]
    g = s.get("glare")
    return g if (g and g["frame"] == slot) else None


def target_glare(s):
    return frame_glare(s, t_index(s))


# ---------------------------------------------------------------- cached features (step 05+)


def feat_key(video, fid, glare=None):
    """File name (without .npy) of the cached ResNet features of one photo (with or without glare)."""
    if glare is None:
        return f"{video}__{fid}"
    return f"{video}__{fid}__g{glare['cx']}_{glare['cy']}_{glare['r']}_{int(glare['a'] * 100)}"


def sample_feat_keys(s):
    """The feature files of a sample's views (a glared view gets its own file)."""
    return [feat_key(s["video"], f, frame_glare(s, i)) for i, f in enumerate(s["frames"])]


def put_label(img, text, org=(10, 40), scale=1.2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 6)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    return img

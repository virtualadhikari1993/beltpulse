"""Step 10 - One decision per OBJECT: count objects and say what type they are (extension E12).

THE DOUBLE-COUNTING PROBLEM
  An object is visible in 30-600 frames. Counting detections per frame would count one stick hundreds of times.
  Fix: link detections across frames with BELT PHYSICS into tracks, then count TRACKS.
    detect  : per frame, the detector's object squares -> blobs -> boxes
    link    : predict where each track's last box is now (shift by belt motion); a detection overlapping the
              prediction is the same object (joins the track); an unmatched detection starts a new track
    count   : a track counts once, and only if it has >= MIN_LEN detections (drops 1-2 frame flickers;
              a fixed lamp reflection does not move with the belt, so it cannot form a long track)
    classify: a transformer reads the track's sequence (one token per frame) and decides ONCE:
              ore (= false alarm, not counted) / wood / plastic / canga / jaspilito / tool

WHAT IS TRAINED HERE
  Each frame of a track -> ResNet features pooled inside the object's box (mean + max = 512 numbers) = one token.
  - SeqTransformer : reads the whole token sequence (variable length, some frames glared) -> one class   [ours]
  - MeanPool       : averages the tokens, then a small MLP -> one class                                   [averaging]
  - FrameMLP       : classifies every frame alone; per-object decision by voting:
                       OR (any frame says impurity), k-of-N (>= 30% of frames), mean                      [voting]
  Training tracks: labeled objects (data/labels/boxes.json, linked with the same physics tracker)
                   + "ore tracks": random ore regions followed with belt physics in ore-only videos.

OUTPUTS (outputs/10_per_object/<tag>/)
  results.md                per-object accuracy (clean / glared) and counts per test video
  counting_<video>.jpg      detected tracks drawn on a frame, with predicted type
  track_attention.jpg       one test object: its frames + how much the transformer attended to each

Usage:
  .venv\\Scripts\\python src\\10_per_object.py --smoke      # fast: every 8th frame, 5 epochs
  .venv\\Scripts\\python src\\10_per_object.py              # full
"""
import os
import json
import random
import argparse
from collections import Counter, defaultdict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (ROOT, IMG_W, IMG_H, GRID_W, GRID_H, CELL_W_ORIG, MODEL_W, MODEL_H, BELT_Y0, BELT_Y1,
                    frames_of, frame_path, load_motion, cumulative_shift, apply_glare, split_of, NORMAL_VIDEOS,
                    SPLITS, put_label, VARIANTS)
from backbone import Backbone
from model import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true", help="fast smoke test")
ap.add_argument("--detector", default="smoke100", help="run tag of the trained per-frame detector (step 06)")
ap.add_argument("--detector-mode", default="single")
args = ap.parse_args()
EVERY = 8 if args.smoke else 3            # use every EVERY-th frame of a track
DEMO_EVERY = 6 if args.smoke else 2       # counting demo: process every DEMO_EVERY-th frame
EPOCHS, STEPS = (5, 60) if args.smoke else (30, 200)
MAX_TRACK = 24 if args.smoke else 64      # frames per track kept for training/testing
MIN_LEN = 3                               # a track must have >= 3 detections to be counted
CLASSES = ["ore", "wood", "plastic", "canga", "jaspilito", "tool"]
tag = "smoke" if args.smoke else "full"
OUT = os.path.join(ROOT, "outputs", "10_per_object", tag)
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(ROOT, "data", "objects"), exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(0); np.random.seed(0); torch.manual_seed(0)
CELL_H = IMG_H / GRID_H


# ================================================================ helpers
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0


def link(dets, motion, max_gap, same_class=False):
    """Physics tracker. dets: {frame: [(box, payload), ...]} with box = [x1,y1,x2,y2(,class)].
    Returns tracks: [{'frames': [...], 'boxes': [...], 'payloads': [...]}]."""
    active, done = [], []
    for f in sorted(dets):
        still = []
        for t in active:
            (still if f - t["frames"][-1] <= max_gap else done).append(t)
        active = still
        preds = []
        for t in active:
            dx = cumulative_shift(motion, t["frames"][-1], f)
            b = t["boxes"][-1]
            preds.append([b[0] + dx, b[1], b[2] + dx, b[3]])       # where physics says the object is now
        cands = []
        for i, (b, _) in enumerate(dets[f]):
            for j, (t, p) in enumerate(zip(active, preds)):
                if same_class and t["boxes"][-1][4] != b[4]:
                    continue
                ov = iou(p, b)
                dist = abs((p[0] + p[2]) / 2 - (b[0] + b[2]) / 2) + abs((p[1] + p[3]) / 2 - (b[1] + b[3]) / 2)
                if ov > 0.1 or dist < 64:
                    cands.append((-ov, dist, i, j))
        used_d, used_t = set(), set()
        for _, _, i, j in sorted(cands):
            if i not in used_d and j not in used_t:
                b, pl = dets[f][i]
                active[j]["frames"].append(f); active[j]["boxes"].append(b); active[j]["payloads"].append(pl)
                used_d.add(i); used_t.add(j)
        for i, (b, pl) in enumerate(dets[f]):
            if i not in used_d:                                          # new object enters
                active.append({"frames": [f], "boxes": [b], "payloads": [pl]})
    return done + active


def box_cells(box):
    """Grid cells covered by a box (at least one)."""
    c1, c2 = int(max(0, box[0]) // CELL_W_ORIG), int(np.ceil(min(IMG_W, box[2]) / CELL_W_ORIG))
    r1, r2 = int(max(0, box[1]) // CELL_H), int(np.ceil(min(IMG_H, box[3]) / CELL_H))
    m = np.zeros((GRID_H, GRID_W), bool)
    m[r1:max(r2, r1 + 1), c1:max(c2, c1 + 1)] = True
    return m


def pool(feat, mask):
    """Token for one object in one frame: mean and max of its ResNet cells (256 + 256 = 512 numbers)."""
    v = feat[:, torch.from_numpy(mask).to(feat.device)]
    return torch.cat([v.mean(1), v.amax(1)]).half().cpu().numpy()


def random_glare_on(box):
    return {"cx": int(random.uniform(max(0, box[0]), min(IMG_W, box[2]))), "cy": int(random.uniform(box[1], box[3])),
            "r": random.randint(90, 200), "a": round(random.uniform(0.7, 1.0), 2)}


def load_img(video, fid, glare=None):
    img = cv2.imread(frame_path(video, fid))
    if glare:
        img, _ = apply_glare(img, glare)
    return cv2.resize(img, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)


backbone = Backbone(dev)

# ================================================================ 1. training tracks (labeled objects + ore)
TRACKS = os.path.join(ROOT, "data", "objects", f"tracks_{tag}.pt")
if os.path.exists(TRACKS):
    tracks = torch.load(TRACKS, weights_only=False)
else:
    boxes = json.load(open(os.path.join(ROOT, "data", "labels", "boxes.json")))
    by_video = defaultdict(dict)
    for k, bxs in boxes.items():
        v, f = k.split("/")
        by_video[v][int(f)] = [(b, None) for b in bxs]
    tracks = []
    for v, dets in sorted(by_video.items()):                     # link labeled boxes into objects
        for t in link(dets, load_motion(v), max_gap=3, same_class=True):
            if len(t["frames"]) >= 4:
                tracks.append({"video": v, "split": split_of(v), "cls": CLASSES.index(t["boxes"][0][4]),
                               "frames": t["frames"], "boxes": [b[:4] for b in t["boxes"]]})
    n_obj = len(tracks)
    for v in NORMAL_VIDEOS:                                       # ore tracks: random regions ridden by the belt
        motion, fr = load_motion(v), [f for f, _ in frames_of(v)]
        have = set(fr)
        for _ in range(8):
            w, h = random.randint(150, 400), random.randint(120, 300)
            x1, y1 = random.randint(500, IMG_W - w), random.randint(BELT_Y0 + 40, BELT_Y1 - h - 20)
            f0 = random.choice(fr[:-150])
            fs, bs = [], []
            for f in range(f0, f0 + 150):
                if f not in have:                                     # gap in the frame numbering
                    break
                dx = cumulative_shift(motion, f0, f)
                if x1 + w + dx < 60:
                    break
                fs.append(f); bs.append([x1 + dx, y1, x1 + w + dx, y1 + h])
            tracks.append({"video": v, "split": split_of(v), "cls": 0, "frames": fs, "boxes": bs})
    # features: every EVERY-th frame, clean and with glare on the object
    for t in tracks:
        idx = list(range(0, len(t["frames"]), EVERY))
        if len(idx) > MAX_TRACK:
            idx = [idx[int(i)] for i in np.linspace(0, len(idx) - 1, MAX_TRACK)]
        t["frames"] = [t["frames"][i] for i in idx]; t["boxes"] = [t["boxes"][i] for i in idx]
        clean, glared = [], []
        for f, b in zip(t["frames"], t["boxes"]):
            feats = backbone([load_img(t["video"], f), load_img(t["video"], f, random_glare_on(b))])
            m = box_cells(b)
            clean.append(pool(feats[0], m)); glared.append(pool(feats[1], m))
        t["clean"], t["glare"] = np.stack(clean), np.stack(glared)
    torch.save(tracks, TRACKS)
    print(f"built {n_obj} object tracks + {len(tracks) - n_obj} ore tracks")

for sp in ["train", "val", "test"]:
    c = Counter(CLASSES[t["cls"]] for t in tracks if t["split"] == sp)
    print(f"{sp:5s} tracks: " + ", ".join(f"{k} {c[k]}" for k in CLASSES))


# ================================================================ 2. models
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(0.1), nn.Linear(2 * d, d))

    def forward(self, x, pad):
        a, w = self.att(x, x, x, key_padding_mask=pad, need_weights=True)
        x = self.n1(x + a)
        return self.n2(x + self.ff(x)), w


class SeqTransformer(nn.Module):
    """[CLS] + one token per frame -> 2 transformer layers -> class from [CLS]."""

    def __init__(self, d=96, heads=4, layers=2, max_len=80):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(512, d), nn.LayerNorm(d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, max_len + 1, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.head = nn.Linear(d, len(CLASSES))

    def forward(self, x, pad):
        B, L, _ = x.shape
        h = torch.cat([self.cls.expand(B, 1, -1), self.inp(x)], 1) + self.pos[:, :L + 1]
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad], 1)
        for blk in self.blocks:
            h, w = blk(h, pad)
        self.last_attn = w[:, 0, 1:]                     # how much [CLS] attended to each frame (last layer)
        return self.head(h[:, 0])


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 96), nn.ReLU(), nn.Dropout(0.2), nn.Linear(96, len(CLASSES)))

    def forward(self, x):
        return self.net(x)


class MeanPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()

    def forward(self, x, pad):
        keep = (~pad).unsqueeze(-1).float()
        return self.mlp((x * keep).sum(1) / keep.sum(1).clamp(min=1))


def make_seq(t, p_glare, rng, sub=True):
    """Token sequence of a track; each frame glared with probability p_glare; random sub-sequence when training."""
    L = len(t["frames"])
    if sub and L > 4:
        n = rng.randint(4, L)
        s = rng.randint(0, L - n)
        idx = list(range(s, s + n))
    else:
        idx = list(range(L))
    toks = [t["glare"][i] if rng.random() < p_glare else t["clean"][i] for i in idx]
    return np.stack(toks).astype(np.float32), idx


def batch(seqs):
    L = max(len(s) for s in seqs)
    x = np.zeros((len(seqs), L, 512), np.float32)
    pad = np.ones((len(seqs), L), bool)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s; pad[i, :len(s)] = False
    return torch.from_numpy(x).to(dev), torch.from_numpy(pad).to(dev)


train_t = [t for t in tracks if t["split"] == "train"]
val_t = [t for t in tracks if t["split"] == "val"]
test_t = [t for t in tracks if t["split"] == "test"]
cnt = Counter(t["cls"] for t in train_t)
weight = torch.tensor([1.0 / max(cnt[c], 1) for c in range(len(CLASSES))], device=dev)
weight = weight / weight.sum() * len(CLASSES)


def train_seq(model, name):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    rng = random.Random(1)
    best, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        for _ in range(STEPS):
            ts = [rng.choice(train_t) for _ in range(16)]
            x, pad = batch([make_seq(t, rng.uniform(0, 0.5), rng)[0] for t in ts])
            loss = F.cross_entropy(model(x, pad), torch.tensor([t["cls"] for t in ts], device=dev), weight=weight)
            opt.zero_grad(); loss.backward(); opt.step()
        acc = evaluate_seq(model, val_t, 0.3)["type_acc_all"]
        if acc >= best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"{name:14s} best val accuracy {best:.2f}")
    return model


def train_frame(model):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    rng = random.Random(2)
    for ep in range(EPOCHS):
        model.train()
        for _ in range(STEPS):
            ts = [rng.choice(train_t) for _ in range(64)]
            x = []
            for t in ts:
                i = rng.randrange(len(t["frames"]))
                x.append(t["glare"][i] if rng.random() < 0.3 else t["clean"][i])
            x = torch.from_numpy(np.stack(x).astype(np.float32)).to(dev)
            loss = F.cross_entropy(model(x), torch.tensor([t["cls"] for t in ts], device=dev), weight=weight)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def predict_seq(model, seqs):
    model.eval()
    x, pad = batch(seqs)
    return model(x, pad).softmax(-1).cpu().numpy()


@torch.no_grad()
def predict_vote(frame_model, seqs, rule):
    """Per-object decision from per-frame predictions."""
    frame_model.eval()
    out = []
    for s in seqs:
        p = frame_model(torch.from_numpy(s).to(dev)).softmax(-1).cpu().numpy()     # (L, classes)
        p_imp = 1 - p[:, 0]
        if rule == "OR":
            imp = (p_imp > 0.5).any()
        elif rule == "k-of-N":
            imp = (p_imp > 0.5).mean() >= 0.3
        else:
            imp = p_imp.mean() > 0.5
        typ = 1 + int(p[:, 1:].mean(0).argmax())
        out.append(typ if imp else 0)
    return out


def evaluate_seq(model_or_rule, ts, p_glare, frame_model=None, seed=5):
    rng = random.Random(seed)
    seqs = [make_seq(t, p_glare, rng, sub=False)[0] for t in ts]
    if frame_model is not None:
        pred = predict_vote(frame_model, seqs, model_or_rule)
    else:
        pred = list(predict_seq(model_or_rule, seqs).argmax(1))
    y = [t["cls"] for t in ts]
    imp_true = np.array([c > 0 for c in y]); imp_pred = np.array([c > 0 for c in pred])
    return {"type_acc_all": float(np.mean([a == b for a, b in zip(pred, y)])),
            "impurity_found": f"{int((imp_true & imp_pred).sum())}/{int(imp_true.sum())}",
            "false_alarms": f"{int((~imp_true & imp_pred).sum())}/{int((~imp_true).sum())}",
            "type_acc_impurities": float(np.mean([a == b for a, b, it in zip(pred, y, imp_true) if it])) if imp_true.any() else float("nan")}


seq_model = train_seq(SeqTransformer().to(dev), "SeqTransformer")
mean_model = train_seq(MeanPool().to(dev), "MeanPool")
frame_model = train_frame(MLP().to(dev))

# ================================================================ 3. per-object accuracy on test tracks
lines = [f"# Per-object decisions — run `{tag}`", "",
         f"Test tracks: {sum(t['cls'] > 0 for t in test_t)} objects + {sum(t['cls'] == 0 for t in test_t)} ore regions "
         f"(test videos never seen in training). Up to {MAX_TRACK} frames per track, every {EVERY}th frame.", "",
         "| method | glare per frame | impurities found | false alarms on ore | correct type (impurities) | correct (all) |",
         "|---|---|---|---|---|---|"]
for pg in [0.0, 0.5]:
    for name, kw in [("SeqTransformer (ours)", dict(model_or_rule=seq_model)), ("MeanPool (average)", dict(model_or_rule=mean_model)),
                     ("FrameMLP + OR", dict(model_or_rule="OR", frame_model=frame_model)),
                     ("FrameMLP + k-of-N (30%)", dict(model_or_rule="k-of-N", frame_model=frame_model)),
                     ("FrameMLP + mean", dict(model_or_rule="mean", frame_model=frame_model))]:
        r = evaluate_seq(ts=test_t, p_glare=pg, **kw)
        lines.append(f"| {name} | {int(pg * 100)}% | {r['impurity_found']} | {r['false_alarms']} | "
                     f"{r['type_acc_impurities']:.2f} | {r['type_acc_all']:.2f} |")

# ================================================================ 4. counting demo on the test videos
detector = build_model(args.detector_mode, VARIANTS["base"]).to(dev)
detector.load_state_dict(torch.load(os.path.join(ROOT, "data", "models", args.detector, f"{args.detector_mode}.pt"),
                                    map_location=dev, weights_only=True))
detector.eval()
LAMP = {"cx": 700, "cy": 330, "r": 150, "a": 0.95}               # a fixed overhead-lamp reflection (does NOT move with the belt)
gt_boxes = json.load(open(os.path.join(ROOT, "data", "labels", "boxes.json")))

lines += ["", "## Counting on the test videos", "",
          f"Every {DEMO_EVERY}th frame of each test video, detector = `{args.detector}/{args.detector_mode}`. "
          "**Lamp** = a fixed glare spot painted on every frame (does not move with the belt).", "",
          "| video | lamp | true objects | raw detections (frames x blobs) | tracks (>= 3 detections) | counted impurities (transformer) | counted by type |",
          "|---|---|---|---|---|---|---|"]
attn_example = None
for v in SPLITS["test"]:
    gt_dets = defaultdict(list)
    for k, bxs in gt_boxes.items():
        if k.startswith(v + "/"):
            gt_dets[int(k.split("/")[1])] = [(b, None) for b in bxs]
    gt_tracks = [t for t in link(gt_dets, load_motion(v), 3, True) if len(t["frames"]) >= 4] if gt_dets else []
    true_count = Counter(t["boxes"][0][4] for t in gt_tracks)
    for lamp in [False, True]:
        fr = [f for f, _ in frames_of(v)][::DEMO_EVERY]
        dets, n_raw = {}, 0
        for i in range(0, len(fr), 16):
            fs = fr[i:i + 16]
            feats = backbone([load_img(v, f, LAMP if lamp else None) for f in fs])
            with torch.no_grad():
                prob = torch.sigmoid(detector(feats.unsqueeze(1).expand(-1, 3, -1, -1, -1).float(),
                                              torch.zeros(len(fs), 3, device=dev))).cpu().numpy()
            for f, fe, p in zip(fs, feats, prob):
                n, lab = cv2.connectedComponents((p > 0.5).astype(np.uint8), connectivity=8)
                dets[f] = []
                for k in range(1, n):
                    m = lab == k
                    if m.sum() < 2:
                        continue
                    rr, cc = np.nonzero(m)
                    box = [cc.min() * CELL_W_ORIG, rr.min() * CELL_H, (cc.max() + 1) * CELL_W_ORIG, (rr.max() + 1) * CELL_H]
                    dets[f].append((box, pool(fe, m)))
                    n_raw += 1
        tr = [t for t in link(dets, load_motion(v), max_gap=2 * DEMO_EVERY + 1) if len(t["frames"]) >= MIN_LEN]
        types = Counter()
        if tr:
            seqs = []
            for t in tr:
                toks = np.stack(t["payloads"]).astype(np.float32)
                if len(toks) > 80:
                    toks = toks[np.linspace(0, len(toks) - 1, 80).astype(int)]
                seqs.append(toks)
            pred = predict_seq(seq_model, seqs).argmax(1)
            for t, c in zip(tr, pred):
                t["type"] = CLASSES[c]
                if c > 0:
                    types[CLASSES[c]] += 1
            # picture: a frame in the middle of the longest counted track, all active tracks drawn
            counted = [t for t in tr if t["type"] != "ore"] or tr
            ref = max(counted, key=lambda t: len(t["frames"]))
            f_mid = ref["frames"][len(ref["frames"]) // 2]
            img = cv2.imread(frame_path(v, f_mid))
            if lamp:
                img, _ = apply_glare(img, LAMP)
            for n_t, t in enumerate(tr):
                if f_mid in t["frames"]:
                    b = t["boxes"][t["frames"].index(f_mid)]
                    col = (0, 0, 255) if t["type"] == "ore" else (0, 255, 0)
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 4)
                    put_label(img, f"track {n_t}: {t['type']} ({len(t['frames'])} frames)", (int(b[0]) + 4, int(b[1]) + 30), 0.8)
            put_label(img, f"{v.replace('boardcamera_', '')} frame {f_mid}" + ("  + lamp glare" if lamp else ""), (10, 700), 1.0)
            cv2.imwrite(os.path.join(OUT, f"counting_{v.replace('boardcamera_', '')}{'_lamp' if lamp else ''}.jpg"), img)
        lines.append(f"| {v.replace('boardcamera_', '')} | {'yes' if lamp else 'no'} | "
                     f"{sum(true_count.values())} ({', '.join(f'{k} {n}' for k, n in true_count.items()) or 'none'}) | "
                     f"{n_raw} | {len(tr)} | {sum(types.values())} | {', '.join(f'{k} {n}' for k, n in types.items()) or '-'} |")

# ================================================================ 5. attention picture for one test object
obj = [t for t in test_t if t["cls"] > 0]
if obj:
    t = max(obj, key=lambda t: len(t["frames"]))
    rng = random.Random(7)
    toks, idx = make_seq(t, 0.4, rng, sub=False)
    glared = [not np.array_equal(toks[i], t["clean"][i].astype(np.float32)) for i in range(len(idx))]
    probs = predict_seq(seq_model, [toks])[0]
    att = seq_model.last_attn[0, :len(idx)].cpu().numpy()
    show = np.linspace(0, len(idx) - 1, min(12, len(idx))).astype(int)
    cells = []
    for i in show:
        f, b = t["frames"][i], t["boxes"][i]
        img = cv2.imread(frame_path(t["video"], f))
        if glared[i]:
            rng2 = random.Random(f)
            img, _ = apply_glare(img, {"cx": int((max(0, b[0]) + min(IMG_W, b[2])) / 2), "cy": int((b[1] + b[3]) / 2), "r": 140, "a": 0.9})
        x1, x2 = int(max(0, b[0])), int(min(IMG_W, b[2]))
        crop = cv2.resize(img[int(b[1]):int(b[3]), x1:max(x2, x1 + 10)], (200, 160))
        bar = np.full((40, 200, 3), 255, np.uint8)
        cv2.rectangle(bar, (0, 10), (int(200 * min(1, att[i] / att.max())), 30), (0, 140, 255), -1)
        put_label(crop, f"{f}" + (" GLARE" if glared[i] else ""), (4, 22), 0.55)
        put_label(bar, f"attn {att[i] * 100:.0f}%", (4, 28), 0.55)
        cells.append(np.vstack([crop, bar]))
    strip = np.hstack(cells)
    title = np.full((50, strip.shape[1], 3), 40, np.uint8)
    put_label(title, f"{t['video'].replace('boardcamera_', '')}: true {CLASSES[t['cls']]}, transformer says "
                     f"{CLASSES[int(probs.argmax())]} ({probs.max() * 100:.0f}%). Bars = attention of [CLS] on each frame", (8, 34), 0.7)
    cv2.imwrite(os.path.join(OUT, "track_attention.jpg"), np.vstack([title, strip]))

open(os.path.join(OUT, "results.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines).replace("—", "-"))
print("pictures in", OUT)

"""Step 5 - Run every photo ONCE through a frozen, pre-trained ResNet-18 and save the result to disk.

What ResNet gives us: for each of the 40x22 grid squares, a list of 256 numbers describing what that
patch looks like (colour, edges, texture). Similar-looking patches get similar lists.
ResNet is never trained here ("frozen"), so the output for a photo never changes -> compute once, save.

Usage:
  .venv\\Scripts\\python src\\05_extract_features.py --limit 100   # smoke test: 100 train samples (+ some val/test)
  .venv\\Scripts\\python src\\05_extract_features.py               # everything

Output:
  data/samples/subset_<tag>.json              which samples belong to this run
  data/features/all/<key>.npy                 256 x 22 x 40 float16 per photo
  data/features/all/<key>.white.npy           22 x 40 share of near-white pixels per square (glare-skip baseline, E9)
  outputs/05_features/<tag>/what_resnet_sees_XX.jpg
"""
import os
import json
import argparse
import random
import numpy as np
import cv2
import torch
import torchvision
from tqdm import tqdm

from common import (ROOT, run_tag, sample_feat_keys, load_model_frame, put_label, MODEL_W, MODEL_H,
                    GRID_W, GRID_H, VARIANTS, FEAT_DIR, frame_glare, t_index, white_grid)

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0, help="number of TRAIN samples (0 = all)")
ap.add_argument("--variant", default="base", choices=list(VARIANTS), help="base = 3 views; T8w7 = 8 views (step 04b)")
ap.add_argument("--batch", type=int, default=16)
args = ap.parse_args()
tag = run_tag(args.limit, args.variant)

# ---------------------------------------------------------------- 1. choose samples for this run
samples = json.load(open(os.path.join(ROOT, "data", "samples", VARIANTS[args.variant]["samples"])))
if args.limit:
    random.seed(0)
    want = {"train": args.limit, "val": max(20, args.limit // 4), "test": max(40, args.limit // 2)}
    subset = []
    for split, n in want.items():
        pos = [s for s in samples if s["split"] == split and s["grid_ones"]]
        neg = [s for s in samples if s["split"] == split and not s["grid_ones"]]
        subset += random.sample(pos, min(len(pos), n // 2)) + random.sample(neg, min(len(neg), n - n // 2))
else:
    subset = samples
json.dump(subset, open(os.path.join(ROOT, "data", "samples", f"subset_{tag}.json"), "w"))
print(f"run '{tag}': " + ", ".join(f"{sp} {sum(s['split'] == sp for s in subset)}" for sp in ["train", "val", "test"]))

# every distinct photo (video, frame, glare) that these samples need
jobs = {}
for s in subset:
    for slot, key in enumerate(sample_feat_keys(s)):
        jobs[key] = (s["video"], s["frames"][slot], frame_glare(s, slot))
FDIR = FEAT_DIR   # shared cache: a photo already processed for another run/variant is reused
os.makedirs(FDIR, exist_ok=True)
todo = [k for k in jobs if not os.path.exists(os.path.join(FDIR, k + ".npy"))]
print(f"{len(jobs)} photos needed, {len(todo)} not cached yet")

# ---------------------------------------------------------------- 2. the frozen ResNet-18
# Pretrained on ImageNet (1.2M everyday photos). We keep the layers up to 'layer3':
#   640x352 photo -> conv1/maxpool (/4) -> layer1 -> layer2 (/8) -> layer3 (/16) -> 256 x 22 x 40
dev = "cuda" if torch.cuda.is_available() else "cpu"
resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
backbone = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                               resnet.layer1, resnet.layer2, resnet.layer3).eval().to(dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)   # ImageNet colour statistics
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


@torch.no_grad()
def extract(imgs_bgr):
    x = torch.from_numpy(np.stack(imgs_bgr)[..., ::-1].copy()).to(dev)     # BGR -> RGB
    x = x.permute(0, 3, 1, 2).float() / 255.0
    x = (x - MEAN) / STD
    return backbone(x).half().cpu().numpy()                                 # B x 256 x 22 x 40


for i in tqdm(range(0, len(todo), args.batch), desc="ResNet"):
    keys = todo[i:i + args.batch]
    imgs = [load_model_frame(*jobs[k]) for k in keys]
    for k, f, im in zip(keys, extract(imgs), imgs):
        np.save(os.path.join(FDIR, k + ".npy"), f)
        np.save(os.path.join(FDIR, k + ".white.npy"), white_grid(im).astype(np.float16))

# photos cached before the glare-skip baseline existed: add only their white-pixel grid
todo_w = [k for k in jobs if not os.path.exists(os.path.join(FDIR, k + ".white.npy"))]
for k in tqdm(todo_w, desc="white grid"):
    np.save(os.path.join(FDIR, k + ".white.npy"), white_grid(load_model_frame(*jobs[k])).astype(np.float16))
size_mb = sum(os.path.getsize(os.path.join(FDIR, f)) for f in os.listdir(FDIR)) / 1e6
print(f"features in {FDIR}  ({size_mb:.0f} MB, one file = 256x{GRID_H}x{GRID_W} numbers)")

# ---------------------------------------------------------------- 3. picture: what does ResNet "see"?
# 256 numbers per square cannot be drawn. PCA squeezes them to the 3 most important directions,
# which we show as R, G, B. Squares that ResNet finds similar get similar colours.
OUT = os.path.join(ROOT, "outputs", "05_features", tag)
os.makedirs(OUT, exist_ok=True)
show = [s for s in subset if s["grid_ones"]][:4] + [s for s in subset if not s["grid_ones"]][:1]
feats = [np.load(os.path.join(FDIR, sample_feat_keys(s)[t_index(s)] + ".npy")).astype(np.float32) for s in show]
X = np.concatenate([f.reshape(256, -1).T for f in feats])          # all squares x 256
mu = X.mean(0)
_, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
for i, (s, f) in enumerate(zip(show, feats)):
    p = (f.reshape(256, -1).T - mu) @ Vt[:3].T                      # squares x 3
    p = (p - p.min(0)) / (p.max(0) - p.min(0) + 1e-6)
    pca_img = (p.reshape(GRID_H, GRID_W, 3) * 255).astype(np.uint8)
    pca_img = cv2.resize(pca_img, (MODEL_W, MODEL_H), interpolation=cv2.INTER_NEAREST)
    photo = load_model_frame(s["video"], s["t"], frame_glare(s, t_index(s)))
    put_label(photo, f"photo (t={s['t']})", (8, 30), 0.8)
    put_label(pca_img, "ResNet features (3 main directions as colour)", (8, 30), 0.7)
    cv2.imwrite(os.path.join(OUT, f"what_resnet_sees_{i:02d}.jpg"), np.hstack([photo, pca_img]))
print("pictures in", OUT)

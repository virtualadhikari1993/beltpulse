"""PyTorch Dataset that serves cached ResNet features (steps 06, 07, 07b, 08)."""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from common import sample_feat_keys, glare_grid, target_glare, GRID_W, GRID_H, CELL_W_ORIG, FEAT_DIR


class FeatureSamples(Dataset):
    """One item = (features of the T views, belt shift in grid cells, 0/1 answer grid, glare mask of the target view)
    and, with with_white=True, a 5th entry: share of near-white pixels per square of each view (T, 22, 40),
    used by the 'glareskip' baseline (written by step 05)."""

    def __init__(self, samples, fdir=FEAT_DIR, with_white=False):
        self.samples, self.fdir, self.with_white = samples, fdir, with_white

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        feats = np.stack([np.load(os.path.join(self.fdir, k + ".npy")) for k in sample_feat_keys(s)])
        grid = np.zeros((GRID_H, GRID_W), np.float32)
        for r, c in s["grid_ones"]:
            grid[r, c] = 1
        shift = np.array(s["shift_px"], np.float32) / CELL_W_ORIG          # pixels -> grid cells
        g = target_glare(s)
        gmask = glare_grid(g) if g else np.zeros((GRID_H, GRID_W), np.uint8)
        out = (torch.from_numpy(feats).float(), torch.from_numpy(shift), torch.from_numpy(grid), torch.from_numpy(gmask))
        if self.with_white:
            white = np.stack([np.load(os.path.join(self.fdir, k + ".white.npy")) for k in sample_feat_keys(s)])
            out += (torch.from_numpy(white).float(),)
        return out

"""BeltPulse-Net model (the part that is actually trained) + the baselines.

Input : ResNet features of T views     (B, T, 256, 22, 40)   e.g. base: t-k, t, t+k   T8w7: t-3k ... t+4k
        belt shift of each view        (B, T)  in grid cells (content of frame t is at x+shift in that view)
Output: one score per grid square of the target view t (B, 22, 40); sigmoid(score) = probability "foreign object here"

Pipeline inside the model:
  1. Bottleneck ("encoder")   : 1x1 convolutions squeeze 256 numbers per square to D=96
  2. Kinematic alignment      : slide every view by its belt shift so an object riding on the belt sits
                                in the same square in all T grids (fixed maths, nothing learned)
  3. Cross-frame attention    : each square of view t looks at a win x win neighbourhood in all T aligned
                                views (T*win*win tokens) and decides how much to trust each one.
                                - win=3 absorbs small alignment errors (belt jitter)
                                - win=7 (+-3 squares, ~+-100 px) lets attention FIND an object that rolled or
                                  slid away from where physics put it (extension E10)
                                - many views (T=8): attention can skip glared / blurred / out-of-view ones (E11)
  4. Spatial transformer      : all 880 squares of view t look at each other (global belt context)
  5. Head                     : one score per square

Modes (for comparison):
  beltpulse : everything above
  noalign   : same, but skip step 2 (shift = 0)  -> shows whether the kinematics matter
  avg       : align, then simply AVERAGE the T views (no attention across frames)
  median    : align, then take the MEDIAN of the T views per square and feature (E9: a glare seen in only
              a minority of the views is removed without any training of "which view to trust")
  glareskip : align, then average only the views that are NOT washed out at this square (>= 50% near-white
              pixels, fixed rule, E9). If a square is white in every view (e.g. white plastic), all views are kept.
  single    : only view t                          -> the classic single-frame approach
  (Voting baselines need no extra model: step 07 runs `single` on every view and combines the answers.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import WHITE_FRAC

MODES = ["single", "avg", "median", "glareskip", "noalign", "beltpulse"]
FUSE_MODES = ("avg", "median", "glareskip")   # fixed-rule fusion, then the same network as `single`


def kinematic_align(x, shift_cells):
    """x: (B, T, D, H, W); shift_cells: (B, T). Returns aligned x and a validity mask (B, T, W).
    aligned[..., col] = x[..., col + shift]  (bilinear for fractional shifts, zeros outside the image)."""
    B, T, D, H, W = x.shape
    cols = torch.arange(W, device=x.device, dtype=x.dtype)
    rows = torch.arange(H, device=x.device, dtype=x.dtype)
    src_col = cols.view(1, 1, W) + shift_cells.view(B, T, 1).to(x.dtype)       # (B, T, W)
    gx = (src_col / max(W - 1, 1)) * 2 - 1
    gy = (rows / max(H - 1, 1)) * 2 - 1
    grid = torch.stack(torch.broadcast_tensors(gx.view(B, T, 1, W), gy.view(1, 1, H, 1)), dim=-1)
    out = F.grid_sample(x.reshape(B * T, D, H, W), grid.reshape(B * T, H, W, 2),
                        mode="bilinear", padding_mode="zeros", align_corners=True)
    valid = (src_col >= 0) & (src_col <= W - 1)
    return out.view(B, T, D, H, W), valid


class LocalCrossFrameAttention(nn.Module):
    """Multi-head attention where the query is one square of view t and the keys/values are the
    win x win neighbourhood of that square in each of the Tn aligned views (Tn * win² tokens).

    Same maths as nn.MultiheadAttention over unfolded patches, but computed by sliding whole feature
    maps (no copy of every patch), so 8 views x 7x7 windows fit in 4 GB of GPU memory."""

    def __init__(self, d, heads, T, win):
        super().__init__()
        self.h, self.hd, self.win = heads, d // heads, win
        self.q, self.k, self.v = nn.Conv2d(d, d, 1), nn.Conv2d(d, d, 1), nn.Conv2d(d, d, 1)
        self.o = nn.Linear(d, d)
        self.time_emb = nn.Parameter(torch.randn(T, d) * 0.02)            # "which view is this token from"
        self.offset_emb = nn.Parameter(torch.randn(win * win, d) * 0.02)  # "where in the window"

    def forward(self, x, valid, t_loc, times):
        """x: (B, Tn, d, H, W) aligned views; valid: (B, Tn, W); returns (B, H, W, d) and attention (B, Tn*win², H, W)."""
        B, Tn, d, H, W = x.shape
        h, hd, win, r = self.h, self.hd, self.win, self.win // 2
        emb = self.time_emb[times][:, None, :] + self.offset_emb[None]             # (Tn, win², d)
        q_in = x[:, t_loc] + emb[t_loc, (win * win) // 2].view(1, d, 1, 1)       # query token = square itself
        q = self.q(q_in).view(B, h, hd, H, W) * hd ** -0.5
        xf = x.reshape(B * Tn, d, H, W)
        k, v = self.k(xf).view(B, Tn, d, H, W), self.v(xf).view(B, Tn, d, H, W)
        kb = F.linear(emb, self.k.weight[:, :, 0, 0]).view(Tn, win * win, h, hd)   # embedding part of each key
        vb = F.linear(emb, self.v.weight[:, :, 0, 0]).view(Tn, win * win, h, hd)

        def patches(t):  # (B, d, H, W) -> (B, h, hd, win², H, W): the win x win neighbourhood of every square
            return F.unfold(t, win, padding=r).view(B, h, hd, win * win, H, W)

        # scores: one view at a time, all win² positions at once
        S = torch.cat([torch.einsum("bhcyx,bhcoyx->bhoyx", q, patches(k[:, f])) +
                       torch.einsum("bhcyx,ohc->bhoyx", q, kb[f]) for f in range(Tn)], dim=2)   # (B, h, Tn*win², H, W)
        m = (~valid).clone()
        m[:, t_loc] = False                                                        # never mask the target view
        m = m.view(B, 1, Tn, 1, 1, W).expand(B, 1, Tn, win * win, 1, W).reshape(B, 1, Tn * win * win, 1, W)
        A = S.masked_fill(m, float("-inf")).softmax(2)

        out = 0
        for f in range(Tn):                                                        # weighted sum of values
            Af = A[:, :, f * win * win:(f + 1) * win * win]
            out = out + torch.einsum("bhoyx,bhcoyx->bhcyx", Af, patches(v[:, f])) + torch.einsum("bhoyx,ohc->bhcyx", Af, vb[f])
        out = self.o(out.reshape(B, d, H, W).permute(0, 2, 3, 1))                  # (B, H, W, d)
        return q_in.permute(0, 2, 3, 1), out, A.mean(1)


class BeltPulseNet(nn.Module):
    def __init__(self, mode="beltpulse", c_in=256, d=96, heads=4, H=22, W=40, win=3, T=3):
        super().__init__()
        assert mode in MODES
        self.mode, self.d, self.win, self.H, self.W, self.T = mode, d, win, H, W, T
        self.t_idx = (T - 1) // 2                                   # position of the target view
        # 1. bottleneck
        self.proj = nn.Sequential(nn.Conv2d(c_in, d, 1), nn.BatchNorm2d(d), nn.ReLU(), nn.Conv2d(d, d, 1))
        # 3. cross-frame attention
        self.temporal = LocalCrossFrameAttention(d, heads, T, win)
        self.norm1 = nn.LayerNorm(d)
        self.ff1 = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.norm2 = nn.LayerNorm(d)
        # 4. spatial transformer over the 22x40 squares
        self.pos = nn.Parameter(torch.zeros(1, H * W, d))
        self.spatial = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True)
        # 5. head
        self.head = nn.Linear(d, 1)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, feats, shift_cells, return_attn=False, white=None):
        """white: (B, T, H, W) share of near-white pixels per square, needed only for mode 'glareskip'."""
        B, T, C, H, W = feats.shape
        d, win = self.d, self.win
        x = self.proj(feats.flatten(0, 1)).view(B, T, d, H, W)                  # 1. bottleneck
        ti = self.t_idx

        if self.mode == "single":
            x, valid, t_loc, times = x[:, ti:ti + 1], torch.ones(B, 1, W, dtype=torch.bool, device=x.device), 0, [ti]
        else:
            sh = shift_cells if self.mode != "noalign" else torch.zeros_like(shift_cells)
            x, valid = kinematic_align(x, sh)                                    # 2. alignment
            t_loc, times = ti, list(range(T))
            if self.mode in FUSE_MODES:
                v = valid.view(B, T, 1, 1, W).expand(B, T, 1, H, W)
                if self.mode == "glareskip":
                    assert white is not None, "mode 'glareskip' needs the white-pixel grids (FeatureSamples(with_white=True))"
                    wa, _ = kinematic_align(white.unsqueeze(2).to(x.dtype), sh)            # (B, T, 1, H, W)
                    keep = v & (wa < WHITE_FRAC)
                    v = torch.where(keep.any(1, keepdim=True), keep, v)                    # all views white -> keep all
                if self.mode == "median":
                    x = torch.where(v, x, torch.full_like(x, float("nan"))).nanmedian(1, keepdim=True).values
                else:
                    v = v.to(x.dtype)
                    x = (x * v).sum(1, keepdim=True) / v.sum(1, keepdim=True).clamp(min=1)
                valid, t_loc, times = torch.ones(B, 1, W, dtype=torch.bool, device=x.device), 0, [ti]
        Tn = x.shape[1]

        # 3. cross-frame attention: each square of view t vs. its win x win neighbourhood in all Tn views.
        #    Tokens from a view where this square is outside the image (after shifting) are ignored.
        q, out, attn = self.temporal(x, valid, t_loc, times)                     # (B,H,W,d), (B,H,W,d), (B,Tn*win²,H,W)
        h = self.norm1(q + out)
        h = self.norm2(h + self.ff1(h)).view(B, H * W, d)

        h = self.spatial(h + self.pos)                                           # 4. global context
        logits = self.head(h).view(B, H, W)                                      # 5. score per square
        if return_attn:
            # attention of every square on each of its Tn x win² patches: (B, H, W, Tn, win²)
            # (patch order inside the window: row by row, top-left first; centre = win²//2)
            self.last_token_attn = attn.permute(0, 2, 3, 1).reshape(B, H, W, Tn, win * win)
            per_frame = self.last_token_attn.sum(-1)                             # (B, H, W, Tn)
            return logits, per_frame
        return logits


def build_model(mode, variant_cfg):
    return BeltPulseNet(mode, win=variant_cfg["win"], T=variant_cfg["T"])

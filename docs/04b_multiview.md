# 04b — Many views + wide attention window (extensions E10 + E11, variant `T8w7`)

**Scripts:** `src/04b_build_samples_multi.py`, then steps 05–08 with `--variant T8w7`
**Code:** `src/model.py` (`T`, `win` parameters; `LocalCrossFrameAttention`), `src/common.py` (`VARIANTS`)

## Why
With only 3 perfectly aligned views, plain averaging is hard to beat (`docs/proposal_review.md`). The transformer gets a real job when:
- **E11, many views:** with 8 views, some are glared, blurred, or show the object half outside the picture. Averaging mixes all of them in; attention can skip the bad ones.
- **E10, wide window:** physics alignment is exact only for objects lying still on the belt. Rolling or rotating objects are off by up to ~5 squares.
  With a 7×7 window (±3 squares ≈ ±100 px) attention can *search* for where the object really is. Averaging cannot.

## What changed
| | `base` | `T8w7` |
|---|---|---|
| views per example | 3: t−k, t, t+k | 8: t−3k … t+4k (target = view index 3) |
| k | object moves ~96 px between views | object moves ~64 px between views (~450 px over all 8) |
| attention window | 3×3 (27 tokens) | 7×7 (8 × 49 = 392 tokens per square) |
| glare | at most 1 view | each view independently with 30% chance (test: half the examples always glared on t) |
| examples (train/val/test) | 2,699 / 470 / 525 | 2,627 / 458 / 507 |
| batch size | 8 | 4 (GPU memory) |

Engineering note: the first version copied every patch (392 copies per square) and needed 3.8 GB at batch 4.
`LocalCrossFrameAttention` computes the same attention by sliding whole feature maps: 1.4 GB, 0.17 s per step.

## New in the evaluation (`07`) and pictures (`08`)
- `07` now also reports the **average attention a view gets where that view is glared vs. clean**.
  If the transformer learned to distrust glare, "glared" is clearly below "clean" (and below uniform 1/T).
- `08` draws all T views of one square's 7×7 window. The cyan dot = attention-weighted centre of the window, with its offset in squares.
  If attention follows a rolling object, the dot moves away from the centre.

## Smoke test (40 training samples, 3 epochs, ~1.5 min): proves it runs, numbers meaningless
| model | object IoU | IoU glare-on-t | recall under glare |
|---|---|---|---|
| single | 0.33 | 0.33 | 0.43 |
| avg | 0.43 | 0.43 | 0.56 |
| noalign | 0.31 | 0.31 | 0.44 |
| beltpulse | 0.35 | 0.34 | 0.61 |

Attention: glared views 0.156 vs clean 0.122 (uniform 0.125). After 3 epochs it has **not** learned to distrust glare yet.
`outputs/08_attention/smoke40_T8w7/square_00_A_glare.jpg` shows the alignment working (the stick sits in the same place in all 8 views)
and still-uniform attention. Needs the full run.

## Commands
```powershell
.venv\Scripts\python src\04b_build_samples_multi.py
.venv\Scripts\python src\05_extract_features.py --variant T8w7 [--limit 40]
.venv\Scripts\python src\06_train.py           --variant T8w7 [--limit 40 --epochs 3]
.venv\Scripts\python src\07_evaluate.py        --variant T8w7 [--limit 40]
.venv\Scripts\python src\08_explain_attention.py --variant T8w7 [--limit 40]
```
Full run estimate: features ~30 min (~7–8 GB), BeltPulse ~1.5 h, the 3 other models ~1–1.5 h.

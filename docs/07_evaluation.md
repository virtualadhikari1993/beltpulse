# 07 — Evaluation (the final exam)

**Script:** `src/07_evaluate.py [--limit N]`  **Output:** `outputs/07_eval/<tag>/metrics.md`, `examples_XX.jpg`, `attention.png`

Test videos were never used in training: `7_wood_3`, `canga3`, `jaspilito_4`, `things_3`, `iron_ore_4`.
(Until 2026-11-19 `jaspilito_3` was in test instead of `jaspilito_4`; the older smoke tables below used that split.)

## The metrics in plain words
| metric | meaning |
|---|---|
| **object IoU** | overlap ÷ union of predicted and true object squares. 1.0 = perfect, 0 = no overlap |
| **mIoU** | average of object IoU and background IoU (the metric named in the proposal) |
| **F1** | balances *precision* (flagged squares that are really objects) and *recall* (real object squares that were found) |
| **recall under glare** | of the object squares hidden under the glare blob in photo *t*, how many were still found. **This is where BeltPulse should win.** |

Each is reported for all test samples, for clean ones, and for ones whose middle photo *t* has glare.

## Full run, 3 views (2026-11-19): the first meaningful result
2,716 training samples, 30 epochs, one random seed. Test: 508 samples from 5 unseen videos. Full table: `outputs/07_eval/full/metrics.md`.

| model | object IoU | AP | IoU glare-on-t | recall under glare | real glare `jaspilito_4` IoU |
|---|---|---|---|---|---|
| single | 0.583 | 0.813 | 0.562 | 0.637 | 0.628 |
| single + OR vote | 0.585 | 0.856 | 0.574 | **0.911** | **0.742** |
| single + majority vote | 0.615 | 0.841 | 0.609 | 0.750 | 0.654 |
| single + mean vote | 0.620 | 0.847 | 0.614 | 0.778 | 0.678 |
| **avg** | **0.658** | **0.868** | **0.659** | 0.850 | 0.725 |
| median | 0.621 | 0.855 | 0.616 | 0.873 | 0.735 |
| glareskip | 0.630 | 0.839 | 0.637 | 0.814 | 0.716 |
| noalign | 0.615 | 0.834 | 0.612 | 0.789 | 0.704 |
| beltpulse | 0.635 | 0.849 | 0.636 | 0.769 | 0.699 |

**What it says about the three claims**
1. **Several views beat one view: confirmed, clearly.** Every multi-view model beats `single` (0.583). `avg` gains +0.075 IoU and finds 85% of the
   object squares hidden under glare vs 64%. On the real-glare video `jaspilito_4` multi-view gains about +0.1. The PR curve of `single` is the lowest everywhere.
2. **Kinematic alignment: helps, mostly on one video.** `beltpulse` vs `noalign`: +0.02 overall, but +0.12 on `canga3`
   (0.730 vs 0.612, and 1,469 vs 4,098 false-alarm squares). Without alignment the views disagree about where the object is.
3. **Learned attention beats simple fusion: not supported with 3 views.** `avg` (0.658) is best on IoU, AP and glare-on-t; `beltpulse` (0.635)
   is about level with `glareskip` (0.630) and `median` (0.621). Yet the attention **did** learn the intended behaviour: glared views get
   **0.160** of the weight vs 0.340 for clean ones (uniform 0.333). It learned to distrust glare, but that did not turn into better accuracy than plain averaging.
   Likely reason: with 2 clean views out of 3, averaging already dilutes the glare to 1/3, so there is little left to gain, and the extra flexibility costs some generalisation.

**Combine-then-decide beats decide-then-combine.** Feature averaging (`avg` 0.658) beats every vote on `single`'s answers (best: mean vote 0.620).
OR voting finds the most hidden squares (0.911 under glare) but doubles the false alarms (e.g. 8,159 vs 2,698 false-alarm squares on `things_3`).

**Cautions**
- One seed per model. Gaps below ~0.02 (`beltpulse` vs `glareskip` vs `median`) may be noise; `avg` vs `beltpulse` (0.023) needs 3 seeds to settle.
- Validation picked a different order (glareskip 0.748 > median 0.745 > avg 0.739 > beltpulse 0.725). Val is only 3 videos; test is the number to report.
- Val IoU ~0.73 vs test ~0.63: the test videos are harder or differ more from training.
- `jaspilito_4` has only 42 test samples; its per-video numbers are noisy.
- False alarms on the ore-only video `iron_ore_4` are tiny for every model (5–198 of 162,800 squares, at most 0.12%).
- Counting by blobs (`07b`) is weak: a single object is predicted as 1.63 regions on average, exact count right 55% of the time. Step 10 (tracking) is the remedy.

**Next:** the 8-view run (T8w7), where averaging is expected to break down (many glared views, rolling objects); 3 seeds for `avg` and `beltpulse`; E16 (is the spatial transformer needed?).

## Full run, 8 views + 7x7 window (T8w7, 2026-11-20)
2,644 train / 458 val / 490 test samples, 30 epochs, one seed. Training took 6.2 h (the two attention models ~100 min each).
Full table: `outputs/07_eval/full_T8w7/metrics.md`. The test samples differ slightly from the 3-view run (490 vs 508), so compare trends, not third decimals.

| model | val IoU | test object IoU | AP | glare-on-t | recall under glare | `jaspilito_4` | false alarms `iron_ore_4` |
|---|---|---|---|---|---|---|---|
| single | 0.723 | 0.622 | 0.838 | 0.611 | 0.808 | 0.705 | 220 |
| single + OR vote | - | 0.579 | 0.867 | 0.569 | **0.945** | 0.638 | 731 |
| single + 2-of-N vote | - | 0.643 | 0.878 | 0.639 | 0.910 | 0.699 | 416 |
| single + majority vote | - | 0.666 | 0.871 | 0.671 | 0.839 | 0.750 | 41 |
| single + mean vote | - | 0.671 | 0.877 | 0.674 | 0.850 | **0.756** | 53 |
| **avg** | **0.765** | **0.682** | **0.892** | **0.684** | 0.881 | 0.747 | 25 |
| median | 0.753 | 0.647 | 0.871 | 0.651 | 0.820 | 0.730 | 29 |
| glareskip | 0.723 | 0.640 | 0.847 | 0.645 | 0.829 | 0.538 | 21 |
| noalign | 0.716 | 0.626 | 0.842 | 0.620 | 0.833 | 0.678 | 0 |
| beltpulse | 0.748 | 0.666 | 0.871 | 0.662 | 0.850 | 0.751 | 518 |

**Findings**
1. **More views help every multi-view method.** `avg` 0.658 (3 views, seed 0) -> 0.682; `beltpulse` 0.626 (3-seed mean) -> 0.666; votes gain most
   (mean vote 0.620 -> 0.671), because with 8 views one glared view is outvoted 7 to 1.
2. **Alignment matters more with 8 views.** `beltpulse` vs `noalign`: +0.040 overall (3 views: +0.02), and `canga3` 0.726 vs 0.607.
   Views spread further in time, so unaligned views disagree more about where the object is.
3. **Attention still does not beat averaging.** `beltpulse` 0.666 is second among trained fusion models, level with the majority vote and below
   `avg` (0.682) and the mean vote (0.671). The gap to `avg` (0.016) is inside the seed spread seen with 3 views (sd 0.033 for `avg`), so with one seed this is a tie at best.
4. **The glare signal in attention got weaker.** Glared views get 0.088 vs 0.128 for clean ones (uniform 1/8 = 0.125): ratio 0.69, vs 0.47 with 3 views.
   With 8 views, most squares have plenty of clean views, so the model has less need to single out glare.
5. **`beltpulse` has the most false alarms on the ore-only video** (518 squares of ~160k, 0.3%; `avg` 25). A precision weakness to look at (threshold or per-video check).
6. **`glareskip` breaks on real glare** (`jaspilito_4` 0.538 vs ~0.75 for others): the fixed whiteness rule drops views that still carry the object.

**What can be claimed:** multi-view helps (confirmed, larger with 8 views); kinematic alignment helps (confirmed, larger with 8 views);
learned attention learns to down-weight glare but is not more accurate than feature averaging, at 3 or 8 views.
The case for attention would have to rest on something averaging cannot give: an interpretable per-view trust score, or rolling/occlusion cases (E15).

## Three seeds, 3 views (2026-11-20): is `avg` really better than `beltpulse`?
`avg` and `beltpulse` retrained with seeds 1 and 2 (`06 --seed N`, results in `outputs/07_eval/full_s1`, `full_s2`). Test object IoU:

| model | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| avg | 0.658 | 0.599 | 0.604 | 0.620 ± 0.033 |
| beltpulse | 0.635 | 0.629 | 0.614 | **0.626 ± 0.011** |

- Seed 0's lead for `avg` was luck: over three seeds the two are tied (gap 0.006, well inside the spread).
- `beltpulse` is about 3x more stable across random starts and usually gives fewer false alarms (e.g. `things_3`: ~2,900 vs 5,400-6,300 false-alarm squares on seeds 1-2).
- `avg` keeps higher recall under glare (0.85-0.87 vs 0.77-0.83).
- Claim 3 revised: with 3 views, attention **matches** averaging and is more consistent, but does not beat it. The 8-view run is the decisive test.

## Simple baselines added (extension E9, 2026-11-19)
Does learned attention beat **simple ways** of using several photos? Five baselines, so the transformer has to earn its place:

| baseline | how | trained? |
|---|---|---|
| `single + OR vote` | run `single` on every view, slide the answers into view *t*'s position, flag if **any** view says yes | no (reuses `single`) |
| `single + 2-of-N vote` | ... flag if **at least 2** views say yes (8-view variant only) | no |
| `single + majority vote` | ... flag if **more than half** of the views say yes | no |
| `single + mean vote` | ... average the probabilities | no |
| `median` | per square, the **median** of the aligned views' features. A glare seen in a minority of the views disappears | yes (same network as `single`) |
| `glareskip` | average only the views that are **not washed out** at this square (>= 50% of its pixels near-white, min channel >= 235). If the square is white in **every** view (white plastic stays white), all views are kept | yes |

The white-pixel threshold was measured: synthetic glare squares are 100% white, real frames have ~3-4% white squares (real lamp glare),
white plastic ~11-15% of its squares. The "white in every view → keep all" exception is what stops the rule from deleting white plastic.

New in `metrics.md`: **AP** (area under the precision-recall curve, all thresholds), a **per-video** table (object IoU / false-alarm squares),
and `pr_curves.png` (all models + votes; dashed = votes). A sorter chooses its own operating point, so compare curves, not only the 0.5 threshold.

**Smoke test (100 samples, new split): only proves the code runs.**

| model | object IoU | AP | IoU glare-on-t | recall under glare | false alarms on `iron_ore_4` (squares) |
|---|---|---|---|---|---|
| single | 0.526 | 0.767 | 0.469 | 0.429 | 11 |
| single + OR vote | 0.533 | 0.821 | 0.537 | **0.840** | 15 |
| single + majority vote | 0.563 | 0.816 | 0.519 | 0.595 | 0 |
| single + mean vote | 0.572 | 0.822 | 0.533 | 0.636 | 0 |
| avg | 0.529 | 0.799 | 0.533 | 0.807 | 1 |
| median | 0.593 | 0.817 | 0.583 | 0.753 | 0 |
| glareskip | 0.580 | 0.810 | 0.563 | 0.753 | 9 |
| noalign | 0.560 | 0.805 | 0.570 | 0.796 | 4 |
| beltpulse | 0.605 | 0.825 | 0.607 | 0.812 | 5 |

The textbook voting pattern is already visible: **OR** finds the most hidden object squares (0.84) but raises the most false alarms
(e.g. 1,005 false-alarm squares on `things_3` vs 429 for mean). **Majority / mean** are quiet but miss more under glare.
With 50 test samples and ±0.05 run-to-run noise, the ranking of the models is **not** meaningful yet.

## Smoke-test result after the label fixes (two-keyframe correction + 24 extra keyframes)

| model | object IoU (all) | mIoU | F1 | IoU clean | IoU glare-on-t | recall under glare |
|---|---|---|---|---|---|---|
| single | 0.538 | 0.750 | 0.699 | 0.562 | 0.498 | 0.647 |
| avg | 0.591 | 0.779 | 0.743 | 0.588 | 0.595 | 0.769 |
| noalign | 0.542 | 0.751 | 0.703 | 0.530 | 0.566 | 0.759 |
| beltpulse | 0.579 | 0.773 | 0.734 | 0.583 | 0.572 | 0.719 |

All models improved slightly with the better labels. Still only 100 training samples, so it is still not meaningful (see below).

## Earlier smoke-test result (100 training samples, original LLM labels): NOT meaningful yet

| model | object IoU (all) | mIoU | F1 | IoU clean | IoU glare-on-t | recall under glare |
|---|---|---|---|---|---|---|
| single | 0.506 | 0.733 | 0.672 | 0.515 | 0.491 | 0.651 |
| avg | 0.553 | 0.757 | 0.712 | 0.556 | 0.547 | 0.759 |
| noalign | 0.508 | 0.731 | 0.674 | 0.490 | 0.541 | 0.756 |
| beltpulse | 0.528 | 0.744 | 0.691 | 0.522 | 0.539 | 0.693 |

(An earlier training of the same beltpulse model gave 0.524 / … / recall under glare 0.598. Same code and data, only the random start differed.
So with this little data, **run-to-run noise is as big as the differences between models**.)

Why we should **not** draw conclusions from this table:
- 100 training examples are far too few. The models overfit (see `outputs/06_train/smoke100/curves.png`).
- 50 test examples: a handful of samples can swing these numbers by ±0.05.
- The labels are rough LLM boxes.

What the smoke test **does** show: every step runs, the shapes fit, the GPU works, and all models learn *something* (IoU ≈ 0.5 vs ≈ 0 at the start).

## Several objects in one photo (`src/07b_multi_object.py`)
The model labels every square impurity/ore, so **several objects simply give several red regions**; there is no limit on the number.
What it does **not** do:
- **tell objects apart.** It does not know which red squares belong to which object. We split the red squares into "blobs"
  (touching squares = one blob) to count them, but two objects that touch merge into one blob (e.g. canga lying on a plastic strip).
- **say what type** each object is. Training is "impurity yes/no"; the class names in the labels are not used yet.

Data: 156 samples have 2+ objects (train 63, val 49, test 44).

Smoke-test pictures (`outputs/07_eval/smoke100/multi_object_XX.jpg`, truth left, prediction right, one colour per blob):
- `multi_object_00`: 4 true objects → the big canga rock is found cleanly (red). The left canga + plastic strip merge into one green blob, and there are two small false alarms.
- `multi_object_01`: the canga lying on the plastic strip becomes **one** blue blob. The red region on the right is a canga rock entering the picture.
  It is **missing from the truth** (a label error from propagation), so the model is right and the label is wrong.

Counting is unreliable in the smoke test (single-object photos are predicted as ~1.9 regions on average, due to small stray blobs).

## Pictures
- `examples_XX.jpg`: 3 input photos | truth (green) | single-frame prediction | BeltPulse prediction (red).
  - `examples_00`: wood stick with glare on *t*. Both models find most of the stick, including part of the area under the glare.
  - `examples_02`: a small tool under glare. single-frame gets part of it, BeltPulse misses it. Small, rare objects need more data.
- `attention.png`: average attention BeltPulse gives each photo, for object squares with and without glare on *t*.
  **The hoped-for behaviour:** under glare, the bar for "photo t" drops and *t−k*/*t+k* rise.
  In the smoke run that is *not* happening yet (photo *t* still gets ~0.36). The model has not learned to distrust glare from 100 examples.
  This plot is the key one to check after the full run.

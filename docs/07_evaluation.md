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

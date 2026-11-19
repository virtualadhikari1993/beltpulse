# 04 — Building the training examples (3 frames + glare + answer grid)

**Script:** `src/04_build_samples.py`  **Output:** `data/samples/samples.json`, pictures in `outputs/04_samples/`
**Helpers in `src/common.py`:** `boxes_to_grid`, `apply_glare`, `load_model_frame`, `SPLITS`

## In plain words
A training example is like a small quiz for the model:
- **Question:** three photos of the belt: a bit earlier (*t−k*), now (*t*), a bit later (*t+k*). One of them may be spoiled by a fake glare spot.
- **Answer:** a 40 × 22 grid of 0/1 for the middle photo. 1 means "foreign object in this square".

## 1. Box → grid (`outputs/04_samples/box_to_grid.jpg`)
- The photo is shrunk to 640 × 352 (the CNN in step 05 needs a fixed size).
- The CNN reduces size by 16, so it "sees" the photo as **40 × 22 squares**. One square ≈ 32 × 32 pixels of the original photo.
- A square gets **1** if at least 25 % of it is inside a labeled box, otherwise **0**.
- On average only ~6 % of squares are 1. Most of the belt is ore, so the answer is mostly 0s.

## 2. Picking the neighbours *t−k* and *t+k*
We want the object to move about **3 squares (~96 px)** between the views:
- far enough that a glare spot does not cover the same part of the object in all three,
- close enough that the object is still in view.

So **k depends on belt speed**: fast belt → small k (as low as 2), slow or stopped belt → large k (up to 12).
`shift_px` records how far the content moved, for example `[+104, 0, −102]`. Step 06 uses it to line the three views up.

## 3. Fake glare (`outputs/04_samples/glare_examples.jpg`)
A soft white blob with a fully white centre, like an overhead lamp reflecting off a shiny surface.
- Size: radius 90–220 px. Strength: 70–100 %.
- 70 % of the time it is placed **on the object** (the hard case), otherwise anywhere on the belt.
- **Train / val:** 25 % of examples have no glare. The rest have glare on one random photo of the three.
- **Test:** half clean, half with glare **on the middle photo *t***. So at test time we can compare "clean" vs "middle photo blinded".
- Only the *recipe* (position, size, strength) is stored. The glare is painted again whenever the photo is loaded, so no extra image files are needed.

## 4. Positive and negative examples
- **Positive:** every 2nd labeled frame from step 03 (neighbouring frames are almost identical, so taking all of them adds little).
- **Negative:** every 6th frame from the 6 ore-only videos. The answer grid is all zeros. This teaches "dark rock = fine".

## 5. Train / validation / test split, by whole video
If frame 100 were in training and frame 101 in testing, the test would be far too easy, because the photos are near copies.
So each video goes entirely into one split:

| Split | Videos |
|---|---|
| test | `7_wood_3`, `canga3`, `jaspilito_4`, `things_3`, `iron_ore_4` (ore only) |

Split changed on 2026-11-19: `jaspilito_4` (real glare, motion blur, 2 rolling intervals) moved to test and `jaspilito_3` to train, so real difficult frames are judged on a video the models never trained on. New counts: train 2,716 / val 470 / test 508 samples.
| val | `canga`, `jaspilito_2`, `iron_ore_3` (ore only) |
| train | everything else |

Each object type (wood, canga, jaspilito, tools) appears in train **and** in test.

## Result

| split | examples | with object | ore only | with glare | object squares |
|---|---|---|---|---|---|
| train | 2,699 | 963 | 1,736 | 2,007 | 6.4 % |
| val | 470 | 166 | 304 | 354 | 3.9 % |
| test | 525 | 340 | 185 | 266 | 9.1 % |

The examples use 6,349 different photos (step 05 runs the CNN on each one).

## What one entry of `samples.json` looks like
```json
{"split": "train", "video": "boardcamera_iron_ore_7_wood", "t": 2515, "k": 9,
 "frames": [2506, 2515, 2524],
 "shift_px": [104.0, 0.0, -102.0],
 "glare": {"frame": 2, "cx": 1150, "cy": 300, "r": 140, "a": 0.9},
 "boxes": [[...]],
 "grid_ones": [[3, 0], [3, 1], ...]}
```
`grid_ones` lists the (row, column) of every square that is 1. Every square not listed is 0.

## Pictures to look at
- `sample_00.jpg`: jaspilito, belt almost stopped (shift only +7 px even with k = 12). Glare on *t−k*.
- `sample_01.jpg`: wood stick, shift ±100 px. Glare on *t+k* covers the stick's end, while the other two photos show it clearly.
- `sample_04.jpg`: an ore-only example. No red squares.

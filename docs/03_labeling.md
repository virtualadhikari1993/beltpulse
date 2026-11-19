# 03 — Labels: keyframes + kinematic propagation

**Scripts:** `src/03a_select_keyframes.py` → (label) → `src/03c_propagate_labels.py`
**Outputs:** `data/labels/`, `outputs/03_labeling/`

## Why
The dataset says *which* frames contain an impurity but not *where*. Drawing boxes on ~3,000 frames by hand
is slow. Instead we label a few **keyframes** and let the belt motion copy those boxes to the frames in between.

## Steps
1. **Passes** — consecutive anomalous frames are grouped into "passes" (one object crossing the view). 30 passes total.
2. **Keyframes (`03a`)** — inside each pass, a keyframe is taken every ~700 px of belt travel → **67 keyframes**.
   Each is saved with a 100-px coordinate grid in `outputs/03_labeling/keyframes/`.
3. **Label the keyframes** — boxes `[x1, y1, x2, y2, class]` in original 1280×720 pixels.
   Classes: `wood, plastic, canga, jaspilito, tool`. (The model will start with a single class "impurity".)
4. **Propagate (`03c`)** — each frame of a pass takes the boxes of its nearest keyframe (in belt travel),
   shifted by the summed dx. Boxes that leave the image are dropped; frames > 500 px of travel from any keyframe are left unlabeled.
   Result: **67 keyframes → 2,949 labeled frames, 3,323 boxes** in `data/labels/boxes.json`.

## Dry run: labels written by an LLM
For the first end-to-end run, Claude read the grid sheets (`src/03b_llm_label_sheets.py`, `outputs/03_labeling/sheets/`)
and wrote approximate boxes to `data/labels/keyframe_boxes_llm.json`. **These are rough** (±30–50 px).
They are good enough to see the whole pipeline working, not for final numbers.

## Rolling / sliding objects: two-keyframe correction + extra keyframes
Plain physics assumes the object sits still on the belt. Rocks roll and sticks rotate, so their copied boxes drift and then
**jump** when the next keyframe takes over. Two fixes (extension E8):

**1. Two-keyframe correction (in `03c`).** A frame between keyframes A and B gets
`position = (1−a)·(A's box moved by physics) + a·(B's box moved by physics)`, with `a = (frame−A)/(B−A)`.
At A it equals A's label, at B it equals B's label, and the error is spread smoothly in between (no jump).
Sides of a box cut off by the image border are taken from the other keyframe. Objects seen in only one keyframe (entering/leaving) use plain physics.
Before/after pictures: `outputs/03_labeling/correction_<video>.jpg` (red = plain physics, green = corrected).
Plain-physics labels are kept in `data/labels/boxes_plain.json` for comparison.

**2. Extra LLM keyframes where objects roll (`03f`).** `03c` compares, for each keyframe pair, B's real box with A's physics prediction.
More than 60 px off (more than the ~30–50 px label noise) → flagged in `data/labels/rolling_flags.csv`.
`03f` adds 1 extra keyframe in the middle of each flagged interval (3 if > 100 px off), and these are labeled directly (LLM),
**not** by physics. They count as ground truth in rolling regions, after your review.

Result of the first round: 14 flagged intervals (max 161 px off) → **24 extra keyframes** → re-run → 7 flagged intervals (max 120 px).
Running `03f` + labeling again would shrink this further.
Many flags are sticks that **rotate** (the box height changes), not only slide.

To review the extra keyframes: `outputs/03_labeling/labels_keyframes/_review_EXTRA_rolling_p1.jpg` / `_p2.jpg` (purple border).
In `labels_pages/` and the clips they are marked **LLM EXTRA (rolling)**, regular keyframes **LLM keyframe** (red border), the rest **physics**.

Order after changing labels: `03c` → `03d` (pictures) → `03e` (inventory) → `04` → `05`/`06`/`07`.

## Where to SEE the labels (`src/03d_view_labels.py`)
| What | Where |
|---|---|
| The 67 keyframes with the LLM boxes, all keyframes of one video on one page | `outputs/03_labeling/labels_keyframes/_overview_<video>.jpg` |
| The 24 **extra** keyframes in rolling regions (purple border), for review as ground truth | `outputs/03_labeling/labels_keyframes/_review_EXTRA_rolling_p*.jpg` |
| Each keyframe full size (box coordinates printed next to each box) | `outputs/03_labeling/labels_keyframes/<video>__<frame>.jpg` |
| All 2,949 labeled frames as a clip per video ("KEYFRAME" vs "propagated" shown top-left) | `outputs/03_labeling/labels_videos/<video>.mp4` |
| Physics labels as pictures: every 4th labeled frame, 20 per page, LLM keyframes with a **red border** (46 pages) | `outputs/03_labeling/labels_pages/<video>_pXX.jpg` |
| Raw numbers: keyframe boxes | `data/labels/keyframe_boxes_llm.json` |
| Raw numbers: all propagated boxes | `data/labels/boxes.json` |

Box colours: wood = yellow, plastic = white, canga = orange, jaspilito = pink, tool = green.

## Checking the result
Open `outputs/03_labeling/propagation_*.jpg` and the `.mp4` clips:
- `canga2`: the box follows the rock well across the pass.
- `7_wood`: good in most frames. Frame ~2437 is wrong: the stick was still being placed by hand and moved faster than the belt.
  **Propagation assumes objects ride with the belt.** Anything that slides or rolls breaks that assumption.
- `labels_pages/iron_ore_canga2_p01.jpg`: the rock **rolls**. Between the keyframes (2629 → 2665) the box drifts
  ahead of the rock, then jumps when the next keyframe takes over (frame 2651, the rock has also moved down).
  After 2665 the rock lags behind its box. Objects that roll or slide do not move exactly with the belt, so more keyframes are needed there.
- Fix already applied: when a keyframe box touches the image border (object partly outside), propagated copies extend to that border.

## Doing it properly later (your own labels)
Recommended tool: **Roboflow** (web, free) or **CVAT** (web or local, free).

1. Create a project → *Object Detection*. Upload the images in `outputs/03_labeling/keyframes/` (they have a grid; for clean
   images, upload the originals from `data/raw/ironore/image-files/<video>/<id>-<video>.jpeg`).
2. Add classes: `wood, plastic, canga, jaspilito, tool`.
3. Draw one tight box per object. Include partly visible objects at the edges.
4. Export → **YOLO v5/v8 txt** format. Each line is `class x_center y_center width height`, normalised to 0–1.
5. Convert to our JSON format (`[x1,y1,x2,y2,class]` in pixels, key `"<video>/<frame_id>"`) and run
   `.venv\Scripts\python src\03c_propagate_labels.py your_file.json`. A converter script will be added when you get there.

Tips: about 1 minute per keyframe, so ~1 hour for all 67. Lower `KEY_SPACING_PX` in `03a` for more keyframes and
more accurate propagated labels.

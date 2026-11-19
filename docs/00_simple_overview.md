# The project in plain words

Read this first. The other docs (01, 02, 03, …) give the technical details of each step.

---

## 1. What is the dataset?
Researchers built a small conveyor belt in a lab and poured **iron ore** (dark grey gravel) onto it.
Sometimes they also threw in things that **should not be there**:

- wooden sticks
- orange rocks ("canga")
- big pink rocks ("jaspilito")
- white plastic pieces
- tools: hammer, pliers, wrenches, a wooden block

A camera looking straight down filmed the belt.

## 2. How big is it?

| | Count |
|---|---|
| Videos | **19** |
| Videos with **only ore** (normal) | 6 |
| Videos **with foreign objects** | 13 |
| Frames (still photos cut out of the videos) | **≈ 26,460** |
| Frames per second | ≈ 120, so neighbouring photos are almost identical |

A **frame** is simply one photo from the video.

## 3. What was missing?
The researchers only sorted photos into two piles:
- "this photo **has** a foreign object" (≈ 3,000 photos)
- "this photo is ore only"

They **never marked *where*** in the photo the object is. Our model must point at the object,
so we had to create that information ourselves.

## 4. How did we create the labels?

**Step A: measure how fast the belt moves** (`src/02_belt_motion.py`).
Comparing two consecutive photos tells us, e.g., "everything slid 18 pixels to the left".
We measured this for every pair of frames. The speed differs per video, and sometimes the belt even stops.

**Step B: label a few photos by hand** (`src/03a_…`, `src/03b_…`).
Out of the ≈ 3,000 "has an object" photos we picked only **67** ("keyframes") and drew a rectangle around each object.
For this first trial run, Claude did the drawing by looking at the pictures. These boxes are rough, and you will redo them yourself later.

**Step C: let the physics do the rest** (`src/03c_propagate_labels.py`).
If the rock is at some position in photo 2656, and the belt moves 19 px left per frame,
then in photo 2657 the rock is 19 px further left. So we copy the rectangle and slide it.

> **67 hand-made labels → 2,949 labeled photos with 3,323 rectangles.**

This works only for objects that ride along with the belt. If someone is still placing an object by hand, or it rolls, the box will be off.

## 5. What does a label look like?
A **box in the picture, written as 4 numbers plus a name**. Real entries from `data/labels/boxes.json`:

```
canga2, photo 2656 → [589, 247, 856, 507, "canga"]
canga2, photo 2657 → [570, 247, 837, 507, "canga"]
```

```
 (0,0) ───────────────────────────────► x  (1280 pixels wide)
   │
   │        (589,247) ┌──────────┐
   │                  │  orange  │
   │                  │   rock   │
   │                  └──────────┘ (856,507)
   ▼
   y (720 pixels tall)
```

- `589, 247` = top-left corner (x, y)
- `856, 507` = bottom-right corner
- `"canga"` = what the object is

From photo 2656 to 2657 the box moved **589 → 570 on x** (19 px left), and y stayed the same. That is the belt moving.
A photo with several objects has several boxes.
To see them drawn on real photos, open `outputs/03_labeling/propagation_*.jpg`.

## 6. From boxes to a grid (step 04)
The model does not read rectangles. We lay a grid over each photo, like graph paper with **40 × 22 squares**.
Every square that is mostly covered by a box gets **1** ("object here"), and every other square gets **0** ("ore / belt").
The model learns to guess 0 or 1 for each square. That grid is our coarse "segmentation mask".

## 7. One training example = 3 photos + 1 grid (step 04)
To beat glare, the model looks at **three photos of the same spot in time**: a bit before (*t−k*), now (*t*), a bit after (*t+k*).
- We know how far the belt moved between them (step A), so we know where the same object is in each photo.
- We paint **fake glare** (a bright white blob) onto one of the three photos, so the model must learn to use the clean ones.
- The answer the model must give is the 0/1 grid for the middle photo *t*.

We also add examples from the **ore-only** videos. Their grid is all zeros, which teaches the model what "nothing wrong" looks like.

**Result of step 04:** 3,694 examples in total.

| | examples | with an object | ore only |
|---|---|---|---|
| train (to learn from) | 2,699 | 963 | 1,736 |
| val (to tune settings) | 470 | 166 | 304 |
| test (final exam, never seen in training) | 525 | 340 | 185 |

Whole videos go into one group, so the model is tested on recordings it has never seen.
Pictures: `outputs/04_samples/` (start with `box_to_grid.jpg`, then `sample_01.jpg`).

## 8. Step 05: turn photos into descriptions (ResNet)
Raw pixels are a poor description: the same wood looks totally different in glare or shadow.
We run each photo **once** through **ResNet-18**, a ready-made image network trained by others on 1.2 million photos.
It outputs, for each of the 40 × 22 squares, a list of **256 numbers** describing what the patch looks like.
We don't change ResNet ("frozen"); we only borrow its knowledge ("transfer learning").
The results are saved to disk ("caching"), so training never has to run ResNet again.

## 9. Step 06: the model we train
For each square of the middle photo, the model:
1. squeezes the 256 numbers down to 96,
2. **slides** the earlier and later photo by the measured belt movement, so the object sits on the same square in all three,
3. **looks at that spot in all 3 photos** and decides which to trust (a glared photo should get less trust): this is the *attention*,
4. compares the square with the rest of the belt,
5. outputs "object" or "ore".

We train 4 versions to compare: **single** (only the middle photo), **avg** (3 photos simply averaged),
**noalign** (3 photos without sliding), and **beltpulse** (ours: slide + attention).

### How does the model know what an impurity is?
Not by itself. "Unusual" does not mean "impurity": big ore lumps, the belt edge, springs and glare are unusual but fine.
The model is **taught by example (supervised learning)**:
it guesses per square → we compare with the 0/1 answer grid → the error (loss) is used to adjust its weights → repeat thousands of times.
The **ore-only examples** (≈ 1,700 of 2,700 in training, all-zero answers) are what teach it that big rocks, belt edges and glare are *not* impurities:
every time it flags one, it is corrected.

**Could it learn from the normal videos alone?** Yes, that is *anomaly detection*, with no impurity labels:
- *Fill in the hidden square:* train the transformer, on ore-only data, to predict a hidden square from its neighbours and the other 2 photos. At test time an impurity is predicted badly, and a big error = anomaly.
- *Memory bank (PatchCore):* store descriptions of many normal squares and flag any test square that is far from all of them.

The catch: these flag anything *new*, not specifically impurities. A partial fix uses belt motion: objects move with the belt, but a lamp reflection stays fixed relative to the camera.
Plan: supervised = main method. Anomaly detection = optional extra comparison ("no labels vs. few labels").

## 10. Step 07: the final exam
On test videos never seen in training, we measure how much the predicted object squares overlap the true ones (IoU, F1),
especially for squares hidden under glare, and draw predicted vs. true squares on the photos.

## 11. Why not just use an LLM for detection?
The LLM labelled our keyframes well, so why build a model at all?

| | LLM | BeltPulse (measured on the RTX 3050 laptop, `src/09_speed.py`) |
|---|---|---|
| time per photo | several seconds | **≈ 7 ms** (ResNet 4.2 + transformer 3.0) |
| keeps up with 120 fps camera | no (~1,000× too slow) | **yes**, 140–158 fps |
| cost | paid per image (~10 million images/day per camera) | free after training |
| internet / cloud | needed | not needed, runs on a small local GPU |

Also: the LLM labelled only **67** photos, and only ones already known to contain an object. The other 2,882 labels came from belt physics.
On a live stream it would also have to decide *which* photos contain anything.

**Framing for the report, "teacher → student":** a big, slow model (LLM) labels a few photos offline →
belt physics spreads them to thousands → a small, fast model (BeltPulse) learns from them and runs live.
Contributions: real-time speed on cheap hardware, very few labels needed, and (to be shown) robustness to glare.
Final test numbers should be checked against labels you made or verified yourself, not only LLM labels.

## 12. Giving the transformer a real job (E10, E11, E12)
With only 3 perfectly lined-up photos, plain averaging does as well as the transformer. So we gave it harder jobs:
- **8 photos instead of 3** (some glared, some blurred, some with the object half out of view): it should learn to skip the bad ones.
- **A wider search window (7×7 squares):** a rolling rock isn't exactly where physics says, so the transformer *searches* nearby for it.
- **One decision per object:** follow each object through all its frames and read the whole sequence, then say once: ore, wood, plastic, canga, jaspilito or tool.

**Counting without double counting:** each object appears in dozens of frames. We link the detections frame to frame
using the belt motion ("the stick from the last frame should now be 18 px further left, and here it is"). Each linked chain is one object,
counted once. In the smoke test this turned 119 detections into 6 tracks for 3 sticks.

## 13. Where we are
The whole flow ran on a **100-sample smoke test** (≈ 3 minutes total). Every step works.
The numbers from it are **not meaningful** (too little data, rough labels). All models score IoU ≈ 0.5, and none is clearly better yet.
Next: the full run with all 3,694 samples, and later your own hand-made labels.

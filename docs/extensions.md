# Possible extensions

Ideas beyond the core pipeline (steps 01–08). New ideas get appended here as they come up.
Each entry: what, why, how, effort, status.

---

## E1. Pixel-precise segmentation (outlines instead of box-shaped squares)
**What:** predict the real outline of each object instead of 32×32 px squares filled from a box.

**Why:** today the answer grids come from boxes, so even a perfect model learns "box area", not "stick shape".
A long diagonal stick fills a big rectangle that is mostly ore.

**How:**
1. **Better labels with SAM (Segment Anything Model):** give SAM each keyframe box as a prompt, and it returns the object's pixel mask.
   Propagate masks with belt motion exactly like boxes (`03c`, shift the mask instead of the box).
2. **Decoder:** add a small upsampling head after the transformer (40×22 → 80×44 → 160×88 …).
   Feed it earlier, sharper ResNet features (`layer1`/`layer2`, also cached), U-Net style.
3. Evaluate with pixel IoU instead of square IoU.

**Effort:** ~1–2 days. **Status:** idea.

---

## E2. Anomaly detection from normal videos only (no impurity labels)
**What:** learn only from the ore-only videos, then flag anything that does not look like normal ore.

**Why:** in a real plant, impurity labels are rare and new impurity types appear. Compare "few labels (supervised)" vs "no labels".

**How (two variants):**
- **E2a. Memory bank / PatchCore (no training):** store the cached ResNet descriptions of many normal squares.
  For a test square, the distance to its nearest stored neighbour = anomaly score. Cheap, since features are already cached. Fits as a 5th comparison model in step 07.
- **E2b. "Fill in the hidden square" with the transformer:** train BeltPulse on normal 3-photo samples to *predict* a hidden square's
  description from its neighbours and the other 2 aligned photos. At test time a large prediction error = anomaly.

**Caveat:** these flag anything *new* (an unusually big rock, an unseen glare), not specifically impurities.
Include glare in the normal training data. Fine-tuning afterwards on a few labeled frames can combine both worlds.

**Effort:** E2a ~1 hour, E2b ~1 day. **Status:** idea.

---

## E3. Use motion to tell glare from objects
**What:** give the model an explicit cue that separates "moves with the belt" from "fixed relative to the camera".

**Why:** an object rides on the belt, so after kinematic alignment it stays on the same square in all 3 photos.
A lamp reflection is fixed in the camera image, so after alignment it *jumps* between squares.
That is a physical signature of glare that a single photo can never have.

**How:** feed the transformer, in addition, the *non-aligned* features (or the difference aligned − non-aligned),
or add a per-square "consistency across the 3 aligned photos" feature. Test on real glare (e.g. `iron_ore_5`, `jaspilito_4`), not only synthetic glare.

**Effort:** ~half a day. **Status:** idea.

---

## E5. Separate the objects (instance segmentation / counting)
**What:** output not just "impurity squares" but "object 1, object 2, …", so touching objects are not merged and can be counted or picked up individually by a sorter.

**Why:** today a canga rock lying on a plastic strip becomes one blob (`outputs/07_eval/smoke100/multi_object_01.jpg`).

**How (simple → advanced):**
1. Post-processing only: connected components on the predicted grid (already done in `07b`), plus removing tiny blobs. Free, but cannot split touching objects.
2. Add a second output per square: "distance to the object centre" or "centre heat-map" (CenterNet-style), then group squares around each centre.
3. Use the belt motion: squares that belong to one rigid object move together, which helps split objects over time.

**Effort:** 1 = done; 2 = ~1 day. **Status:** idea.

---

## E6. Say *what* the impurity is (multi-class)
**What:** predict wood / plastic / canga / jaspilito / tool per square instead of yes/no.

**Why:** the labels already contain the class; a sorter might handle wood and metal tools differently.

**How:** change the head from 1 output to 6 (ore + 5 classes), the loss from BCE to cross-entropy, and the grid from 0/1 to class numbers in `04_build_samples.py`.
Report a per-class IoU. Watch out: few examples per class (e.g. plastic), so classes may need weighting.

**Effort:** ~half a day. **Status:** idea.

---

## E7. LLM as teacher, measured properly (auto-labelling study)
**What:** treat the LLM as an automatic labeller and measure how good its labels are, and how good a model trained on them gets.

**Why:** the pipeline is a teacher → student setup (LLM labels offline, BeltPulse runs live). Two questions an examiner will ask:
"How accurate are the LLM labels?" and "Does a model trained on LLM labels do as well as one trained on human labels?"

**How:**
1. Label the same 67 keyframes yourself (Roboflow/CVAT).
2. Compare LLM boxes vs. your boxes: box IoU, missed objects, wrong classes.
3. Train BeltPulse twice (LLM labels vs. your labels) and evaluate both on test labels **you** verified.
4. Optional: let the LLM label more keyframes (cheap) and see if "more but noisier" beats "fewer but cleaner".

**Effort:** ~1 day (mostly your labelling). **Status:** idea.

---

## E8. Detect and correct rolling / sliding objects during label propagation
**What:** improve the physics-copied boxes for objects that do not move exactly with the belt (rolling rocks, sticks still being placed),
and automatically flag where that happens.

**Why:** propagation assumes the object sits still on the belt. In `outputs/03_labeling/labels_pages/iron_ore_canga2_p01.jpg`
the orange rock rolls: the box drifts ahead of it, jumps at the hand-over to the next keyframe (frame 2651), and then lags behind.
Wrong labels teach the model wrong answers.

**How (physics as the first guess, image as the correction):**
1. **Two-keyframe correction (no image analysis):** propagate keyframe A's box forward to keyframe B and compare with B's real box.
   The difference = how far the object moved relative to the belt. Spread it linearly over the frames in between, so jumps become smooth drift.
2. **Local template matching:** take the object patch from the previous frame and search only ±40 px around the physics-predicted
   position in the next frame. Snap the box to the best match. Run forward from A and backward from B, and trust frames where both agree.
3. **Optical flow inside the box:** compare the object's own motion with the belt's motion.
   Equal → riding on the belt. Different → **rolling/sliding detected**: correct the box, and/or report "add a keyframe here" (automatic label QC).

**Measure it:** on frames you label by hand, compare box IoU for plain physics vs. 1 vs. 2/3.
Report how many frames were flagged as rolling per object type (rocks will roll, sticks and blocks mostly won't).

**Effort:** 1 = ~1 hour, 2 = ~half a day, 3 = ~half a day.
**Status:** method 1 **done** (`03c`). Rolling flags + 24 extra LLM keyframes in flagged intervals **done** (`03f`);
flagged intervals went 14 → 7, max error 161 → 120 px. Methods 2 and 3 are still ideas.

---

## E9. Stronger simple baselines: does the transformer earn its place?
**What:** compare BeltPulse's learned attention against the simplest ways of using 3 photos.

**Why:** "am I over-engineering?" The transformer's only real addition over the `avg` model is *choosing, per square, which view to trust*.
In the smoke test `avg` (0.59) ≈ `beltpulse` (0.58), and the attention weights are near-uniform, so it currently behaves like averaging.
If simple fusion matches it on the full data, the honest conclusion is "alignment + multiple views is what matters; simple fusion suffices".
That is a valid, reportable result.

**How:**
1. **Vote / late fusion (no training):** run the trained `single` model on each of the 3 aligned photos, then combine the 3 probability grids with:
   - **OR / max** ("any frame says yes"): fewest misses, but false alarms add up: 1 − (1 − p)^N (p = 2% per frame → ~6% for 3 frames, ~18% for 10).
     Glare is a white blob and the dataset contains white plastic, so glare can cause false alarms too.
   - **k-of-N** (at least k frames say yes): adjustable middle ground.
   - **mean / majority:** fewest false alarms, but an object visible in only one frame is outvoted.
   Report a **precision–recall curve** for each: missing an impurity (contamination) usually costs more than a false reject (a bit of good ore blown out),
   so the operating point is a sorting decision, not a fixed rule.
2. **Rule-based glare masking (no training):** ignore squares whose photo is nearly white/saturated, and average the remaining views.
   Variant: **median of the aligned views** (per square, per feature). A glare that shows up in only 1 of 3 aligned views is removed by the median,
   with no training and no glare detector. If the transformer cannot beat median fusion on real glare, its glare handling is just a learned median.
   This beats learning on our synthetic white-blob glare if the transformer adds nothing beyond "skip white".
3. Evaluate all models on (a) synthetic glare, (b) **real** difficult frames: real glare / motion blur in `iron_ore_5`, `jaspilito_4`, `canga3`.
   The transformer is justified only where it beats 1 and 2.

**Effort:** ~2–3 hours. **Status:** **implemented** (2026-11-19): votes OR / 2-of-N / majority / mean in `07`, modes `median` and `glareskip`
in `src/model.py`, AP + PR curves + per-video table in `07`. Real difficult frames: `jaspilito_4` moved to test. See `docs/07_evaluation.md`.

---

## E10. Wider attention window: let the transformer find where the object really moved
**What:** enlarge the cross-frame attention window from 3×3 to e.g. 7×7 squares around the physics-predicted position.

**Why:** physics alignment is exact only for objects riding still on the belt; rolling/rotating objects are off by up to ~5 squares (E8).
Physics gives the rough spot, attention finds the precise one. Searching for the patch that matches is exactly what attention computes;
averaging/voting (E9) cannot do this. Bonus: the attention peak offset = the object's motion relative to the belt (learned rolling detection).

**Prediction to test:** on rolling intervals, `avg` averages the true patch in *t* with plain ore from the shifted views, so the object signal is diluted to ~1/3.
Expect `avg` ≤ `single` there, and `beltpulse` with a 7×7 window ≥ `single`. Non-transformer alternative: re-align with template matching or optical flow (E8 methods 2/3), then average.

**How:** `win` parameter in `src/model.py` (3 → 5/7). Tokens per square grow 27 → 147, so check GPU memory and use a smaller batch.
Evaluate separately on rolling intervals (`data/labels/rolling_flags.csv`) vs. still objects.

**Effort:** ~2–3 hours. **Status:** **implemented** as variant `T8w7` (with E11), see `docs/04b_multiview.md`. Smoke test only; full run pending.

---

## E11. Many frames instead of 3
**What:** feed 8–16 aligned views per decision instead of t−k, t, t+k.

**Why:** with 3 views, simple averaging is hard to beat. With many views, some are glared, blurred, half outside the image, or duplicates (belt stopped).
Averaging all of them mixes in junk; attention can learn to skip useless views. This is a stronger case for a transformer than 3 views.

**How:** step 04 picks T frames spread over the object's visible range (spacing by belt travel), `shift_px` for each; the model already accepts any T
(time embedding must be sized to T). Compare against `avg`/vote with the same T.

**Effort:** ~half a day (+ more cached features). **Status:** **implemented** as variant `T8w7` (8 views, with E10), see `docs/04b_multiview.md`. Smoke test only; full run pending.

---

## E12. One decision per object over its whole pass (sorter-level output)
**What:** follow each object through all frames it is visible in (using belt physics) and let a transformer read the variable-length sequence
of its descriptions to output one answer per object: impurity type or ore.

**Why:** a real sorter needs one reject/accept per object, not a mask per frame. Variable-length sequences of mixed quality are a classic
transformer job. It gives a per-object accuracy that is easy to explain.

**How:** per pass, pool the features inside the propagated box in every frame → sequence of tokens → small transformer encoder + [CLS] token → class.
Train on passes from train videos, test on passes from test videos (only ~30 passes in total, so augment by random sub-sequences).
Baselines over the same frames: **OR** (any frame says yes; with 300 frames even 1% per-frame false alarms flag almost every ore lump),
**k-of-N** (e.g. ≥ 20% of frames), **mean**. Compare on per-object precision/recall.

**Effort:** ~1–2 days. **Status:** **implemented** in `src/10_per_object.py`, including counting with a physics tracker (no double counting)
and object type, see `docs/10_per_object.md`. Smoke: with 50% glared frames, OR-voting gave 3/8 false alarms on ore and the sequence transformer 0/8.
**Next:** to claim "the transformer improves counting", run the counting demo with each track filter (none / FrameMLP+OR / k-of-N / mean / MeanPool / SeqTransformer)
on the same tracks and report the count error per video (|counted − true|), with and without lamp glare. The tracker does the counting; the filter only decides which tracks count.

---

## E13. DETR-style object queries (detect individual objects with a transformer decoder)
**What:** a transformer decoder with N learned "object slots". Each slot claims one object and outputs its box/mask and type (DETR, Mask2Former).

**Why:** the standard modern way to separate touching objects and count them (see E5, E6).

**Caveat:** slow to train and data-hungry, which is risky with ~1,000 object examples. Mention as future work rather than promise.

**Effort:** several days. **Status:** future work.

---

## E14. Pretrained vision transformer as the feature extractor (DINOv2 instead of ResNet-18)
**What:** replace the frozen ResNet-18 (step 05) with a frozen self-supervised vision transformer (e.g. DINOv2 ViT-S/14) and cache its patch features instead.

**Why:** the realistic place for a transformer in a small-data project is a big pretrained one, not a small one trained from 100–1,000 samples.
DINOv2 patch features are strong for "which patches belong together" and for anomaly scoring (E2a works well on them), with no fine-tuning.
It is also an honest answer to "where does the transformer help?": better per-patch descriptions help *every* model, single-frame included.

**How:** `05_features.py` → load DINOv2 from torch hub, resize frames to a multiple of 14, keep the patch tokens (384 numbers each),
resample to the 22×40 grid (or retrain with the native grid). Rerun 06/07 for `single`, `avg`, `beltpulse`, and compare against the ResNet runs.

**Effort:** ~half a day (+ re-caching features; ViT-S fits on the RTX 3050 at batch 1–4). **Status:** idea.

---

## E15. Rolling metric per object (how far it moves relative to the belt)
**What:** report, per tracked object, its motion relative to the belt (px per frame, max drift, % of frames rolling), next to its type.

**Why:** a sorter fires the air jet where physics says the object will be; a rolling object is not there, so the jet misses.
Also a type cue (round canga rolls, flat wood/plastic mostly slide or stay), and automatic label QC (E8).
And it checks whether the transformer really does the matching we claim (E10).

**How (three readouts, compare them):**
1. **Geometry (no learning, baseline):** in `10_per_object.py`, per track: (box-centre movement − belt movement) per frame.
2. **Attention readout (free by-product):** `model.last_token_attn` is (B, H, W, views, win²). Per object square and per view,
   the attention-weighted mean of the window offsets = where the model found the object, relative to the physics guess.
   Average over the object's squares, divide by the time gap. Sanity check: the offsets for *t−k* and *t+k* should point in opposite directions.
   Caveat: not trained for this; if attention is near-uniform (as in the smoke run), the readout is ≈ 0 and meaningless.
3. **Trained head (optional):** add a 2-number output (dx, dy) to `SeqTransformer`, trained on the residuals between keyframes (`03c`).

**Ground truth:** keyframe residuals in `data/labels/rolling_flags.csv` (residual_x/y over a known gap; only 7 intervals → plus manual checks).
**Metric:** error in px vs ground truth; rolling yes/no accuracy at a threshold (e.g. > 1 square of drift).

**Effort:** 1 ≈ 1 hour, 2 ≈ 2 hours, 3 ≈ half a day. **Status:** idea.

---

## E16. Does the spatial transformer (belt context) help? Ablation
**What:** train `avg` (or `beltpulse`) with the spatial transformer (Part B) replaced by (a) nothing (each square decided alone) and (b) a few 3x3 conv layers of similar size.

**Why:** all six models contain the same spatial transformer, so the main comparison only tests the *cross-frame* attention (Part A).
"The transformer is not needed" can therefore only be said about fusing the views. Whether a transformer helps by letting all 880 squares
look at each other ("is this patch unusual compared to the rest of the belt?", long sticks across many squares) is untested without this ablation.

**How:** a `spatial` option in `src/model.py`: `transformer` (now) / `none` / `conv`. Train on the 3-view variant only, ~17 min per model.

**Effort:** ~1 hour code + ~35 min training. **Status:** idea.

---

## E4. Visualise which patches the transformer listens to — **done**
Implemented as step 08 (`src/08_explain_attention.py`, `docs/08_attention_explained.md`).

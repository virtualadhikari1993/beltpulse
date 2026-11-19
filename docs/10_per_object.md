# 10 — One decision per object: counting and object type (extension E12)

**Script:** `src/10_per_object.py [--smoke]`  **Output:** `outputs/10_per_object/<smoke|full>/`
**Helper:** `src/backbone.py` (the frozen ResNet-18)

## Why
A sorter needs **one decision per object** ("eject this when it reaches the air jet"), not a mask in every frame.
And it needs to **count**: how many impurities, and of which type.

## Not counting the same object many times
An object is visible in 30–600 frames. Counting detections per frame would count one stick hundreds of times
(in the smoke run, `7_wood_3` had **119 raw detections** for **3 sticks**).

**Fix: link detections into tracks with belt physics, then count tracks.**
1. **Detect** per frame: the detector's object squares → blobs (touching squares) → boxes.
2. **Link** (`link()` in the script): for every open track, predict where its last box is now, moved by the measured belt motion.
   A new detection that overlaps the prediction (IoU > 0.1, or centre within 64 px) is the **same object** and joins the track.
   An unmatched detection starts a new track. A track not seen for a few frames is closed.
3. **Count** each track **once**, and only if it has ≥ 3 detections (drops 1–2-frame flickers).
4. **Decide** per track with a transformer: ore (false alarm, not counted) or wood / plastic / canga / jaspilito / tool.

A fixed lamp reflection does not move with the belt, so it usually cannot form a long track.
When the belt is slow or stopped it can, and then the transformer has to call it "ore".

## The per-object transformer (legit job: variable-length sequence of mixed quality)
- Each frame of a track → ResNet features **pooled inside the object's box** (mean + max = 512 numbers) = one token.
- `SeqTransformer`: `[CLS]` + up to 64 frame tokens → 2 transformer layers → one class from `[CLS]`.
  Trained on random sub-sequences, with glare on a random 0–50% of the frames.
- Baselines on the **same tokens**:
  - `MeanPool`: average all tokens, then MLP (the averaging approach)
  - `FrameMLP` + voting: classify each frame alone, then **OR** (any frame says impurity), **k-of-N** (≥ 30% of frames), **mean**

Training objects: the labeled objects (boxes from step 03, linked into tracks with the same physics tracker) plus **ore tracks**:
random ore regions followed with belt physics in the ore-only videos (8 per video).
Smoke run: 38 object tracks + 48 ore tracks (train: 18 objects + 32 ore, test: 14 objects + 8 ore).

## Smoke result (every 8th frame, 5 epochs, ~3–4 min): small test set, treat as a demo

**Per object (14 test objects + 8 ore regions, perfect tracks from the labels):**

| method | glare per frame | impurities found | false alarms on ore | correct type |
|---|---|---|---|---|
| SeqTransformer | 0% | 14/14 | 0/8 | 0.93 |
| MeanPool | 0% | 14/14 | 0/8 | 0.93 |
| FrameMLP + OR / k-of-N / mean | 0% | 14/14 | 0/8 | 1.00 |
| SeqTransformer | 50% | 14/14 | 0/8 | 0.93 |
| MeanPool | 50% | 13/14 | 0/8 | 0.93 |
| **FrameMLP + OR** | 50% | 14/14 | **3/8** | 0.93 |
| FrameMLP + k-of-N / mean | 50% | 14/14 | 0/8 | 0.93 |

This shows the predicted OR-voting weakness: with glare on half the frames, "any frame says impurity" flags 3 of 8 plain ore regions.
The sequence transformer has none. With only 22 test tracks this is a demonstration, not proof.

**Counting on the test videos** (detector = smoke `single` model, every 6th frame):

| video | true objects | raw detections | tracks (≥ 3) | counted impurities | by type |
|---|---|---|---|---|---|
| 7_wood_3 | 3 wood | 119 | 6 | 4 | wood 3, plastic 1 |
| canga3 | 4 canga + 2 plastic | 61 | 6 | 5 | canga 3, plastic 1, tool 1 |
| jaspilito_3 | 2 jaspilito | 60 | 6 | 2 | jaspilito 1, canga 1 |
| things_3 | 1 wood + 2 tool | 51 | 3 | 3 | wood 1, tool 1, plastic 1 |
| iron_ore_4 (ore only) | 0 | 11 | 1 | 1 | canga 1 (false) |

With the fixed lamp glare added: `iron_ore_4` goes to 0 counted (the lamp track is called ore), but `jaspilito_3` rises to 4
(slow belt, so lamp blobs link into tracks). Physics tracking turns **hundreds of detections into a handful of objects**.
The remaining errors come from the weak smoke detector (100 training samples) and **touching objects merging**:
in `counting_iron_ore_canga3.jpg` a canga rock lying on a plastic strip becomes one track, called "tool".

## Pictures
- `counting_<video>.jpg` / `_lamp.jpg`: tracks active in one frame; green = counted impurity (with type), red = called ore.
- `track_attention.jpg`: one test stick over 10 frames, 6 of them glared. Bars = attention of `[CLS]` on each frame.
  The two fully glared frames (2388, 2396) get the lowest weight (6%); clear frames get 8–14%.

## Caveats
- Per-object test uses the labeled tracks (perfect tracking). The counting demo uses the real detector + tracker.
- Only ~30 passes exist in the whole dataset, so per-object numbers rest on very few objects.
- Touching objects merge into one blob → one track (see E5 for separating them).

# 01 — Data exploration

**Script:** `src/01_explore.py`  **Outputs:** `outputs/01_explore/`

## What we wanted to know
1. What does each recording show?
2. Are the frames consecutive (needed for a temporal model)?
3. Which labels does the dataset actually provide?

## What is in the download

| Folder | Content | Labels |
|---|---|---|
| `original-raw-videos/` | 19 MP4s, 1280×720, ~119 fps | none |
| `image-files/<video>/` | ~26k extracted frames, named `<frame_id>-<video>.jpeg` | none |
| `image-files-manual-split/` | frames sorted into `normal/` and `annomaly/` | frame-level |
| `split-ds-normal-filtered/` | `train/normal-op` (12,951), `test/normal-op` (2,991), `anomalous/anomaly` (2,991) | frame-level |
| `yolo-dataset-center/` | 2,474 frames + YOLO txt | **1 class: `conveyor`** (the belt region, *not* impurities) |
| `yolo-v8-weights/best.pt` | YOLOv8 that finds the belt | — |
| `scripts/` | the authors' frame extraction / cropping scripts | — |

## Findings
- **Frames in `image-files` are consecutive** (frame ids increase by 1, ~120 fps). So we can take frames *t−k, t, t+k* of the same object directly.
- **Videos fall into two groups:**
  - Pure iron ore (normal): `iron_ore`, `iron_ore_2` … `iron_ore_6`.
  - With foreign objects: `7_wood*` (wooden sticks), `canga*` (orange rock + white plastic pieces), `jaspilito*` (large pink rock), `things*` (wooden block, hammer, pliers, wrenches).
- **No impurity locations are labeled.** The YOLO boxes mark the belt. We only know *which frames* contain an impurity (the `anomalous` folder), not *where* — this is why we need step 03.
- **Real visual difficulties are present:** glare on the bare belt, heavy motion blur in fast videos (`iron_ore_5`, `canga3`, `jaspilito_4`), objects half-buried in ore.

## Pictures to look at
- `overview.jpg` — one frame per video.
- `sequence_iron_ore_7_wood.jpg` — 6 consecutive frames; the ore slides slightly left each frame.
- `anomaly_samples.jpg` — what the authors counted as anomalous.

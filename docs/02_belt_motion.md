# 02 — Belt motion (the "kinematic" part)

**Script:** `src/02_belt_motion.py`  **Outputs:** `data/meta/motion/<video>.csv`, `outputs/02_belt_motion/`

## Idea
Everything on a conveyor moves together. If the belt moved `dx` pixels between frame *t* and *t+1*,
then an object at position *x* in frame *t* is at *x + dx* in frame *t+1*. No object tracker is needed —
this is the "kinematic alignment" of the proposal.

## How we measure dx: phase correlation
`cv2.phaseCorrelate(a, b)` compares two images in the frequency domain and returns the single
(dx, dy) shift that best lines them up, plus a confidence. Because most of the picture is the moving
belt/ore, that shift *is* the belt displacement.

Details:
- Only image rows 60–640 are used (`BELT_Y0/BELT_Y1` in `common.py`). The steel frame above and below the belt is static and would pull the estimate toward 0.
- Images are halved in size for speed; dx is multiplied back by 2.
- `common.load_motion()` applies a 5-frame running median, removing single-frame glitches (the spikes in the plots) but keeping real speed changes.

## Results (`outputs/02_belt_motion/summary.txt`)
- The belt moves **horizontally, right → left (dx < 0)**; dy ≈ 0. The proposal assumed motion along Y — it is X here.
- Speed differs per video, from about −9 to −48 px/frame. So **dx cannot be one hard-coded constant**; we use the measured value per frame.
- `dx_boardcamera_iron_ore_jaspilito.png`: the belt **stops and restarts** several times. This is the real-world "belt slip/stutter" from the project form, and the reason we use per-frame dx.
- `iron_ore` and `iron_ore_6` show a median dx ≈ 0: in those recordings the belt was mostly standing still.

## How this is used later
- Step 03: move hand-drawn boxes from a keyframe to neighbouring frames.
- Step 04/06: shift frames *t−k* and *t+k* so the same object lands on the same feature-grid cell as in frame *t*.

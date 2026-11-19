# Proposal review: what holds, what is overclaimed, how to phrase the report

Reviewed: the submitted course form ("BeltPulse-Net v1", official) and the later PDF ("BeltPulse-Net v2").
Question: *if the transformer turns out unnecessary, is the proposal bogus?* **No.** It is a hypothesis.
A negative result for one component is a finding, as long as the report does not repeat the overclaims.

## The three real claims

| # | Claim | How we test it | Status |
|---|---|---|---|
| 1 | Several views of the same object beat one view when a view is corrupted (glare) | `single` vs `avg` / `beltpulse` | smoke hint: under glare `avg` found 77% of hidden object squares vs 65% for `single` (weak, 100 samples) |
| 2 | Belt kinematics align the views without an object tracker | `noalign` vs `avg`; alignment unit test | mechanism works (step 02, alignment test); accuracy effect to be confirmed in the full run |
| 3 | Learned cross-frame attention is the right way to fuse the views | `beltpulse` vs `avg`, + E9 baselines (voting, saturation masking) | open; smoke: attention ≈ averaging (near-uniform weights) |

Claims 1–2 are the heart of the proposal; the transformer is claim 3.
If 1–2 hold and 3 does not, the central idea is confirmed and the transformer is an honest negative finding.

## Statements to correct in the final report

**Form (official)**
- *Dataset adequate as-is.* It has no impurity location labels (frame-level yes/no only). Labels were built: LLM keyframes + belt-physics propagation (step 03).
- *Constant belt speed, attention "wiggles" for slips.* Speed varies per video and the belt even stops, so motion is measured per frame (step 02). The 3×3 attention window is the "wiggle".
- *CNN scans the first frame and triggers further processing only when it finds something unusual.* Not implemented: the CNN runs on every frame (~4 ms).
- *Low-compute edge hardware.* Measured ~7 ms/frame on a laptop RTX 3050 (step 09). Not measured on an actual edge device.

**PDF v2**
- "Completely mitigate", "pixel-perfect", "glare-free", "near-zero error": overclaims. Output is a 40×22 grid (32-px squares) learned from box labels.
- Motion along Y with a hard-coded ΔY: the belt moves along **X**, and speed is not constant.
- Frozen pretrained autoencoder: cannot freeze an untrained autoencoder. Implemented as a learned 1×1 bottleneck.
- Prior work = SORT/DeepSORT: a strawman. Closest prior work is **video feature aggregation**, e.g. *Flow-Guided Feature Aggregation for Video Object Detection* (Zhu et al., ICCV 2017):
  align neighbouring-frame features with optical flow, then fuse with adaptive weights. Our design is this, with optical flow replaced by belt kinematics.
- TACO/ZeroWaste synthetic dataset: dropped. The real iron-ore dataset (form) is used.

## What is actually specific to this project (novelty, stated modestly)
1. Using the known rigid belt motion (measured by phase correlation) instead of per-pixel optical flow to align views. It is cheaper and has no flow errors on textureless or glared regions.
2. Label propagation through belt physics, with two-keyframe correction and rolling detection: 91 LLM keyframes → 2,949 labeled frames.
3. An honest analysis of where multi-view fusion helps (glare, real blur) and where it does not (rolling objects, simple fusion vs attention).

## Suggested framing for the report
> **Hypothesis:** on a conveyor, belt motion lets us align several views of the same object for free. Fusing these aligned views makes impurity
> segmentation more robust to glare than single-frame models.
> **We test:** (1) multi-view vs single-view, (2) kinematic alignment vs none, (3) learned attention vs simple averaging / voting / saturation masking.
> **Result:** whatever the full run shows, including "attention ≈ averaging" if that is the outcome.

The proposal would only be in real trouble if claim 1 also fails (multi-view no better than single-view).
Even then, report the likely reasons (synthetic glare too easy, label noise) and test on real glare/blur frames.

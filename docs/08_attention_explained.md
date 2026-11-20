# 08 — Looking inside the transformer (one square at a time)

**Script:** `src/08_explain_attention.py [--limit N]`  **Output:** `outputs/08_attention/<tag>/square_XX_<kind>.jpg`

## What the picture shows
- **Top:** photo *t*. The **red** box is the square being explained; the **yellow** box is its 3×3 surroundings.
  The title gives the truth (impurity / ore) and the model's answer (% impurity).
- **Bottom:** the 27 patches the cross-frame attention compared, zoomed in:
  the 3×3 surroundings in photo *t−k*, in photo *t*, and in photo *t+k*.
  In *t−k* and *t+k* the window is taken at the **aligned** position (shifted by the belt motion), so it should show the same piece of the belt.
- **Numbers:** the attention weight of each patch (all 27 add up to 100%). Thicker, yellower border = more weight.
  The panel header gives the total per photo.

Three kinds of squares are shown:
| file | square | what we hope to see |
|---|---|---|
| `*_A_glare` | object, photo *t* covered by fake glare | low % on photo *t*, high on *t−k* / *t+k* |
| `*_B_clean` | object, no glare | weight spread over all 3 (all views are useful) |
| `*_C_ore` | ore, no object | anything; the head should answer "ore" |

## Smoke-test result (100 training samples)
All weights are about 3–5% per patch (≈ 33% per photo), close to a **plain average** (1/27 = 3.7%).
Example `square_00_A_glare`: photo *t* is pure white yet still gets 31%. The model still answers 61% impurity (correct),
because the two clean photos make up 69% of the blend, not because it chose to ignore the glare.

So the smoke-test model has **not learned to be selective yet**; it behaves almost like the `avg` baseline.
Whether it learns to down-weight glared patches with the full data is the key question for the full run.
If it does not, ideas to push it: train longer, more glare in training, or add the glare/motion cue from `docs/extensions.md`.

## Why not just a single-frame detector on every frame, then vote?
A single-frame detector *will* find the impurity in the one clean frame. Finding it is not the problem.
The problem is **combining** the per-frame answers, because the vote only sees yes/no and not *why* each frame said it.

Two cases with identical votes (8 frames, 1 = "impurity"):

| case | truth | what happened | votes |
|---|---|---|---|
| A | impurity | 7 frames glared hide it, 1 clean frame shows it | 0 0 0 0 0 0 0 1 |
| B | plain ore | 7 clean frames, 1 glare blob looks like white plastic | 0 0 0 0 0 0 0 1 |

Any rule on the votes (OR, majority, k-of-N, mean) gives the same answer for A and B, so it gets one of them wrong:
OR is right on A and gives a false alarm on B; majority is right on B and misses A.

The transformer gets the frame *features*, not the votes. It can see that in A the "yes" frame is the clean one,
and in B the "yes" frame is the glared one where the bright patch also jumps between aligned squares. So it can weight
each frame by how trustworthy it is at that square. That per-square, per-frame trust is the transformer's whole job here.

**Honest caveat:** a hand rule ("skip frames that are saturated white here, vote on the rest") also solves A and B.
But it also throws away real white plastic. The transformer is worth it when "is this frame trustworthy here?" is hard to write
as a rule: partial or dim glare, motion blur, an object half outside the frame, misalignment from rolling, white plastic vs glare.
Whether it actually learns this is what the E9 baselines test.

### Then why doesn't BeltPulse win in the smoke test?
The A/B example shows the transformer *can* beat **voting on decisions**. It does not show that it *does*, and our table compares something else:
1. **No voting baseline has been run yet.** `avg` averages the aligned *features* (fixed 1/3 weight each) and then decides. It does not vote,
   so it does not fall into the A/B trap the way OR or majority do. Both multi-frame models already beat `single`.
2. **3 frames, glare on only one.** Averaging 2 clean frames with 1 glared frame already dilutes the glare to 1/3.
   Adaptive weighting only pays off when fixed averaging breaks down, e.g. 7 of 8 frames glared.
3. **It has to learn the trust from data.** With 100 training samples the attention stayed near-uniform (≈ averaging),
   and run-to-run noise (±0.05) is as large as the gap between models.
4. **Hard cases are rare.** Most test squares are easy, so a gain on a few A/B-like squares barely moves the average IoU.
   Report the hard cases separately (real glare, many glared views, rolling intervals).

### What is learned vs what is computed
The saved model holds three learned 96×96 matrices (W_q, W_k, W_v) plus view and window-position embeddings: a *rule* for judging patches.
The trust weights (27 per square, summing to 1) are computed again for every square of every photo at prediction time and are not saved.
Drawn step by step in `docs/pipeline_architecture.html`, section "How the trust is computed".

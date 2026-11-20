# From photo to answer: every size, with one real example

Example sample (train, video `7_wood`): target frame t = 2437, k = 12 → frames 2425, 2437, 2449.
Belt shift `[105.2, 0, -97.4]` px. Glare painted on view t. One box: wood stick at x 784–1280, y 153–433.

| stage | size | meaning |
|---|---|---|
| original frame | 1280 × 720 px, 3 colours | one camera photo |
| model photo | 640 × 352 px | resized by half (step 04/05) |
| grid | 40 × 22 = 880 squares | each square = 16 × 16 model px = 32 × ~33 original px |
| answer grid (label) | 22 × 40 of 0/1 | 1 = at least 25% of the square is inside a box; this example: rows 4–12, columns 24–39 |
| one sample | 3 photos + 3 shifts + 1 answer grid | views t−k, t, t+k; k = smallest gap where the belt moved ≥ 96 px (3 squares) |
| ResNet output per photo | 256 × 22 × 40 | 256 numbers describing each square (they also "see" pixels around the square) |
| model input (batch of 8) | 8 × 3 × 256 × 22 × 40 | 8 samples, 3 views each |
| after bottleneck | 8 × 3 × 96 × 22 × 40 | 96 numbers per square |
| after alignment | same | each view slid by its shift (105.2 px = 3.3 squares) so the stick is on the same squares in all 3 |
| cross-frame attention | per square: 1 query vs 27 tokens | 3 × 3 window around the square × 3 views; output 96 numbers per square |
| spatial transformer | 880 tokens of 96 | all squares of view t attend to each other |
| head | 22 × 40 scores | sigmoid → probability; > 0.5 → "impurity here" |

Counts (3-view variant): 2,716 train / 470 val / 508 test samples. Train: 980 with an object, 1,736 ore only, 2,016 with glare.
Distinct photo files (glared versions count separately): 8,068.

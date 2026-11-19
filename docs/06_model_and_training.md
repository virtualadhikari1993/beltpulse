# 06 — The model and its training

**Scripts:** `src/model.py` (the network), `src/dataset.py` (loads cached features), `src/06_train.py [--limit N]`
**Output:** `data/models/<tag>/<mode>.pt`, `outputs/06_train/<tag>/curves.png`

## What goes in and what comes out
- **In:** the ResNet features of 3 photos (*t−k*, *t*, *t+k*), each 256 × 22 × 40, plus how far the belt moved between them.
- **Out:** one score per square of photo *t* (22 × 40). Score → probability "foreign object here"; above 0.5 counts as 1.

## Inside BeltPulse-Net (`src/model.py`)
| # | Part | Plain words | Learned? |
|---|---|---|---|
| 1 | Bottleneck (1×1 convs, 256 → 96) | squeeze each square's description to the essentials | yes |
| 2 | Kinematic alignment | slide photos *t−k* and *t+k* by the measured belt shift so the same object sits on the same square in all 3 grids | **no**, plain physics |
| 3 | Cross-frame attention | every square of *t* looks at a 3×3 neighbourhood in all 3 aligned photos (27 small patches) and decides how much to trust each. Glared patch → should get less trust. The 3×3 window tolerates small alignment errors from belt jitter. | yes |
| 4 | Spatial transformer | all 880 squares of *t* look at each other: "is this patch unusual compared to the rest of the belt?" | yes |
| 5 | Head | one score per square | yes |

### Where exactly is the transformer?
Two places in `forward()` of `src/model.py`:

**Attention in plain words:** each item asks a *question* (query). Every other item offers a *label* (key) and *content* (value).
The model scores how well each label matches the question, turns the scores into weights that add up to 1, and blends the contents.
A transformer = attention + a small feed-forward network.

**Part A, cross-frame attention** (`self.temporal`, an `nn.MultiheadAttention`, called at line ~97).
For one square of photo *t*, `F.unfold` (line ~87) gathers its 3×3 neighbourhood in each of the 3 aligned photos = 27 tokens.
The square is the question, and attention weights the 27 tokens, e.g. `t−k: 0.40, t (glared): 0.10, t+k: 0.50`.
If *t* is washed out by glare, the model can learn to rely on the other two. These weights are what `outputs/07_eval/<tag>/attention.png` plots.

**Part B, spatial transformer** (`self.spatial`, an `nn.TransformerEncoderLayer`, line ~101).
All 880 squares of photo *t* attend to each other: "do I look like the rest of the belt, or am I unusual?"
That is the "global belt context" from the proposal.

| model | Part A sees | Part B |
|---|---|---|
| single | 9 patches of photo *t* only | yes |
| avg | 9 patches of the averaged photo | yes |
| noalign | 27 patches, photos not aligned (wrong spots) | yes |
| beltpulse | 27 patches, aligned | yes |
| median, glareskip | 9 patches of the fused photo (median / non-washed-out average) | yes |

All of them use a transformer. They differ in *what Part A gets to look at*, which is how the comparison isolates the benefit of multiple aligned photos.

About 270,000 learned numbers in total. Tiny compared to ResNet's 11 million, which is why training is fast.

## The six models we train (to prove what helps)
| mode | sees | purpose |
|---|---|---|
| `single` | only photo *t* | the classic one-frame approach the proposal argues against |
| `avg` | 3 aligned photos, simply averaged | "is attention better than plain averaging?" |
| `noalign` | 3 photos **without** sliding them | "does the kinematic alignment matter?" |
| `median` | 3 aligned photos, median per square (E9) | "does a fixed robust rule remove glare just as well?" |
| `glareskip` | 3 aligned photos, washed-out views skipped by a brightness rule (E9) | "is attention more than 'skip the white view'?" |
| `beltpulse` | 3 aligned photos + attention | our method |

Voting baselines (`single` run on every view, then OR / majority / mean) need no extra training; step 07 builds them. See `docs/07_evaluation.md`.

All six share the same code; only step 2/3 changes.

## How training works
- One **epoch** = the model sees every training example once.
- For each batch of 8 examples: the model guesses → we measure how wrong it is (**loss**) → the optimizer (AdamW) nudges the weights to be less wrong.
- Loss = **BCE** (right/wrong per square; object squares count 5× because they are rare) + **Dice** (overlap of predicted vs true object area).
- After each epoch we test on the **validation** videos and keep the best version (by object IoU).
- Smoke test: 25 epochs, ~30–40 s per model on the RTX 3050.

## Reading `curves.png`
- Left: training loss (solid) keeps falling. Validation loss (dashed) flattens after ~8 epochs.
  That gap is **overfitting**: with only 100 examples the model starts memorising them. More data (the full run) is the cure.
- Right: validation object IoU per epoch. Jumpy, because only 25 validation samples.

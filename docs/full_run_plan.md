# Full-run plan: what gets trained, in what order, and what it is compared against

Written before the full training run (2026-11-19). Checks that every project claim has a model and a baseline.

## Flow

```
labels (03) ─► samples (04 base: 3 views │ 04b T8w7: 8 views)  ─► ResNet features cached (05)
                                                                   │
            ┌──────────────────────────────────────────────────────┴───────────────┐
            ▼                                                                      ▼
  06 train, per-frame detectors (base AND T8w7):                 10 per-object (needs a trained detector):
     single │ avg │ noalign │ beltpulse                             tracker (physics) ─► tracks
            │                                                       SeqTransformer │ MeanPool │ FrameMLP+votes
            ▼                                                                      │
  07 evaluate (test videos) ─► 08 attention pictures              per-object accuracy, counts per video
```

## What gets trained

| # | Model | Variant | Trained on | Script | Est. time |
|---|---|---|---|---|---|
| 1–6 | `single`, `avg`, `median`, `glareskip`, `noalign`, `beltpulse` | base (3 views, 3×3 window) | cached ResNet features of 2,716 train samples | `06_train.py` | ~10 min each |
| 7–12 | same 6 | T8w7 (8 views, 7×7 window) | 2,644 train samples | `06_train.py --variant T8w7` | ~1.5 h beltpulse, ~2 h the rest |
| – | votes: `single` + OR / 2-of-N / majority / mean | both | nothing (built in step 07) | `07_evaluate.py` | minutes |
| 9 | `SeqTransformer` (per object) | – | object tracks + ore tracks | `10_per_object.py --detector full` | – |
| 10–11 | `MeanPool`, `FrameMLP` (per-object baselines) | – | same tracks | `10_per_object.py` | – |

ResNet-18 is frozen (not trained). Everything else is small (~270k parameters for the per-frame models).

## Which claim each comparison tests

| Claim | Comparison | Status |
|---|---|---|
| 1. Several views beat one view under glare | `single` vs `avg` / `beltpulse` | ready |
| 2. Belt kinematics align the views | `noalign` vs `avg` | ready |
| 3. Learned attention beats simple fusion | `beltpulse` vs `avg`, `median`, `glareskip`, votes (E9) | ready |
| 3b. Attention wins where averaging breaks | T8w7 `beltpulse` vs T8w7 `avg`, many glared views | ready |
| 4. Attention finds rolling objects | T8w7 on rolling intervals, `avg ≤ single ≤ beltpulse` (E10) | **subset metric not implemented** |
| 5. One decision per object, no double counting | SeqTransformer vs MeanPool vs FrameMLP + OR / k-of-N / mean | ready (pass `--detector full`) |
| 6. Transformer improves counting | count error per video for each track filter (E12 "Next") | not implemented |
| 7. Rolling metric per object | geometry vs attention readout (E15) | not implemented, optional |

## Gaps found before the run

1. **Labels are still the LLM dry-run labels.** **Decided (2026-11-19):** reviewed by hand and kept as the final labels. Report them as LLM labels, physics-propagated and reviewed by eye.
2. ~~**The real-glare / rolling clips are in train, not test.**~~ **Fixed:** `jaspilito_4` swapped into test (with `jaspilito_3` into train); now 4 of 7 rolling intervals are in test. `iron_ore_5` is still train.
   Old note: **The real-glare / rolling clips are in train, not test.** `iron_ore_5` and `jaspilito_4` are train videos; of the 7 rolling intervals only 2 (`7_wood_3`) are in test.
   "Works on real glare" cannot be claimed from training videos. Fix before step 04 (swap a split) or state the limitation.
3. **One random seed.** Smoke runs showed ±0.05 run-to-run noise, as big as the gaps between models. Run base models with 3 seeds and report mean ± spread.
4. ~~**E9 baselines missing**~~ **Done** (2026-11-19). Old note: **E9 baselines missing** (per-frame + voting, median fusion, glare-skip rule). Claim 3 needs them. Voting needs no training (reuses `single`); median is one extra fusion mode.
5. **`10_per_object.py` defaults to `--detector smoke100`.** Pass `--detector full`.

## Commands for the full run (in this order)
```powershell
# 3 views (~1 h for 6 models)
.venv\Scripts\python src\05_extract_features.py
.venv\Scripts\python src\06_train.py
.venv\Scripts\python src\07_evaluate.py
.venv\Scripts\python src\07b_multi_object.py
.venv\Scripts\python src\08_explain_attention.py
# 8 views + 7x7 window (~4 h)
.venv\Scripts\python src\05_extract_features.py --variant T8w7
.venv\Scripts\python src\06_train.py           --variant T8w7
.venv\Scripts\python src\07_evaluate.py        --variant T8w7
.venv\Scripts\python src\08_explain_attention.py --variant T8w7
# per object (needs the full 3-view `single` model)
.venv\Scripts\python src\10_per_object.py --detector full
```

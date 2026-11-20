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

## Running it unattended (laptop locked) and watching progress
`scripts/run_full.py` runs the steps in order, writes everything to `outputs/logs/full_<variant>.log`,
keeps the current step in `outputs/logs/full_<variant>_status.txt`, and asks Windows not to sleep while it runs.

1. Plug in the charger. Leave the lid **open**, or set "When I close the lid" (plugged in) to *Do nothing*:
   `powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0; powercfg /setactive SCHEME_CURRENT`
   (undo afterwards with `LIDACTION 1`).
2. Start it in a normal PowerShell window (not the VS Code terminal, which dies with VS Code):
   `cd D:\foml_project; .venv\Scripts\python scripts\run_full.py`
3. Lock the laptop (Win + L). Locking does not stop the run.
4. Watch it from a second PowerShell window: `Get-Content D:\foml_project\outputs\logs\full_base.log -Wait -Tail 20`
   (one line per epoch with time left), `nvidia-smi -l 10` (GPU busy?), or `type D:\foml_project\outputs\logs\full_base_status.txt`.
5. If a step fails, the status file names it: rerun with `--from <step>`.

## Reading the log
**Step 05:** `8068 photos needed, 0 not cached yet` = every photo's ResNet features already exist, so `ResNet: 0it` means "nothing to compute", not an error.

**Step 06:** one line per epoch:
`[single 1/6] epoch 6/30  train loss 0.168  val loss 0.345  val IoU 0.713 (best 0.713)  3.3 min, ~13 min left for this model`
- `single 1/6`: which of the 6 models is training.
- `train loss`: error on the training examples. Should fall almost every epoch.
- `val loss` / `val IoU`: on the validation videos, which the model never trains on. These measure generalisation.
- `best`: the highest val IoU so far. Only that epoch's weights are saved, so later worse epochs do no harm.
- Train loss falling while val loss rises = overfitting (memorising the training videos). Expected here; the best-epoch saving handles it.
- Bad signs: `nan` loss, val IoU stuck near 0, or train loss not falling.

Measured full-run speed (3 views): ~0.55 min per epoch, ~17 min per model, **~1 h 45 min for all 6** (longer than the first estimate).

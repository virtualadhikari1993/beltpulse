# BeltPulse-Net — Term Project Workspace

Detecting foreign objects (wood, plastic, rocks like canga/jaspilito, tools) on an iron-ore conveyor belt.
The model combines a frozen CNN, belt-motion (kinematic) alignment across frames, and a temporal transformer.

Dataset: *In-lab Image Dataset of Foreign Objects and Anomalies in Iron Ore Conveyor Belts* — Mendeley Data
https://data.mendeley.com/datasets/s25x2bnshz/1 (CC BY 4.0), extracted into `data/raw/ironore/`.

## Pipeline at a glance

```
raw frames ─► 01 explore ─► 02 belt motion (dx per frame) ─► 03 labels (keyframes + propagation)
                                                                     │
          07 evaluate ◄─ 06 train transformer ◄─ 05 cache CNN features ◄─ 04 build 3-frame samples (+ glare)
```

**New here? Start with [docs/00_simple_overview.md](docs/00_simple_overview.md)**, the whole project in plain words.
**Numbers per video** (raw frames → LLM labels → physics labels → training examples): [docs/data_inventory.md](docs/data_inventory.md).

## Activities (each has a script in `src/` and notes in `docs/`)

| # | Activity | Script(s) | Notes | Status |
|---|----------|-----------|-------|--------|
| 01 | Look at the data | `src/01_explore.py` | [docs/01_data_exploration.md](docs/01_data_exploration.md) | done |
| 02 | Measure belt motion | `src/02_belt_motion.py` | [docs/02_belt_motion.md](docs/02_belt_motion.md) | done |
| 03 | Labels (dry run: LLM keyframes) | `src/03a_select_keyframes.py`, `src/03b_llm_label_sheets.py`, `src/03c_propagate_labels.py` (+ two-keyframe correction), `src/03d_view_labels.py` (view them), `src/03e_data_inventory.py` (counts), `src/03f_extra_keyframes.py` (extra keyframes where objects roll) | [docs/03_labeling.md](docs/03_labeling.md) | done (dry run) |
| 04 | Build 3-frame training samples + synthetic glare | `src/04_build_samples.py` | [docs/04_build_samples.md](docs/04_build_samples.md) | done |
| 05 | Extract & cache CNN features | `src/05_extract_features.py` | [docs/05_features.md](docs/05_features.md) | smoke test done |
| 06 | Train temporal transformer (+ baselines) | `src/model.py`, `src/dataset.py`, `src/06_train.py` | [docs/06_model_and_training.md](docs/06_model_and_training.md) | smoke test done |
| 07 | Evaluate & visualise | `src/07_evaluate.py`, `src/07b_multi_object.py` (several objects per photo) | [docs/07_evaluation.md](docs/07_evaluation.md) | smoke test done |
| 08 | Look inside the transformer (attention per square) | `src/08_explain_attention.py` | [docs/08_attention_explained.md](docs/08_attention_explained.md) | smoke test done |

| 04b | 8 views + 7×7 attention window (E10 + E11, variant `T8w7`) | `src/04b_build_samples_multi.py`; steps 05–08 with `--variant T8w7` | [docs/04b_multiview.md](docs/04b_multiview.md) | smoke test done |
| 10 | One decision per object: counting (no double counting) + type (E12) | `src/10_per_object.py`, `src/backbone.py` | [docs/10_per_object.md](docs/10_per_object.md) | smoke test done |
| 09 | Speed check (real-time?) | `src/09_speed.py` | [docs/00_simple_overview.md](docs/00_simple_overview.md) §11 | done: ≈ 7 ms/frame, 140–158 fps on RTX 3050 |

| 11 | Intermediate-update slide deck | `src/11_make_report_ppt.py` | [reports/BeltPulse_Intermediate_Update.pptx](reports/BeltPulse_Intermediate_Update.pptx) | done (re-run after full training) |

Ideas beyond the core pipeline: [docs/extensions.md](docs/extensions.md).
What the original proposal claimed vs. what holds: [docs/proposal_review.md](docs/proposal_review.md).
Plan and checklist for the full run: [docs/full_run_plan.md](docs/full_run_plan.md).

### Run the whole pipeline
```powershell
# smoke test (≈ 3 min): checks everything works, numbers are not meaningful
.venv\Scripts\python src\05_extract_features.py --limit 100
.venv\Scripts\python src\06_train.py --limit 100
.venv\Scripts\python src\07_evaluate.py --limit 100

# full run (≈ 15 min features + ~10 min per model)
.venv\Scripts\python src\05_extract_features.py
.venv\Scripts\python src\06_train.py
.venv\Scripts\python src\07_evaluate.py
```
Steps 01–04 only need re-running when labels change (`03c` → `04`).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Run any step from the project root, e.g. `.venv\Scripts\python src\02_belt_motion.py`.

## Folder layout

```
data/raw/ironore/   original dataset (do not edit)
data/meta/          derived measurements (belt motion per video)
data/labels/        keyframes, keyframe boxes, propagated boxes
src/                one script per step, numbered; common.py = shared helpers
outputs/<step>/     pictures/videos produced by each step — look at these!
docs/               one markdown note per activity
```

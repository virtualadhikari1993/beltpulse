# 11 — Final report deck

**Output:** `reports/BeltPulse_Final_Report.pptx` (26 slides). The older intermediate decks (`reports/BeltPulse_Intermediate_Update*.pptx`, `src/11_make_report_ppt.py`) are unchanged.

**Rebuild:**
```
.venv\Scripts\python reports\final_report\render_figs.py   # SVG figures of docs/pipeline_architecture.html -> reports/final_report/figs/*.png (headless Chrome)
node reports\final_report\build.js                          # writes the deck (pptxgenjs, installed in reports/final_report/node_modules)
```
Numbers are typed into `build.js` from `outputs/07_eval/full*/metrics.md` and `docs/07_evaluation.md`. Change them there if results change.
`img_dims.json` holds the image sizes, so pictures keep their aspect ratio.

**Story of the deck**
1. Introduction: problem (glare, rolling), data (19 videos, no locations), summary of results.
2. Method: belt motion → labels (LLM keyframes, belt-physics propagation, rolling correction, human review and correction) → samples (3 or 8 views,
   simulated glare) → model (frozen ResNet-18 + 269k trained parameters) → kinematic alignment → cross-frame attention → six fusion models + votes → training.
3. Results: 3-view and 8-view benchmarks, 3 vs 8 views, three seeds, glare examples, the real-glare/rolling video `jaspilito_4`, alignment on `canga3`,
   attention weights, per-video table.
4. Conclusions, limitations (labels, seeds, counting, val/test gap, rolling score, simulated glare), next steps.

**Framing used:** BeltPulse-Net is the whole system (belt alignment + multi-view samples + fusion models), and cross-frame attention is one of its fusion options.
Cross-frame attention matches feature averaging in accuracy (0.626 vs 0.620, 3 seeds) and is about 3× more stable across seeds (sd 0.011 vs 0.033).

## Web version (GitHub)
- **Repo landing page:** `README.md` is the full report in Markdown (GitHub renders it on the repo page; it does not run custom HTML/CSS).
- **Styled web page:** `docs/index.html`, served by GitHub Pages at https://virtualadhikari1993.github.io/beltpulse/
  (Settings → Pages → Source: *Deploy from a branch*, branch `main`, folder `/docs`). `docs/.nojekyll` makes Pages serve the files as they are.
- Both use the pictures and charts in `docs/report/img/`, made by `reports/final_report/make_web_assets.py` (Pages can only serve files inside `docs/`).

"""Collect the pictures and draw the charts used by README.md and docs/index.html (the web version of the final report).

Run from the project root:  .venv\\Scripts\\python reports\\final_report\\make_web_assets.py
Output: docs/report/img/*.png|jpg   (GitHub Pages serves docs/, so the web report can only use files inside it)
"""
import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "docs", "report", "img")
os.makedirs(OUT, exist_ok=True)

COPY = {
    "reports/final_report/figs/pipeline.png": "pipeline.png",
    "reports/final_report/figs/sample.png": "sample.png",
    "reports/final_report/figs/model.png": "model.png",
    "reports/final_report/figs/alignment.png": "alignment.png",
    "reports/final_report/figs/trust.png": "trust.png",
    "reports/final_report/figs/baselines.png": "baselines.png",
    "outputs/04_samples/glare_examples.jpg": "glare_examples.jpg",
    "outputs/01_explore/anomaly_samples.jpg": "anomaly_samples.jpg",
    "outputs/03_labeling/correction_iron_ore_canga2.jpg": "label_correction.jpg",
    "outputs/06_train/full_T8w7/curves.png": "curves_8view.png",
    "outputs/07_eval/full/examples_02.jpg": "example_glare_1.jpg",
    "outputs/07_eval/full/examples_05.jpg": "example_glare_2.jpg",
    "outputs/07_eval/full_T8w7/examples_03.jpg": "example_glare_3.jpg",
    "outputs/08_attention/full_T8w7/square_00_A_glare.jpg": "attention_square.jpg",
    "outputs/07_eval/full_T8w7/pr_curves.png": "pr_curves_8view.png",
}
for src, dst in COPY.items():
    shutil.copyfile(os.path.join(ROOT, src), os.path.join(OUT, dst))

INK, MUTED, GRID = "#1a2128", "#56626d", "#e3e7ea"
TEAL, AMBER, GREY, LIGHT, RUST = "#0d7a72", "#a8650a", "#9aa4ad", "#b9c2c9", "#b4541e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#c6ced5", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def hbar(name, title, rows, lo, hi):
    """rows: (label, value, kind) with kind in vote/fusion/main."""
    col = {"vote": GREY, "fusion": AMBER, "main": TEAL}
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.1))
    y = range(len(rows))
    ax.barh(list(y), [r[1] for r in rows], color=[col[r[2]] for r in rows], height=0.65)
    for i, (_, v, _) in enumerate(rows):
        ax.text(v + (hi - lo) * 0.008, i, f"{v:.3f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(lo, hi)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel("test object IoU")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREY, AMBER, TEAL)]
    ax.legend(handles, ["single frame and votes", "other fusion models", "feature average and cross-frame attention"],
              loc="upper center", bbox_to_anchor=(0.35, -0.16), ncol=3, fontsize=9, frameon=False)
    save(fig, name)


hbar("bench_3view.png", "3 views (seed 0): every multi-view model beats the single frame", [
    ("single frame", 0.583, "vote"), ("single + OR vote", 0.585, "vote"), ("single + majority vote", 0.615, "vote"),
    ("noalign (attention, no alignment)", 0.615, "fusion"), ("single + mean vote", 0.620, "vote"), ("median", 0.621, "fusion"),
    ("glareskip", 0.630, "fusion"), ("beltpulse (cross-frame attention)", 0.635, "main"), ("avg (feature average)", 0.658, "main")], 0.55, 0.70)
hbar("bench_8view.png", "8 views + 7×7 window: the gains grow", [
    ("single + OR vote", 0.579, "vote"), ("single frame", 0.622, "vote"), ("noalign (attention, no alignment)", 0.626, "fusion"),
    ("glareskip", 0.640, "fusion"), ("single + 2-of-N vote", 0.643, "vote"), ("median", 0.647, "fusion"),
    ("single + majority vote", 0.666, "vote"), ("beltpulse (cross-frame attention)", 0.666, "main"), ("single + mean vote", 0.671, "vote"),
    ("avg (feature average)", 0.682, "main")], 0.55, 0.72)


def grouped(name, title, cats, a, b, la, lb, lo, hi, ca=LIGHT, cb=TEAL, fmt="{:.3f}", w=9):
    fig, ax = plt.subplots(figsize=(w, 4.2))
    x = range(len(cats))
    bw = 0.38
    for off, vals, c, lab in ((-bw / 2, a, ca, la), (bw / 2, b, cb, lb)):
        bars = ax.bar([i + off for i in x], vals, bw, color=c, label=lab)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + (hi - lo) * 0.01, fmt.format(v), ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(x), cats)
    ax.set_ylim(lo, hi)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.25 if any("\n" in c for c in cats) else -0.12))
    save(fig, name)


grouped("views_3_vs_8.png", "More views help every multi-view model (test object IoU)",
        ["single\nframe", "single +\nmean vote", "median", "glareskip", "noalign", "cross-frame\nattention", "feature\naverage"],
        [0.583, 0.620, 0.621, 0.630, 0.615, 0.626, 0.620], [0.622, 0.671, 0.647, 0.640, 0.626, 0.666, 0.682], "3 views", "8 views", 0.5, 0.72)
grouped("seeds.png", "Three training seeds, 3 views (test object IoU)", ["seed 0", "seed 1", "seed 2", "mean"],
        [0.658, 0.599, 0.604, 0.620], [0.635, 0.629, 0.614, 0.626], "feature average (sd 0.033)", "cross-frame attention (sd 0.011)", 0.55, 0.68, w=7.5)
grouped("attention_weights.png", "Average attention weight per view on object squares", ["3 views, seed 0", "3 views, seed 1", "8 views"],
        [0.160, 0.126, 0.088], [0.340, 0.342, 0.128], "glared view", "clean view", 0, 0.4, ca=RUST, w=7.5)
grouped("real_glare_rolling.png", "jaspilito_4: real glare, motion blur and rolling (test object IoU)",
        ["single\nframe", "single +\nmean vote", "noalign", "cross-frame\nattention", "median", "feature\naverage"],
        [0.628, 0.678, 0.704, 0.699, 0.735, 0.725], [0.705, 0.756, 0.678, 0.751, 0.730, 0.747], "3 views", "8 views", 0.55, 0.80, fmt="{:.2f}")

vids = ["iron_ore", "iron_ore_2", "iron_ore_3", "iron_ore_4", "iron_ore_5", "iron_ore_6", "7_wood", "7_wood_2", "7_wood_3", "canga", "canga2",
        "canga3", "jaspilito", "jaspilito_2", "jaspilito_3", "jaspilito_4", "things", "things_2", "things_3"]
dx = [0.03, 9.47, 8.69, 12.20, 47.82, 0.01, 11.69, 17.95, 25.49, 24.65, 19.97, 32.81, 1.44, 10.63, 26.18, 39.85, 17.75, 17.28, 20.91]
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.bar(vids, dx, color=AMBER)
ax.set_ylabel("px per frame")
ax.set_title("Median belt speed per video (along X; vertical shift ≈ 0 everywhere)", loc="left", fontsize=12, color=INK)
ax.yaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
save(fig, "belt_speed.png")
print("assets in", OUT, len(os.listdir(OUT)))

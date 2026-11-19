"""Step 11 - Build the intermediate-update slide deck (PowerPoint) from the project's results and pictures.

Run:  .venv\\Scripts\\python src\\11_make_report_ppt.py
Output: reports/BeltPulse_Intermediate_Update.pptx
Re-run after the full training to refresh the deck (update the numbers in the RESULTS section below).
"""
import os
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTD = os.path.join(ROOT, "reports")
os.makedirs(OUTD, exist_ok=True)
O = lambda *p: os.path.join(ROOT, "outputs", *p)

NAVY = RGBColor(0x1F, 0x2A, 0x44)
TEAL = RGBColor(0x1B, 0x99, 0x8B)
ORANGE = RGBColor(0xE8, 0x6A, 0x33)
GREY = RGBColor(0x5F, 0x67, 0x73)
LIGHT = RGBColor(0xF2, 0xF4, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x22, 0x22, 0x22)
PALE_TEAL = RGBColor(0xE3, 0xF4, 0xF2)
PALE_ORANGE = RGBColor(0xFD, 0xEB, 0xE2)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W = 13.333
state = {"n": 0}


# ---------------------------------------------------------------- helpers
def new_slide(title, kicker=None):
    s = prs.slides.add_slide(BLANK)
    state["n"] += 1
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.0))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.12), Inches(11.5), Inches(0.8))
    tf = tb.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = title; p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = WHITE
    if kicker:
        k = s.shapes.add_textbox(Inches(0.5), Inches(1.08), Inches(12.3), Inches(0.45))
        q = k.text_frame.paragraphs[0]; q.text = kicker; q.font.size = Pt(15); q.font.italic = True; q.font.color.rgb = TEAL
    foot = s.shapes.add_textbox(Inches(0.5), Inches(7.08), Inches(12.3), Inches(0.3))
    f = foot.text_frame.paragraphs[0]
    f.text = f"BeltPulse-Net · intermediate update · {state['n']}"
    f.font.size = Pt(10); f.font.color.rgb = GREY; f.alignment = PP_ALIGN.RIGHT
    return s


def text(s, x, y, w, h, items, size=16, color=DARK, bold_first=False, spacing=6):
    """items: list of str or (str, level) or (str, level, {'bold':..,'color':..})."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        if isinstance(it, str):
            it = (it, 0, {})
        elif len(it) == 2:
            it = (*it, {})
        t, lvl, opt = it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        bullet = "•  " if lvl == 0 else "–  "
        p.text = ("" if opt.get("nobullet") else bullet) + t
        p.level = lvl
        p.font.size = Pt(opt.get("size", size - 2 * lvl))
        p.font.bold = opt.get("bold", False)
        p.font.color.rgb = opt.get("color", color)
        p.space_after = Pt(spacing)
        if lvl:
            p.left_indent = Inches(0.3 * lvl)
    return tb


def box(s, x, y, w, h, txt, fill=LIGHT, fc=DARK, size=12, bold=False, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    b = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    b.fill.solid(); b.fill.fore_color.rgb = fill
    if line is None:
        b.line.fill.background()
    else:
        b.line.color.rgb = line; b.line.width = Pt(2.25)
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for m in ("margin_left", "margin_right"):
        setattr(tf, m, Inches(0.06))
    for i, line_txt in enumerate(txt.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line_txt; p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(size if i else size + (1 if bold else 0)); p.font.color.rgb = fc
        p.font.bold = bold and i == 0
    return b


def arrow(s, x1, y1, x2, y2, color=GREY):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(2)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


def image(s, path, x, y, w=None, h=None):
    if not os.path.exists(path):
        return box(s, x, y, w or 4, h or 2, f"[missing: {os.path.basename(path)}]")
    iw, ih = Image.open(path).size
    if w and h:
        if iw / ih > w / h:
            h = w * ih / iw
        else:
            w = h * iw / ih
    elif w:
        h = w * ih / iw
    else:
        w = h * iw / ih
    return s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))


def table(s, x, y, w, rows, col_w=None, size=12, row_h=0.36, highlight=None, header_fill=NAVY, bold_cells=(), left_cols=(0,)):
    nr, nc = len(rows), len(rows[0])
    t = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr)).table
    col_w = col_w or [w / nc] * nc
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = t.cell(i, j); c.text = str(v)
            c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.03)
            p = c.text_frame.paragraphs[0]
            p.font.size = Pt(size); p.alignment = PP_ALIGN.LEFT if j in left_cols else PP_ALIGN.CENTER
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = header_fill; p.font.bold = True; p.font.color.rgb = WHITE
            else:
                c.fill.fore_color.rgb = PALE_TEAL if (highlight and i in highlight) else (WHITE if i % 2 else LIGHT)
                p.font.color.rgb = DARK
                if (i, j) in bold_cells or (highlight and i in highlight):
                    p.font.bold = True
    return t


def note(s, txt):
    s.notes_slide.notes_text_frame.text = txt


# ================================================================ 1. title
s = prs.slides.add_slide(BLANK)
state["n"] += 1
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
text(s, 0.8, 1.7, 11.8, 1.2, [("BeltPulse-Net", 0, {"nobullet": True, "size": 54, "bold": True, "color": WHITE})])
text(s, 0.8, 2.8, 11.8, 1.4, [("Kinematics-aligned, multi-view impurity segmentation on iron-ore conveyor belts", 0,
                               {"nobullet": True, "size": 26, "color": RGBColor(0xBF, 0xE6, 0xE1)})])
text(s, 0.8, 4.5, 11.8, 1.5, [("Intermediate project update · Foundations of Machine Learning term project · IIT Hyderabad", 0,
                               {"nobullet": True, "size": 18, "color": WHITE}),
                              ("November 2026 · status: full pipeline built and smoke-tested; full training pending", 0,
                               {"nobullet": True, "size": 16, "color": RGBColor(0xC8, 0xCE, 0xD8)})])

# ================================================================ 2. problem
s = new_slide("Problem: sorting impurities on a fast conveyor", "Why single-frame vision fails, and what the belt gives us for free")
text(s, 0.5, 1.6, 6.3, 5.3, [
    "Iron ore runs on a conveyor; foreign objects (wood, plastic, rocks of the wrong type, tools) must be detected and ejected",
    "Standard systems judge every camera frame in isolation",
    ("Overhead lamps cause glare: a washed-out frame hides the object or looks like white plastic", 1),
    ("Motion blur, odd orientation, objects half-buried in ore", 1),
    "But the belt is a rigid, (mostly) predictable motion",
    ("The same object is seen in many consecutive frames (camera: 120 fps)", 1),
    ("If we know how far the belt moved, we know where the object is in every frame: no object tracker needed", 1),
    ("Idea: combine several aligned views so a glared view can be compensated by clean ones", 1, {"bold": True, "color": TEAL}),
], size=17)
pic = image(s, O("04_samples", "sample_01.jpg"), 7.0, 1.65, w=5.9)
cy = 1.65 + pic.height / 914400
text(s, 7.0, cy + 0.05, 5.9, 0.7, [("Same stick in 3 frames of the belt. Fake glare blinds the last view; the other two still show it clearly.", 0,
                                   {"nobullet": True, "size": 13, "color": GREY})])
pic2 = image(s, O("01_explore", "overview.jpg"), 7.0, cy + 0.85, h=6.85 - (cy + 0.85))
text(s, 7.0 + pic2.width / 914400 + 0.15, cy + 0.85, 5.9 - pic2.width / 914400 - 0.15, 2.5,
     [("One frame from each of the 19 videos: ore only, and ore with wood, rocks, plastic and tools", 0,
       {"nobullet": True, "size": 13, "color": GREY})])

# ================================================================ 3. proposal
s = new_slide("Proposal recap and the questions we test", "Submitted proposal: CNN texture extraction + physics alignment + cross-frame transformer attention")
text(s, 0.5, 1.6, 6.2, 2.4, [
    ("Proposed four-stage pipeline", 0, {"bold": True}),
    ("1. Pre-trained CNN extracts local texture features", 1),
    ("2. Bottleneck compresses them to compact tokens", 1),
    ("3. Physics translation (belt speed) aligns frames: no tracker", 1),
    ("4. Transformer attends across frames, \"sees through\" glare", 1),
], size=17)
table(s, 0.5, 4.15, 6.2, [
    ["Claim being tested", "Comparison"],
    ["C1  Several views beat one view under glare", "single vs avg / beltpulse"],
    ["C2  Belt kinematics align views (no tracker)", "noalign vs avg"],
    ["C3  Learned attention beats simple fusion", "beltpulse vs avg / voting"],
], col_w=[3.9, 2.3], size=13, row_h=0.45)
box(s, 7.1, 1.65, 5.7, 0.5, "Corrections after working with the real data", fill=ORANGE, fc=WHITE, size=15, bold=True,
    shape=MSO_SHAPE.RECTANGLE)
text(s, 7.1, 2.25, 5.7, 4.6, [
    "The dataset has no impurity locations: labels had to be created (keyframes + physics)",
    "The belt moves along X (not Y), and its speed varies and the belt even stops: motion is measured per frame",
    "Output is a coarse 40×22 grid learned from boxes, not a \"pixel-perfect\" mask",
    "Closest prior work: flow-guided feature aggregation for video detection (FGFA, ICCV 2017). Our variant replaces optical flow by belt kinematics",
    ("With only 3 aligned views, attention has little to do beyond averaging → design extended (E10–E12): "
     "8 views, a 7×7 search window, per-object sequences, jobs simple fusion cannot do", 0, {"bold": True, "color": TEAL}),
    ("Whether attention actually beats simple fusion (C3) is decided by the full run", 0),
], size=15)
note(s, "C1 and C2 are the heart of the proposal. If attention turns out to equal averaging, C1/C2 can still hold; that is a valid finding.")

# ================================================================ 4. dataset
s = new_slide("Data: in-lab iron-ore conveyor recordings", "Mendeley Data s25x2bnshz (Sousa et al., 2025), CC BY 4.0")
table(s, 0.5, 1.6, 6.3, [
    ["Property", "Value"],
    ["Videos", "19 (6 ore only, 13 with foreign objects)"],
    ["Frames", "26,460 consecutive frames, 1280×720"],
    ["Frame rate", "≈ 120 fps (Jetson TX2 camera, top-down)"],
    ["Belt", "35 cm × 1.10 m, up to ≈ 3 m/s"],
    ["Foreign objects", "wood sticks · white plastic · canga rock · jaspilito rock · tools (hammer, pliers, wood block)"],
    ["Original labels", "frame-level only (2,990 frames \"has object\"); YOLO boxes mark the belt, not impurities"],
], col_w=[1.7, 4.6], size=12, row_h=0.55)
table(s, 0.5, 5.6, 6.3, [
    ["Kind", "Videos", "Frames"],
    ["ore only", "6", "13,507"],
    ["wood / canga+plastic / jaspilito / tools", "3 / 3 / 4 / 3", "3,663 / 2,721 / 3,755 / 2,814"],
], col_w=[2.9, 1.3, 2.1], size=11, row_h=0.4)
image(s, O("03_labeling", "labels_keyframes", "_overview_iron_ore_canga3.jpg"), 7.1, 1.6, w=5.8)
text(s, 7.1, 4.95, 5.8, 1.9, [
    ("Belt motion measured by phase correlation between consecutive frames: 9 to 48 px/frame depending on the video; "
     "in some videos the belt stops and restarts (real \"belt slip\")", 0, {"size": 13}),
    ("Real difficulties present: glare on the bare belt, heavy motion blur in fast videos, objects half-buried", 0, {"size": 13}),
], size=13)

# ================================================================ 5. labels
s = new_slide("Creating impurity labels: 91 keyframes → 2,949 frames",
              "Teacher → student: a slow LLM labels a few keyframes offline; belt physics spreads them; a fast model learns")
xs = [0.4, 2.55, 4.7, 6.85, 9.0, 11.15]
steps = [("26,460 frames", "19 videos"), ("2,990 frames", "authors: \"has object\"\n30 object passes"),
         ("91 keyframes", "boxes drawn by an LLM\n67 regular + 24 extra"), ("Physics propagation", "shift boxes by measured\nbelt motion"),
         ("2,949 labeled frames", "3,355 boxes\n5 classes"), ("3,694 examples", "train 2,699 · val 470\ntest 525")]
for i, (x, (a, b)) in enumerate(zip(xs, steps)):
    box(s, x, 1.75, 1.85, 1.25, f"{a}\n{b}", fill=PALE_TEAL if i in (2, 3) else LIGHT, size=11, bold=True,
        line=TEAL if i in (2, 3) else None)
    if i < 5:
        arrow(s, x + 1.86, 2.37, xs[i + 1] - 0.02, 2.37)
text(s, 0.5, 3.25, 6.3, 3.7, [
    ("Keyframe every ≈ 700 px of belt travel; boxes [x1, y1, x2, y2, class]", 0),
    ("Two-keyframe correction: between keyframes A and B, blend A's and B's physics copies → rolling objects followed smoothly, no jump at hand-over", 0),
    ("Rolling detection: where physics misses the next keyframe by > 60 px, 24 extra keyframes were labeled directly (ground truth in rolling regions)", 0),
    ("Flagged intervals 14 → 7, worst error 161 → 120 px", 1),
    ("Labels are being reviewed by hand before the full run", 0, {"bold": True, "color": ORANGE}),
], size=14)
pic = image(s, O("03_labeling", "correction_iron_ore_canga2.jpg"), 7.0, 3.3, w=5.9)
text(s, 7.0, 3.35 + pic.height / 914400, 5.9, 0.6, [("Rolling rock. Red = plain physics copy (drifts off); green = two-keyframe correction.", 0,
                                                    {"nobullet": True, "size": 12, "color": GREY})])

# ================================================================ 6. input / output
s = new_slide("Input and output of one training example", "T views of the same belt spot + optional glare  →  40 × 22 grid of impurity / ore")
image(s, O("04_samples", "box_to_grid.jpg"), 0.5, 1.6, w=12.3)
text(s, 0.5, 4.0, 6.2, 3.0, [
    ("Input", 0, {"bold": True, "color": TEAL}),
    ("T photos of the same spot: t−k … t+k; k chosen from belt speed so the object moves ≈ 2–3 squares between views", 1),
    ("Each photo 640×352 → frozen ResNet-18 → 256 numbers per square (40 × 22 squares, 1 square ≈ 32 px)", 1),
    ("Belt shift of every view (pixels), measured from the video", 1),
    ("Synthetic glare (bright soft blob, 70% placed on the object) on random views", 1),
], size=15)
text(s, 6.9, 4.0, 6.0, 3.0, [
    ("Output / label", 0, {"bold": True, "color": TEAL}),
    ("Per square of the target view t: probability of \"foreign object\"", 1),
    ("Label: square = 1 if ≥ 25% covered by a box (≈ 6% of squares are 1)", 1),
    ("Negatives: ore-only videos (answer all zeros), 60% of examples", 1),
    ("Split by whole video (no near-duplicate frames across splits)", 1),
    ("test: wood_3, canga3, jaspilito_4, things_3, iron_ore_4 · val: canga, jaspilito_2, iron_ore_3", 2),
], size=15)

# ================================================================ 7. architecture
s = new_slide("Architecture", "Orange border = trained · grey = fixed (pre-trained or physics) · two outputs: per-frame mask and per-object decision")
y1, h1 = 1.75, 1.45
row1 = [("T camera frames", "t−3k … t+4k\n640 × 352 each", None),
        ("ResNet-18\n(frozen, cached)", "layer3: 256 × 22 × 40\nper frame, run once", None),
        ("Bottleneck", "1×1 conv\n256 → 96 per square", ORANGE),
        ("Kinematic alignment", "shift each view by\nmeasured belt motion", None),
        ("Cross-frame attention", "query: square in t\nkeys: win×win in all T views", ORANGE),
        ("Spatial transformer", "880 squares attend\nto each other", ORANGE),
        ("Head → mask", "40 × 22 impurity\nprobability", ORANGE)]
bw, gap = 1.62, 0.2
col = lambda i: 0.4 + i * (bw + gap)
for i, (a, b, ln) in enumerate(row1):
    box(s, col(i), y1, bw, h1, f"{a}\n{b}", fill=PALE_ORANGE if ln else LIGHT, line=ln, size=10, bold=True)
    if i < len(row1) - 1:
        arrow(s, col(i) + bw, y1 + h1 / 2, col(i) + bw + gap, y1 + h1 / 2)
text(s, col(5), y1 - 0.4, 2 * bw + gap, 0.35, [("Output 1: impurity mask per frame", 0, {"nobullet": True, "size": 11, "color": TEAL, "bold": True})])
# belt motion feeds the alignment (up) and the tracker (down-right)
bm_y = 3.5
box(s, col(3), bm_y, bw, 0.8, "Belt motion dx\nper frame (phase correlation)", fill=PALE_TEAL, size=10, bold=True)
arrow(s, col(3) + bw / 2, bm_y, col(3) + bw / 2, y1 + h1)
# second row runs right -> left, starting under the mask head
y2 = 4.85
row2 = [(6, "Blobs → boxes", "touching impurity\nsquares per frame", None),
        (5, "Physics tracker", "link boxes frame to\nframe by belt shift", None),
        (4, "Frame tokens", "ResNet pooled in box\n(mean + max, 512)", None),
        (3, "Sequence transformer", "[CLS] + frame tokens\n2 layers", ORANGE),
        (2, "One decision per object", "ore / wood / plastic /\ncanga / jaspilito / tool\n→ counted once", ORANGE)]
for n, (c, a, b, ln) in enumerate(row2):
    box(s, col(c), y2, bw, h1, f"{a}\n{b}", fill=PALE_ORANGE if ln else LIGHT, line=ln, size=10, bold=True)
    if n < len(row2) - 1:
        arrow(s, col(c), y2 + h1 / 2, col(c) - gap, y2 + h1 / 2)
arrow(s, col(6) + bw / 2, y1 + h1, col(6) + bw / 2, y2)                     # mask -> blobs
arrow(s, col(3) + bw, bm_y + 0.4, col(5) + bw / 2, y2)                       # belt motion -> tracker
text(s, col(0), y2 + 0.25, 2 * bw + gap - 0.1, 1.0, [("Output 2 (E12): one decision per object, count and type", 0,
                                                      {"nobullet": True, "size": 11, "color": TEAL, "bold": True})])
text(s, 0.4, 6.35, 12.5, 0.7, [("Variants:  base = 3 views, 3×3 window (27 tokens per square)   ·   T8w7 = 8 views, 7×7 window (392 tokens; "
                               "attention can search ±3 squares for objects that rolled away from the physics prediction)", 0,
                               {"nobullet": True, "size": 13, "color": GREY})])
note(s, "Only ~270k parameters are trained; ResNet (11M) is frozen and its features are cached once. Per-frame cost measured at ~7 ms on an RTX 3050 laptop GPU (140-158 fps).")

# ================================================================ 8. what attention does
s = new_slide("What the transformer is asked to do",
              "Jobs that averaging or voting cannot do: choose among many views, search for moved objects, read variable-length object sequences")
image(s, O("08_attention", "smoke40_T8w7", "square_00_A_glare.jpg"), 0.5, 1.55, h=5.35)
text(s, 5.45, 1.65, 7.45, 5.3, [
    ("Cross-frame attention, for one square of view t", 0, {"bold": True}),
    ("query = the square in t; keys/values = its win×win neighbourhood in every aligned view", 1),
    ("weights sum to 1: ideally low on glared / blurred views, high on clean ones", 1),
    ("wide window: attention can find an object that rolled away from the physics prediction", 1),
    ("Spatial transformer: each square compared with the rest of the belt (context)", 0, {"bold": True}),
    ("Sequence transformer (per object): reads all frames of a tracked object, outputs one class", 0, {"bold": True}),
    ("Left: 8 aligned views of one square (7×7 windows). The stick sits at the same place in every view, so the alignment works. "
     "Glared views are marked; numbers = attention.", 0, {"size": 13, "color": GREY}),
    ("Status: E10/E11/E12 built and smoke-tested. After only 3 epochs on 40 samples attention is still uniform (12–13% per view): "
     "the design gives attention a real job; the full run shows whether it learns it.", 0, {"size": 13, "bold": True, "color": ORANGE}),
], size=15)

# ================================================================ 9. benchmarks
s = new_slide("What we compare against", "Every model sees exactly the same cached features and labels; only the fusion differs")
table(s, 0.5, 1.6, 7.2, [
    ["Per-frame models (mask)", "Sees", "Tests"],
    ["single", "target view t only", "classic single-frame baseline"],
    ["avg", "aligned views, averaged", "simple fusion (C3)"],
    ["noalign", "views not aligned", "value of kinematics (C2)"],
    ["beltpulse", "aligned views + attention", "our method"],
    ["× variants", "base (3 views, 3×3) · T8w7 (8 views, 7×7)", "E10 + E11"],
], col_w=[1.5, 3.0, 2.7], size=12, row_h=0.45, highlight=[4])
table(s, 0.5, 4.55, 7.2, [
    ["Per-object models", "Decision rule"],
    ["SeqTransformer (ours)", "attention over all frames of the track"],
    ["MeanPool", "average frame tokens, then MLP"],
    ["FrameMLP + OR / k-of-N / mean", "classify frames alone, then vote"],
], col_w=[3.0, 4.2], size=12, row_h=0.42, highlight=[1])
text(s, 8.0, 1.6, 4.9, 5.3, [
    ("Metrics", 0, {"bold": True, "color": TEAL}),
    ("object IoU, mIoU (proposal metric), F1 over grid squares", 1),
    ("split: clean target view vs. glare on target view", 1),
    ("recall under glare: object squares hidden by glare in t that are still found", 1),
    ("attention a view gets where it is glared vs. clean", 1),
    ("per object: impurities found, false alarms on ore, correct type", 1),
    ("counting: true objects vs. counted tracks per test video", 1),
    ("speed: ms per frame on the laptop GPU", 1),
    ("Planned extra baselines (E9)", 0, {"bold": True, "color": ORANGE}),
    ("late fusion: single model on each view, then OR / k-of-N / mean", 1),
    ("rule-based: skip saturated (white) squares, average the rest", 1),
], size=15)

# ================================================================ 10. smoke results - masks
s = new_slide("Smoke test 1: per-square segmentation", "Pipeline check on tiny subsets: numbers show the code works, not which model is better")
table(s, 0.5, 1.6, 7.6, [
    ["base · 100 train / 25 val / 50 test · 25 ep.", "obj. IoU", "mIoU", "F1", "IoU glare-on-t", "recall under glare"],
    ["single", "0.521", "0.742", "0.685", "0.461", "0.489"],
    ["avg", "0.546", "0.754", "0.706", "0.560", "0.719"],
    ["noalign", "0.527", "0.742", "0.690", "0.570", "0.737"],
    ["beltpulse", "0.585", "0.776", "0.738", "0.610", "0.810"],
], col_w=[2.9, 0.85, 0.85, 0.85, 1.05, 1.1], size=12, row_h=0.4, highlight=[4])
table(s, 0.5, 3.85, 7.6, [
    ["T8w7 · 40 train / 20 val / 40 test · 3 ep.", "obj. IoU", "mIoU", "F1", "IoU glare-on-t", "recall under glare"],
    ["single", "0.329", "0.638", "0.495", "0.326", "0.433"],
    ["avg", "0.432", "0.695", "0.603", "0.430", "0.559"],
    ["noalign", "0.312", "0.630", "0.475", "0.306", "0.437"],
    ["beltpulse", "0.352", "0.648", "0.521", "0.343", "0.613"],
], col_w=[2.9, 0.85, 0.85, 0.85, 1.05, 1.1], size=12, row_h=0.4, highlight=[2])
text(s, 0.5, 6.0, 7.6, 1.0, [("Attention on glared vs clean views: base 0.346 vs 0.333 (uniform 0.333); T8w7 0.156 vs 0.122 (uniform 0.125): "
                              "not yet learned to distrust glare.", 0, {"nobullet": True, "size": 13, "color": GREY})])
image(s, O("07_eval", "smoke100", "attention.png"), 8.4, 1.55, w=4.5)
text(s, 8.4, 4.6, 4.5, 2.4, [
    ("Multi-view models find more glare-hidden object squares than single-frame (recall 0.49 → 0.72–0.81, base): a hint for C1", 0, {"size": 12}),
    ("Run-to-run noise ≈ ±0.05; T8w7 (E10+E11) is built but only trained 3 epochs on 40 samples, so it's too early to judge", 0, {"size": 12}),
    ("Above: base attention per view stays near uniform (1/3)", 0, {"size": 12}),
    ("Answer requires the full run", 0, {"size": 13, "bold": True, "color": ORANGE}),
], size=12, spacing=3)

# ================================================================ 11. smoke results - objects
s = new_slide("Smoke test 2: one decision per object and counting", "Every 8th frame of each track, 5 epochs · 14 test objects + 8 ore regions")
table(s, 0.5, 1.6, 7.0, [
    ["Method (50% of frames glared)", "impurities found", "false alarms on ore", "correct type"],
    ["SeqTransformer (ours)", "14/14", "0/8", "0.93"],
    ["MeanPool (average)", "13/14", "0/8", "0.93"],
    ["FrameMLP + OR", "14/14", "3/8", "0.93"],
    ["FrameMLP + k-of-N / mean", "14/14", "0/8", "0.93"],
], col_w=[2.8, 1.4, 1.5, 1.3], size=12, row_h=0.4, highlight=[1], bold_cells=[(3, 2)])
text(s, 0.5, 3.75, 7.0, 0.6, [("Without glare all methods are 14/14 found, 0/8 false alarms. OR-voting's false alarms under glare are the expected weakness.", 0,
                              {"nobullet": True, "size": 12, "color": GREY})])
table(s, 0.5, 4.4, 7.0, [
    ["Test video", "true objects", "raw detections", "tracks", "counted"],
    ["7_wood_3", "3", "119", "6", "4"],
    ["canga3", "6", "61", "6", "5"],
    ["jaspilito_3", "2", "60", "6", "2"],
    ["things_3", "3", "51", "3", "3"],
    ["iron_ore_4 (ore only)", "0", "11", "1", "1"],
], col_w=[2.2, 1.2, 1.4, 1.0, 1.2], size=12, row_h=0.36)
image(s, O("10_per_object", "smoke", "counting_iron_ore_canga3.jpg"), 7.8, 1.6, w=5.1)
text(s, 7.8, 4.55, 5.1, 2.4, [
    ("No double counting: detections are linked frame to frame by belt shift; each track ≥ 3 detections is counted once", 0, {"size": 13}),
    ("Remaining errors: weak smoke detector; touching objects merge (above: canga on a plastic strip → one track, called \"tool\")", 0, {"size": 13}),
], size=13)

# ================================================================ 12. full run plan
s = new_slide("Full training run: plan", "After the label review · same scripts, no --limit · overnight on the RTX 3050 (4 GB)")
table(s, 0.5, 1.6, 8.0, [
    ["Run", "train / val / test", "views · window", "epochs · batch", "est. time"],
    ["Mask models, base (4 models)", "2,699 / 470 / 525 examples", "3 · 3×3", "30 · 8", "≈ 1 h"],
    ["Mask models, T8w7 (4 models)", "2,627 / 458 / 507 examples", "8 · 7×7", "30 · 4", "≈ 3–4 h"],
    ["Per-object (5 models)", "50 / 14 / 22 tracks*", "every 3rd frame", "30", "≈ 15 min"],
    ["E9 baselines (late fusion, saturation mask)", "same test sets", "3 and 8", "no training", "minutes"],
], col_w=[2.6, 2.1, 1.1, 1.2, 1.0], size=11, row_h=0.5)
text(s, 0.5, 4.3, 8.0, 0.4, [("* tracks = objects + ore regions: train 18 + 32, val 6 + 8, test 14 + 8", 0,
                              {"nobullet": True, "size": 11, "color": GREY})])
text(s, 0.5, 4.75, 8.0, 2.3, [
    ("Test set: 5 unseen videos (one per object type + one ore-only): 340 object examples + 185 ore examples (base); "
     "half of the test examples have glare on the target view", 0, {"size": 14}),
    ("Validation (model selection): 3 videos (canga, jaspilito_2, iron_ore_3)", 0, {"size": 14}),
    ("Also evaluated on real difficult frames: real glare / motion blur in iron_ore_5, jaspilito_4, canga3", 0, {"size": 14}),
], size=14)
text(s, 8.8, 1.6, 4.1, 5.4, [
    ("Results to report", 0, {"bold": True, "color": TEAL}),
    ("table: object IoU, mIoU, F1 (all / clean / glare-on-t), recall under glare, for 4 models × 2 variants + E9 baselines", 1),
    ("attention on glared vs clean views (does attention learn to distrust glare?)", 1),
    ("attention centre offset on rolling objects (does the 7×7 window follow them?)", 1),
    ("per object: found / false alarms / type accuracy, with precision–recall curves for the voting rules", 1),
    ("counts per test video", 1),
    ("speed per frame", 1),
], size=14)

# ================================================================ 13. enhancements
s = new_slide("Enhancements on top", "Logged in docs/extensions.md · status after this update")
table(s, 0.5, 1.55, 12.3, [
    ["ID", "Enhancement", "Why", "Status"],
    ["E1", "Pixel-precise masks: SAM turns boxes into outlines + upsampling decoder", "today the labels are box-shaped", "idea"],
    ["E2", "Anomaly detection from ore-only videos (memory bank / masked-square prediction)", "no impurity labels needed", "idea"],
    ["E3", "Motion cue: objects move with the belt, lamp glare stays fixed in the camera", "separate glare from objects", "idea"],
    ["E5 / E13", "Separate touching objects (centre heat-map, DETR-style object queries)", "merged blobs miscount", "idea / future"],
    ["E6", "Multi-class mask (which impurity, per square)", "labels already have classes", "idea"],
    ["E7", "Measure LLM label quality vs. own labels; train on both", "teacher quality question", "idea"],
    ["E8", "Rolling objects: two-keyframe correction, extra keyframes; template matching, optical flow", "physics assumes still objects", "partly done"],
    ["E9", "Late-fusion and saturation-mask baselines, real-glare test", "does attention earn its place?", "next"],
    ["E10 / E11", "8 views + 7×7 attention window", "harder job for attention", "built, smoke-tested"],
    ["E12", "Per-object decision, counting with physics tracker, object type", "what a sorter needs", "built, smoke-tested"],
], col_w=[0.9, 6.4, 3.4, 1.6], size=12, row_h=0.42, highlight=[9, 10], left_cols=(0, 1, 2))

# ================================================================ 14. summary
s = new_slide("Summary and next steps")
text(s, 0.5, 1.4, 6.2, 5.5, [
    ("Done", 0, {"bold": True, "color": TEAL, "size": 20}),
    ("Real dataset prepared; belt motion measured per frame", 1),
    ("Impurity labels created: 91 keyframes → 2,949 frames via physics, with rolling correction", 1),
    ("Full pipeline: features → alignment → attention → mask; per-object counting and type", 1),
    ("4 mask models × 2 variants, 5 per-object methods, all smoke-tested", 1),
    ("Real-time: ≈ 7 ms per frame (140–158 fps) on a laptop GPU", 1),
    ("Docs and visualisations for every step", 1),
], size=17)
text(s, 6.9, 1.4, 6.0, 5.5, [
    ("Next", 0, {"bold": True, "color": ORANGE, "size": 20}),
    ("Review and correct the labels (keyframes + extra rolling keyframes)", 1),
    ("Full runs: base, T8w7, per-object", 1),
    ("E9 baselines and real-glare evaluation", 1),
    ("Answer C1–C3 honestly, incl. \"attention ≈ averaging\" if so", 1),
    ("Report: results, attention analysis, limitations, extensions", 1),
    ("Open risk: small number of objects (≈ 30 passes) → wide uncertainty on per-object results", 0, {"size": 14, "color": GREY}),
], size=17)
for i, (a, b) in enumerate([("26,460", "frames, 19 videos"), ("91 → 2,949", "keyframes → labeled frames"),
                            ("3,694", "training examples"), ("≈ 7 ms", "per frame (140–158 fps)")]):
    box(s, 0.5 + i * 3.1, 5.6, 2.9, 1.2, f"{a}\n{b}", fill=PALE_TEAL if i % 2 == 0 else LIGHT, size=13, bold=True)

import sys
path = os.path.join(OUTD, sys.argv[1] if len(sys.argv) > 1 else "BeltPulse_Intermediate_Update.pptx")
prs.save(path)
print("saved", path, f"({state['n']} slides)")

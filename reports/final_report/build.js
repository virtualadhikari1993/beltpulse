// Builds reports/BeltPulse_Final_Report.pptx (final project report).
// Run from the project root:  node reports/final_report/build.js
// Architecture figures come from docs/pipeline_architecture.html (render_figs.py); results from outputs/.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL_DIR
  ? path.join(process.env.PPTX_SKILL_DIR, "scripts", "apply_theme.js")
  : path.join(__dirname, "apply_theme.js"));

const ROOT = path.resolve(__dirname, "..", "..");
const OUT = path.join(ROOT, "reports", "BeltPulse_Final_Report.pptx");
const DIMS = JSON.parse(fs.readFileSync(path.join(__dirname, "img_dims.json"), "utf8"));
const P = (rel) => path.join(ROOT, rel);

// ---------------------------------------------------------------- theme
const HEX = {
  ink: "1A2128", charcoal: "232A31", paper: "FFFFFF", mist: "F1F3F4",
  teal: "0D7A72", rust: "B4541E", amber: "A8650A", slate: "66717C", line: "C6CED5", paleTeal: "E1F1EF", paleRust: "F6E4DA",
};
const THEME = {
  name: "BeltPulse Iron",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: HEX.ink, lt1: HEX.paper, dk2: HEX.charcoal, lt2: HEX.mist,
    accent1: HEX.teal, accent2: HEX.rust, accent3: HEX.amber, accent4: HEX.slate, accent5: "3B6E8F", accent6: "8A9A5B",
    hlink: HEX.teal, folHlink: HEX.slate,
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "BeltPulse-Net: final report";
pres.author = "IITH MTech project";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const SW = 13.333, M = 0.6;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 1.7, w: 7.4, h: 2.2, fontSize: 44, bold: true, color: C.background1, valign: "bottom", fontFace: "Cambria", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.05, w: 7.0, h: 1.6, fontSize: 18, color: "C9D2D8", valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.6, w: 9, h: 1.2, fontSize: 40, bold: true, color: C.background1, fontFace: "Cambria", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 3.85, w: 9, h: 1.0, fontSize: 18, color: "C9D2D8", align: "left" }, text: "" } },
  ],
  slideNumber: { x: SW - 1.1, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: "9BA7B1", align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: SW - 2 * M, h: 0.85, fontSize: 36, bold: true, color: C.background1, valign: "middle", fontFace: "Cambria", align: "left" }, text: "" } },
  ],
  slideNumber: { x: SW - 1.1, y: 6.98, w: 0.5, h: 0.3, fontSize: 10, color: "9BA7B1", align: "right" },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.7, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: SW - 2 * M, h: 0.85, fontSize: 32, bold: true, color: C.text1, valign: "middle", fontFace: "Cambria", align: "left" }, text: "" } },
    { text: { text: "BeltPulse-Net · final report", options: { x: M, y: 6.98, w: 6, h: 0.3, fontSize: 10, color: HEX.slate } } },
  ],
  slideNumber: { x: SW - 1.1, y: 6.98, w: 0.5, h: 0.3, fontSize: 10, color: HEX.slate, align: "right" },
});

// ---------------------------------------------------------------- helpers
let sectionName = "";
function section(title) { sectionName = title; pres.addSection({ title }); }
function content(title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sectionName });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function divider(title, sub, notes) {
  const s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: sectionName });
  s.addText(title, { placeholder: "title" });
  if (sub) s.addText(sub, { placeholder: "body" });
  if (notes) s.addNotes(notes);
  return s;
}
// place an image inside a box, keeping its aspect ratio; align: "c" centre, "l" left/top
function img(s, rel, x, y, w, h, opts = {}) {
  const [iw, ih] = DIMS[rel];
  const k = Math.min(w / iw, h / ih);
  const pw = iw * k, ph = ih * k;
  const px = opts.align === "l" ? x : x + (w - pw) / 2;
  const py = opts.valign === "t" ? y : y + (h - ph) / 2;
  s.addImage({ path: P(rel), x: px, y: py, w: pw, h: ph, altText: opts.alt || "", objectName: opts.name });
  return { x: px, y: py, w: pw, h: ph };
}
// a figure plate (thin border, white) like the HTML page
function plate(s, x, y, w, h, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background1 }, line: { color: HEX.line, width: 1 }, objectName: name });
}
// bullet list; items: string or [boldLead, rest]
function bullets(s, items, x, y, w, h, size = 16, opts = {}) {
  const runs = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    const base = { bullet: opts.bullet === false ? false : { indent: 16 }, paraSpaceAfter: opts.gap ?? 8, color: opts.color || C.text1 };
    if (Array.isArray(it)) {
      runs.push({ text: it[0] + " ", options: { ...base, bold: true } });
      runs.push({ text: it[1], options: { breakLine: !last, color: opts.color || C.text1 } });
    } else runs.push({ text: it, options: { ...base, breakLine: !last } });
  });
  s.addText(runs, { x, y, w, h, fontSize: size, valign: "top", margin: 0, isTextBox: true, objectName: opts.name });
}
function label(s, text, x, y, w, h, o = {}) {
  s.addText(text, { x, y, w, h, fontSize: o.size || 12, color: o.color || HEX.slate, bold: !!o.bold, italic: !!o.italic, align: o.align || "left",
    valign: o.valign || "top", margin: 0, isTextBox: true, fontFace: o.font, objectName: o.name });
}
// big-number callout card
function stat(s, x, y, w, h, value, caption, color = C.accent1, fill = C.background2) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { type: "none" } });
  s.addText(value, { x: x + 0.25, y: y + 0.15, w: w - 0.5, h: h * 0.52, fontSize: 40, bold: true, color, fontFace: "Cambria", valign: "bottom", margin: 0, isTextBox: true });
  s.addText(caption, { x: x + 0.25, y: y + h * 0.56 + 0.1, w: w - 0.5, h: h * 0.44 - 0.2, fontSize: 13, color: C.text1, valign: "top", margin: 0, isTextBox: true });
}
// numbered circle (the deck's motif)
function num(s, n, x, y, d = 0.5, color = C.accent1) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { type: "none" } });
  s.addText(String(n), { x, y, w: d, h: d, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function card(s, x, y, w, h, fill = C.background2) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { type: "none" } });
}
// shared chart styling (same label look on every chart)
const AX = { catAxisLabelColor: HEX.ink, valAxisLabelColor: HEX.slate, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 10, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: HEX.ink,
  valGridLine: { color: "E3E7EA", size: 0.75 }, catGridLine: { style: "none" }, titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: HEX.ink };
function hbar(s, labels, values, colors, x, y, w, h, title, extra = {}) {
  s.addChart(pres.charts.BAR, [{ name: "object IoU", labels, values }], {
    x, y, w, h, barDir: "bar", catAxisOrientation: "maxMin", chartColors: colors, showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.000", valAxisMinVal: extra.min ?? 0.5, valAxisMaxVal: extra.max ?? 0.72, valAxisMajorUnit: 0.05,
    valAxisLabelFormatCode: "0.00", showLegend: false, showTitle: !!title, title, barGapWidthPct: 45, ...AX, ...extra.opts,
  });
}
// simple table with header row
function table(s, header, rows, x, y, w, colW, o = {}) {
  const fs = o.size || 12;
  const head = header.map((t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: fs, valign: "middle" } }));
  const body = rows.map((r, i) => r.map((c, j) => {
    const hl = o.highlight && o.highlight(i, j);
    return { text: String(c), options: { fontSize: fs, color: C.text1, bold: !!hl, fill: { color: hl ? HEX.paleTeal : (i % 2 ? "F7F8F9" : "FFFFFF") }, valign: "middle", align: (j === 0 || (o.left || []).includes(j)) ? "left" : "center" } };
  }));
  s.addTable([head, ...body], { x, y, w, colW, border: { type: "solid", pt: 0.5, color: HEX.line }, rowH: o.rowH || 0.32, margin: 0.05, fontFace: "Calibri" });
}

// ================================================================= SLIDES
// 1 - title
section("Introduction");
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: sectionName });
  s.addText("BeltPulse-Net", { placeholder: "title" });
  s.addText([
    { text: "Finding foreign objects under glare and rolling on an iron-ore conveyor belt, using the belt's own motion", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 10 } },
    { text: "Final project report · IITH MTech · November 2026", options: { fontSize: 14, color: "9BA7B1" } },
  ], { placeholder: "body" });
  const r = img(s, "outputs/04_samples/glare_examples.jpg", 8.25, 1.2, 4.5, 5.1, { alt: "Six conveyor frames with strong glare spots and foreign objects among the ore" });
  label(s, "Dataset frames with simulated glare spots, as used in training and testing; foreign rocks and sticks among the ore", r.x, r.y + r.h + 0.12, r.w, 0.5, { size: 11, color: "9BA7B1", italic: true });
  s.addNotes("BeltPulse-Net detects impurities (wood, plastic, canga, jaspilito, tools) on an iron-ore conveyor from a single top camera. Glare and motion blur hide objects in single frames; the belt's known motion lets us combine several views of the same object.");
}

// 2 - executive summary
{
  const s = content("Summary: aligned views find what one frame misses",
    "Headline numbers are test-set object IoU on 5 videos the models never saw. 8-view results are one seed; 3-view avg/attention are 3 seeds.");
  bullets(s, [
    ["Goal achieved.", "Objects are detected under glare (simulated on all test videos, real on jaspilito_4), motion blur and rolling by combining belt-aligned views: every trained multi-view model beats the single-frame model."],
    ["Belt alignment matters.", "Shifting views by the measured belt motion adds +0.02 (3 views) to +0.04 (8 views) IoU, up to +0.12 on one video."],
    ["More views, better results.", "Going from 3 to 8 views lifts the best model to 0.682 IoU and recall under glare to 0.88."],
    ["Cross-frame attention ≈ averaging, but steadier.", "Accuracy matches feature averaging (0.626 vs 0.620 over 3 seeds) with 3× less spread between training runs, and it learns to trust clean views over glared ones."],
  ], M, 1.45, 6.9, 5.2, 16, { gap: 12 });
  stat(s, 7.9, 1.45, 4.8, 1.55, "0.682", "best test object IoU (8 views, feature averaging) vs 0.622 for one frame");
  stat(s, 7.9, 3.2, 4.8, 1.55, "0.88", "of object squares hidden under glare found (8 views) vs 0.81 single-frame");
  stat(s, 7.9, 4.95, 4.8, 1.55, "3×", "more stable across seeds: cross-frame attention sd 0.011 vs 0.033 for averaging", C.accent2, HEX.paleRust);
}

// 3 - problem
{
  const s = content("The problem: glare hides foreign objects on a moving belt",
    "Foreign objects damage crushers and contaminate product. A camera sees the belt from above, but a strong light reflection washes out parts of the frame, and objects can roll.");
  bullets(s, [
    ["What we look for:", "wood, plastic, tools and wrong ore types (canga, jaspilito) among iron ore."],
    ["Why one frame is not enough:", "a glare spot or motion blur can wipe out the object in that frame."],
    ["What we can use:", "the belt moves steadily along X, so the same object appears in many frames at a predictable position."],
    ["Complication:", "rocks roll and sticks rotate, so they don't move exactly with the belt."],
    ["Output wanted:", "for each frame, which squares of the image contain an impurity (a 22 × 40 grid)."],
  ], M, 1.45, 5.6, 5.2, 16, { gap: 12 });
  plate(s, 6.55, 1.45, 6.2, 3.55);
  img(s, "outputs/04_samples/glare_examples.jpg", 6.7, 1.6, 5.9, 2.3, { alt: "Glare examples" });
  label(s, "Glare is simulated by painting bright blobs on random views (as here); the test video jaspilito_4 also has real glare. Objects ride with the belt, so another view of the same object is often clean.", 6.75, 4.0, 5.8, 0.85, { size: 12, italic: true });
}

// 4 - data
{
  const s = content("Data: 19 belt videos, but no object locations",
    "Mendeley iron-ore conveyor dataset. The authors only flag which frames contain an object, not where it is, so location labels had to be created (next section).");
  stat(s, M, 1.45, 2.85, 1.45, "19", "videos, 1280 × 720, top camera");
  stat(s, M + 3.05, 1.45, 2.85, 1.45, "26,460", "frames in total");
  stat(s, M + 6.1, 1.45, 2.85, 1.45, "2,990", "frames flagged 'has object' by the authors");
  stat(s, M + 9.15, 1.45, 2.98, 1.45, "0", "object locations provided", C.accent2, HEX.paleRust);
  table(s, ["split", "videos", "3-view samples", "8-view samples"], [
    ["train", "11 videos (iron_ore 1/2/5/6, 7_wood 1/2, canga2, jaspilito 1/3, things 1/2)", "2,716", "2,644"],
    ["validation", "canga, jaspilito_2, iron_ore_3", "470", "458"],
    ["test (never seen)", "7_wood_3, canga3, jaspilito_4 (real glare + rolling), things_3, iron_ore_4 (ore only)", "508", "490"],
  ], M, 3.2, 12.13, [1.7, 6.83, 1.8, 1.8], { size: 13, rowH: 0.42, left: [1] });
  label(s, "Split by video, so test videos are completely unseen. The test set holds 263 samples with glare on the target frame (3 views).", M, 5.2, 12.1, 0.4, { size: 13 });
  img(s, "outputs/01_explore/anomaly_samples.jpg", M, 5.6, 12.13, 1.25, { alt: "Strip of frames with foreign objects" });
}

// ---------------------------------------------------------------- method
section("Method");
divider("How BeltPulse-Net works", "Belt motion → labels → aligned samples → trained models",
  "This section follows one frame from the camera to a per-square impurity score.");

// 6 - pipeline
{
  const s = content("The pipeline, end to end",
    "Colours as on the architecture page: teal = trained, amber = fixed physics or rule, grey = data or frozen pretrained network.");
  plate(s, M, 1.4, 12.13, 4.4);
  img(s, "reports/final_report/figs/pipeline.png", M + 0.15, 1.5, 11.83, 4.2, { alt: "Pipeline diagram" });
  const items = [["teal", "trained (learns from labels)", HEX.paleTeal, HEX.teal], ["amber", "fixed physics or rule, no learning", "F9EEDB", HEX.amber], ["grey", "data or frozen pretrained network", "E9EDF0", HEX.slate]];
  items.forEach(([_, t, f, l], i) => {
    const x = M + i * 4.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 6.05, w: 0.42, h: 0.28, rectRadius: 0.05, fill: { color: f }, line: { color: l, width: 1.25 } });
    label(s, t, x + 0.55, 6.05, 3.4, 0.3, { size: 13, color: HEX.ink, valign: "middle" });
  });
}

// 7 - belt motion
{
  const s = content("Step 1: measure how far the belt moves between frames",
    "Phase correlation between consecutive frames gives the shift in pixels. Vertical shift is ~0, so the belt moves along X only. This shift is what lets us line up views.");
  bullets(s, [
    ["Phase correlation", "between consecutive frames gives the belt shift in pixels."],
    ["Belt moves along X only:", "median vertical shift ≈ 0 px in all 19 videos."],
    ["Speed varies by video:", "from ~0 (stopped) to 48 px per frame."],
    ["Used twice:", "to copy labels between frames, and to line up the views inside the model (kinematic alignment)."],
  ], M, 1.45, 4.6, 5.0, 15, { gap: 12 });
  const vids = ["iron_ore", "iron_ore_2", "iron_ore_3", "iron_ore_4", "iron_ore_5", "iron_ore_6", "7_wood", "7_wood_2", "7_wood_3", "canga", "canga2", "canga3", "jaspilito", "jaspilito_2", "jaspilito_3", "jaspilito_4", "things", "things_2", "things_3"];
  const dx = [0.03, 9.47, 8.69, 12.20, 47.82, 0.01, 11.69, 17.95, 25.49, 24.65, 19.97, 32.81, 1.44, 10.63, 26.18, 39.85, 17.75, 17.28, 20.91];
  s.addChart(pres.charts.BAR, [{ name: "belt speed", labels: vids, values: dx }], {
    x: 5.5, y: 1.35, w: 7.25, h: 3.5, barDir: "col", chartColors: [HEX.amber], showValue: false, showLegend: false,
    showTitle: true, title: "Median belt speed per video (px per frame, along X)", valAxisLabelFormatCode: "0", barGapWidthPct: 40,
    ...AX, catAxisLabelFontSize: 9, catAxisLabelRotate: -45,
  });
  img(s, "outputs/02_belt_motion/dx_boardcamera_iron_ore_canga3.png", 5.5, 4.95, 7.25, 1.9, { alt: "Belt shift over time for canga3" });
}

// 8 - labelling
{
  const s = content("Step 2: LLM labels, corrected and reviewed by a human",
    "Location labels did not exist. An LLM drew boxes on keyframes; belt physics copied them to the frames in between; rolling objects got a two-keyframe correction and extra keyframes; a human reviewed and corrected the labels before training.");
  const steps = [
    ["LLM keyframe boxes", "An LLM drew boxes on 67 keyframes read from contact sheets."],
    ["Belt-physics propagation", "Boxes copied to neighbouring frames by the measured belt shift: 2,858 more frames."],
    ["Rolling correction", "Two-keyframe blending, plus 24 extra LLM keyframes where objects roll (flagged when > 60 px off)."],
    ["Human review", "All labels reviewed and corrected by hand before the final training runs."],
  ];
  steps.forEach(([h, t], i) => {
    const y = 1.45 + i * 1.2;
    num(s, i + 1, M, y + 0.05, 0.5, i === 3 ? C.accent2 : C.accent1);
    label(s, h, M + 0.7, y, 4.3, 0.35, { size: 16, bold: true, color: HEX.ink });
    label(s, t, M + 0.7, y + 0.38, 4.3, 0.75, { size: 13, color: HEX.ink });
  });
  plate(s, 5.75, 1.4, 7.0, 3.55);
  img(s, "outputs/03_labeling/correction_iron_ore_canga2.jpg", 5.85, 1.5, 6.8, 2.95, { alt: "Rolling correction: red plain physics, green corrected box" });
  label(s, "A rolling canga rock: red = plain physics copy, green = two-keyframe correction", 5.9, 4.5, 6.7, 0.35, { size: 11, italic: true });
  stat(s, 5.75, 5.15, 2.2, 1.6, "2,949", "labelled frames");
  stat(s, 8.15, 5.15, 2.2, 1.6, "3,355", "object boxes");
  stat(s, 10.55, 5.15, 2.2, 1.6, "14 → 7", "rolling intervals, before → after", C.accent2, HEX.paleRust);
}

// 9 - sample
{
  const s = content("Step 3: from a camera frame to one training sample",
    "Frames are halved to 640x352 = 40x22 squares of 16 px. A frozen ResNet-18 (to layer3) gives 256 numbers per square. A square is 'object' if at least 25% of it lies inside a box.");
  plate(s, M, 1.35, 12.13, 3.55);
  img(s, "reports/final_report/figs/sample.png", M + 0.15, 1.42, 11.83, 3.4, { alt: "Frame to sample diagram" });
  table(s, ["", "3-view variant", "8-view variant (T8w7)"], [
    ["views per sample", "t − k, t, t + k", "t − 3k … t + 4k"],
    ["k", "smallest gap where the belt moved ≥ 96 px", "same"],
    ["attention search window", "3 × 3 squares", "7 × 7 squares (finds objects that roll off the belt path)"],
    ["simulated glare", "one random view (train); view t in half of test", "each view with p = 0.3; view t in half of test"],
  ], M, 5.05, 12.13, [2.6, 4.1, 5.43], { size: 13, rowH: 0.34, left: [1, 2] });
}

// 10 - model
{
  const s = content("Step 4: the per-frame model (269k trainable parameters)",
    "Only the small head network is trained; ResNet is frozen. Parameter counts are for the 3-view model.");
  plate(s, M, 1.4, 12.13, 2.75);
  img(s, "reports/final_report/figs/model.png", M + 0.15, 1.5, 11.83, 2.55, { alt: "Model diagram" });
  s.addChart(pres.charts.BAR, [{ name: "parameters", labels: ["position embeddings", "belt context (spatial transformer)", "frame attention (q/k/v/o)", "feed-forward + norms", "bottleneck 256→96", "head"], values: [84480, 74784, 38400, 37536, 34176, 97] }], {
    x: M, y: 4.3, w: 7.2, h: 2.55, barDir: "bar", catAxisOrientation: "maxMin", chartColors: [HEX.teal], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "#,##0", valAxisHidden: true, showLegend: false, showTitle: true, title: "Trainable parameters by block", barGapWidthPct: 35, ...AX, valGridLine: { style: "none" },
  });
  bullets(s, [
    ["Frozen backbone:", "ResNet-18 features are computed once and cached."],
    ["Bottleneck:", "squeezes 256 → 96 numbers per square."],
    ["Fusion slot:", "the only part that changes between the six models (next slides)."],
    ["Output:", "one impurity score per square, 22 × 40."],
  ], 8.1, 4.4, 4.65, 2.5, 14, { gap: 8 });
}

// 11 - alignment
{
  const s = content("Kinematic alignment lines the object up in every view",
    "Each view is slid by the belt shift so a rock that sits at columns 6, 4 and 2 lines up at column 4. A search window around each square absorbs rolling and labelling error.");
  plate(s, M, 1.4, 12.13, 3.4);
  img(s, "reports/final_report/figs/alignment.png", M + 0.15, 1.5, 11.83, 3.2, { alt: "Alignment diagram" });
  const cols = [
    ["Fixed physics, no learning", "The belt shift from step 1 slides each view so the object lines up. No parameters to learn."],
    ["Search window for rolling", "The model looks at a 3 × 3 (or 7 × 7) patch of squares around the aligned position, so an object that rolled slightly is still found."],
    ["Every fusion model uses it", "Average, median, glare-skip and attention all receive aligned views; 'noalign' turns it off to measure its value."],
  ];
  cols.forEach(([h, t], i) => {
    const x = M + i * 4.13;
    card(s, x, 5.0, 3.87, 1.8);
    label(s, h, x + 0.2, 5.12, 3.5, 0.35, { size: 15, bold: true, color: HEX.ink });
    label(s, t, x + 0.2, 5.5, 3.5, 1.25, { size: 13, color: HEX.ink });
  });
}

// 12 - trust
{
  const s = content("Cross-frame attention: a learned trust weight per view",
    "Learned matrices W_q and W_k turn patches into queries and keys; softmax of the match scores gives weights that sum to 1. A glared view gets a low weight, so the blend relies on the clean views.");
  plate(s, M, 1.4, 12.13, 3.85);
  img(s, "reports/final_report/figs/trust.png", M + 0.15, 1.5, 11.83, 3.65, { alt: "Attention weight computation" });
  bullets(s, [
    ["Learned once:", "W_q, W_k, W_v (96 × 96 each, 4 heads of 24); everything else is computed fresh for every square."],
    ["Result per square:", "a weight for every view; for example 0.59 / 0.01 / 0.40 when the middle view is glared."],
    ["Readable:", "the weights can be shown as a per-view trust score, which plain averaging cannot provide."],
  ], M, 5.45, 12.13, 1.4, 14, { gap: 6 });
}

// 13 - models compared
{
  const s = content("Six trained models and voting baselines, one backbone",
    "Combine-then-decide: views are fused in the fusion slot before one decision. Decide-then-combine: the single-frame model decides per view and the aligned answers are voted.");
  plate(s, M, 1.35, 12.13, 2.85);
  img(s, "reports/final_report/figs/baselines.png", M + 0.1, 1.42, 11.93, 2.7, { alt: "Combine-then-decide versus decide-then-combine" });
  table(s, ["model", "how the views are combined"], [
    ["single", "target frame only (reference)"],
    ["avg", "aligned feature average"],
    ["median", "aligned feature median"],
    ["glareskip", "average, skipping washed-out views (rule)"],
    ["noalign", "cross-frame attention, no belt alignment"],
    ["beltpulse", "cross-frame attention with belt alignment"],
    ["votes", "single-frame answers aligned, then voted"],
  ], M, 4.35, 6.2, [1.4, 4.8], { size: 12, rowH: 0.3, left: [1] });
  bullets(s, [
    ["Same everything else:", "backbone, bottleneck, belt-context transformer, head, loss and schedule. Only the fusion step differs, so differences in results come from fusion."],
    ["Votes:", "OR, 2-of-N, majority and mean of the aligned single-frame answers; no extra training."],
    ["BeltPulse-Net is the whole system:", "belt motion, physics labels, aligned multi-view samples and the fusion models. Cross-frame attention is one of its fusion options."],
  ], 7.1, 4.4, 5.63, 2.5, 13, { gap: 8 });
}

// 14 - training
{
  const s = content("Training setup",
    "Best validation epoch is kept. Validation IoU peaks around epoch 5-12 and then drifts down slightly as training loss keeps falling: mild overfitting, handled by keeping the best checkpoint.");
  table(s, ["setting", "value"], [
    ["loss", "BCE (object squares × 5) + Dice"],
    ["optimiser", "AdamW, lr 5e-4, weight decay 1e-2, cosine schedule"],
    ["epochs / batch", "30 / 8 (3 views), 4 (8 views)"],
    ["checkpoint", "best validation object IoU"],
    ["hardware", "laptop RTX 3050, 4 GB"],
    ["time, 3 views", "94 min (6 models); +31 min per seed"],
    ["time, 8 views", "6.2 h (6 models)"],
  ], M, 1.45, 5.35, [1.55, 3.8], { size: 13, rowH: 0.44, left: [1] });
  label(s, "An unattended runner (scripts/run_full.py) keeps the laptop awake and logs every epoch. Attention models take ~2× longer per epoch at 8 views (392 keys per square).", M, 5.3, 5.35, 1.0, { size: 13, italic: true });
  plate(s, 6.2, 1.4, 6.53, 3.1);
  img(s, "outputs/06_train/full_T8w7/curves.png", 6.3, 1.5, 6.33, 2.9, { alt: "Training curves, 8-view run" });
  label(s, "8-view run: loss (left) and validation object IoU (right) per epoch, for all six models", 6.3, 4.55, 6.3, 0.35, { size: 11, italic: true });
  stat(s, 6.2, 5.1, 3.15, 1.7, "0.765", "best validation IoU (8 views, avg)");
  stat(s, 9.58, 5.1, 3.15, 1.7, "0.748", "best validation IoU (8 views, cross-frame attention)");
}

// ---------------------------------------------------------------- results
section("Results");
divider("Results", "Test set: 5 unseen videos · metric: object IoU over all test squares",
  "Object IoU: overlap between predicted and true impurity squares, pooled over all test samples. Recall under glare: share of object squares hidden by glare in the target frame that are still found.");

const TEAL = HEX.teal, RUST = HEX.rust, GREY = "9AA4AD", AMBER = HEX.amber;

function legendNote(s) {
  [["9AA4AD", "single frame and votes"], [HEX.amber, "other fusion models"], [HEX.teal, "feature average and cross-frame attention"]].forEach(([c, t], i) => {
    const x = M + [0, 2.45, 4.75][i];
    s.addShape(pres.shapes.RECTANGLE, { x, y: 6.6, w: 0.22, h: 0.18, fill: { color: c }, line: { type: "none" } });
    label(s, t, x + 0.3, 6.55, 3.2, 0.3, { size: 11, color: HEX.ink, valign: "middle" });
  });
}

// 16 - 3-view benchmark
{
  const s = content("3 views: every multi-view model beats the single frame",
    "Seed 0 run. Feature averaging is best on this seed; the seed study later shows avg and attention are equivalent on average.");
  const labels = ["single frame", "single + OR vote", "single + majority vote", "noalign (attention, no alignment)", "single + mean vote", "median", "glareskip", "beltpulse (cross-frame attention)", "avg (feature average)"];
  const vals = [0.583, 0.585, 0.615, 0.615, 0.620, 0.621, 0.630, 0.635, 0.658];
  hbar(s, labels, vals, [GREY, GREY, GREY, AMBER, GREY, AMBER, AMBER, TEAL, TEAL], M, 1.4, 7.6, 5.05, "Test object IoU, 3 views (seed 0)", { min: 0.55, max: 0.68 });
  legendNote(s);
  stat(s, 8.55, 1.45, 4.18, 1.6, "+0.075", "IoU from feature averaging over the single frame (0.658 vs 0.583)");
  stat(s, 8.55, 3.25, 4.18, 1.6, "0.85", "recall under glare for avg, vs 0.64 for the single frame");
  bullets(s, [
    ["Combine-then-decide wins:", "fusing features beats voting on single-frame answers (best vote 0.620)."],
    ["OR vote", "finds the most hidden squares (0.91) but doubles false alarms."],
  ], 8.55, 5.1, 4.18, 1.8, 13, { gap: 6 });
}

// 17 - 8-view benchmark
{
  const s = content("8 views: the gains grow, and alignment matters more",
    "One seed. 2-of-N voting only exists for more than 3 views.");
  const labels = ["single + OR vote", "single frame", "noalign (attention, no alignment)", "glareskip", "single + 2-of-N vote", "median", "single + majority vote", "beltpulse (cross-frame attention)", "single + mean vote", "avg (feature average)"];
  const vals = [0.579, 0.622, 0.626, 0.640, 0.643, 0.647, 0.666, 0.666, 0.671, 0.682];
  hbar(s, labels, vals, [GREY, GREY, AMBER, AMBER, GREY, AMBER, GREY, TEAL, GREY, TEAL], M, 1.4, 7.6, 5.05, "Test object IoU, 8 views + 7×7 window", { min: 0.55, max: 0.70 });
  legendNote(s);
  stat(s, 8.55, 1.45, 4.18, 1.6, "0.682", "best test IoU (avg), AP 0.892; cross-frame attention 0.666");
  stat(s, 8.55, 3.25, 4.18, 1.6, "+0.040", "IoU from belt alignment (attention 0.666 vs noalign 0.626)");
  bullets(s, [
    ["Voting catches up:", "with 8 views one glared view is outvoted 7 to 1 (mean vote 0.671)."],
    ["Fixed rules lag:", "median and glare-skip trail the learned and averaged fusion."],
  ], 8.55, 5.1, 4.18, 1.8, 13, { gap: 6 });
}

// 18 - 3 vs 8 views
{
  const s = content("More views help every multi-view model",
    "3-view numbers for avg and beltpulse are 3-seed means; the rest are seed 0. Test sets differ slightly (508 vs 490 samples) because 8-view samples need more frames, so compare trends.");
  const labels = ["single frame", "single + mean vote", "median", "glareskip", "noalign", "cross-frame attention", "feature average"];
  s.addChart(pres.charts.BAR, [
    { name: "3 views", labels, values: [0.583, 0.620, 0.621, 0.630, 0.615, 0.626, 0.620] },
    { name: "8 views", labels, values: [0.622, 0.671, 0.647, 0.640, 0.626, 0.666, 0.682] },
  ], {
    x: M, y: 1.4, w: 8.2, h: 5.4, barDir: "col", barGrouping: "clustered", chartColors: ["B9C2C9", HEX.teal], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.000", valAxisMinVal: 0.5, valAxisMaxVal: 0.72, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00",
    showLegend: true, legendPos: "t", legendFontFace: "+mn-lt", legendFontSize: 12, showTitle: true, title: "Test object IoU, 3 views vs 8 views", barGapWidthPct: 60, ...AX, catAxisLabelFontSize: 11, dataLabelFontSize: 10,
  });
  bullets(s, [
    ["Feature average:", "0.620 → 0.682."],
    ["Cross-frame attention:", "0.626 → 0.666."],
    ["Mean vote:", "0.620 → 0.671, the biggest jump; more views make voting robust."],
    ["Alignment gap widens:", "attention minus noalign goes from +0.02 to +0.04, because views further apart in time disagree more about where the object is."],
    ["Single frame:", "uses only one view; its change reflects the slightly different test set."],
  ], 9.1, 1.5, 3.65, 5.3, 13, { gap: 9 });
}

// 19 - seeds
{
  const s = content("Cross-frame attention ≈ averaging, and 3× more stable",
    "Same data, same settings, three random starts. Averaging swings by 0.06 between seeds; cross-frame attention stays within 0.02. On average they are equal.");
  table(s, ["test object IoU (3 views)", "seed 0", "seed 1", "seed 2", "mean ± sd"], [
    ["feature average (avg)", "0.658", "0.599", "0.604", "0.620 ± 0.033"],
    ["cross-frame attention (beltpulse)", "0.635", "0.629", "0.614", "0.626 ± 0.011"],
  ], M, 1.45, 7.6, [2.9, 1.0, 1.0, 1.0, 1.7], { size: 14, rowH: 0.48, highlight: (i, j) => i === 1 && j === 4 });
  s.addChart(pres.charts.BAR, [
    { name: "feature average", labels: ["seed 0", "seed 1", "seed 2", "mean"], values: [0.658, 0.599, 0.604, 0.620] },
    { name: "cross-frame attention", labels: ["seed 0", "seed 1", "seed 2", "mean"], values: [0.635, 0.629, 0.614, 0.626] },
  ], {
    x: M, y: 3.05, w: 7.6, h: 3.8, barDir: "col", barGrouping: "clustered", chartColors: ["B9C2C9", HEX.teal], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.000", valAxisMinVal: 0.55, valAxisMaxVal: 0.68, valAxisMajorUnit: 0.02, valAxisLabelFormatCode: "0.00", barGapWidthPct: 60,
    showLegend: true, legendPos: "t", legendFontFace: "+mn-lt", legendFontSize: 12, showTitle: true, title: "Test object IoU by training seed (3 views)", ...AX,
  });
  stat(s, 8.55, 1.45, 4.18, 1.6, "0.011", "seed-to-seed sd for cross-frame attention, vs 0.033 for averaging");
  bullets(s, [
    ["Equivalent accuracy:", "0.626 vs 0.620 mean IoU; the gap is far inside the spread."],
    ["More reliable to train:", "any single run lands close to the mean, so one training run can be trusted."],
    ["Fewer false alarms", "on most test videos in seeds 1 and 2 (e.g. things_3: ~2,900 vs 5,400–6,300 squares)."],
    ["Trade-off:", "averaging keeps slightly higher recall under glare (0.85–0.87 vs 0.77–0.83)."],
  ], 8.55, 3.3, 4.18, 3.5, 13, { gap: 9 });
}

// 20 - glare examples
{
  const s = content("Under glare: the glare spot is ignored, the object is kept",
    "Columns: three input views, truth (green), single-frame prediction (red), BeltPulse cross-frame attention prediction (red). Single-frame mistakes the glare blob for an object; the multi-view model does not.");
  const strips = ["outputs/07_eval/full/examples_02.jpg", "outputs/07_eval/full/examples_05.jpg", "outputs/07_eval/full_T8w7/examples_03.jpg"];
  strips.forEach((f, i) => img(s, f, M, 1.45 + i * 1.3, 12.13, 1.2, { alt: "Example predictions under glare" }));
  const heads = ["view t − k", "view t (glare)", "view t + k", "truth", "single frame", "multi-view (attention)"];
  heads.forEach((h, i) => label(s, h, M + i * 2.02, 5.35, 2.0, 0.3, { size: 12, bold: true, color: HEX.ink, align: "center" }));
  stat(s, M, 5.75, 3.9, 1.1, "0.64 → 0.85", "recall under glare, 3 views (single → avg)");
  stat(s, M + 4.12, 5.75, 3.9, 1.1, "0.81 → 0.88", "recall under glare, 8 views (single → avg)");
  stat(s, M + 8.24, 5.75, 3.89, 1.1, "0.61 → 0.68", "IoU on glared target frames, 8 views");
}

// 21 - real glare + rolling + alignment
{
  const s = content("Real glare, motion blur and rolling: the hardest test video",
    "jaspilito_4 is the only test video with real (not synthetic) glare, strong motion blur and two rolling intervals. canga3 shows what alignment buys.");
  const labels = ["single frame", "single + mean vote", "noalign", "cross-frame attention", "median", "feature average"];
  s.addChart(pres.charts.BAR, [
    { name: "3 views", labels, values: [0.628, 0.678, 0.704, 0.699, 0.735, 0.725] },
    { name: "8 views", labels, values: [0.705, 0.756, 0.678, 0.751, 0.730, 0.747] },
  ], {
    x: M, y: 1.4, w: 7.4, h: 3.6, barDir: "col", barGrouping: "clustered", chartColors: ["B9C2C9", HEX.teal], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.00", valAxisMinVal: 0.55, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00", showLegend: true, legendPos: "t",
    legendFontFace: "+mn-lt", legendFontSize: 11, showTitle: true, title: "jaspilito_4 (real glare + rolling): object IoU", barGapWidthPct: 60, ...AX, catAxisLabelFontSize: 10, dataLabelFontSize: 9,
  });
  s.addChart(pres.charts.BAR, [
    { name: "IoU", labels: ["noalign, 3 views", "attention, 3 views", "noalign, 8 views", "attention, 8 views"], values: [0.612, 0.730, 0.607, 0.726] },
  ], {
    x: M, y: 5.05, w: 7.4, h: 1.85, barDir: "bar", catAxisOrientation: "maxMin", chartColors: [GREY, TEAL, GREY, TEAL], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.000", valAxisMinVal: 0.5, valAxisMaxVal: 0.78, valAxisHidden: true, valGridLine: { style: "none" }, showLegend: false, showTitle: true, title: "canga3: belt alignment on vs off", barGapWidthPct: 30,
    ...AX, valGridLine: { style: "none" }, catAxisLabelFontSize: 11, dataLabelFontSize: 10,
  });
  stat(s, 8.3, 1.45, 4.43, 1.55, "+0.11", "IoU on real glare + rolling (3 views): single 0.628 → median 0.735, feature average 0.725");
  bullets(s, [
    ["Detected under real glare and rolling:", "with 8 views the aligned fusion models reach 0.73–0.75 IoU and the mean vote 0.76."],
    ["Wide window helps rolling objects:", "the 7 × 7 search window lets attention follow an object that drifts off the belt path."],
    ["Alignment is decisive on canga3:", "+0.12 IoU and about 1,500 vs 3,800 false-alarm squares."],
  ], 8.3, 3.2, 4.43, 3.6, 13, { gap: 10 });
}

// 22 - what attention learned
{
  const s = content("What cross-frame attention learned: distrust glared views",
    "Average attention weight a view receives on object squares. Uniform weight would be 1/3 (3 views) or 1/8 (8 views). Glared views get clearly less than uniform.");
  s.addChart(pres.charts.BAR, [
    { name: "glared view", labels: ["3 views, seed 0", "3 views, seed 1", "8 views"], values: [0.160, 0.126, 0.088] },
    { name: "clean view", labels: ["3 views, seed 0", "3 views, seed 1", "8 views"], values: [0.340, 0.342, 0.128] },
  ], {
    x: M, y: 1.4, w: 6.0, h: 4.0, barDir: "col", barGrouping: "clustered", chartColors: [HEX.rust, HEX.teal], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.000", valAxisMinVal: 0, valAxisMaxVal: 0.4, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0", showLegend: true, legendPos: "t",
    legendFontFace: "+mn-lt", legendFontSize: 12, showTitle: true, title: "Average attention weight per view (object squares)", barGapWidthPct: 70, ...AX,
  });
  bullets(s, [
    ["3 views:", "a glared view gets about half the weight of a clean one (0.16 vs 0.34)."],
    ["8 views:", "glared 0.088 vs clean 0.128; with many clean views available, the model needs to single out glare less."],
    ["Learned, not hand-coded:", "no glare label was given; the model found this from the detection task alone."],
  ], M, 5.55, 6.0, 1.4, 13, { gap: 6 });
  plate(s, 6.95, 1.4, 5.78, 5.45);
  img(s, "outputs/08_attention/full_T8w7/square_00_A_glare.jpg", 7.05, 1.5, 5.58, 4.75, { alt: "Attention map for one square across eight views" });
  label(s, "One square under glare, 8 views: yellow = where attention looks inside each 7 × 7 window; cyan dot = its centre of mass", 7.1, 6.3, 5.5, 0.5, { size: 11, italic: true });
}

// 23 - per video
{
  const s = content("Per test video (8 views): consistent gains, low false alarms",
    "Object IoU / false-alarm squares. iron_ore_4 is ore only, so any detection there is a false alarm; about 162,800 squares were scored on it.");
  table(s, ["model", "7_wood_3", "canga3", "jaspilito_4", "things_3", "iron_ore_4 (ore only): false alarms"], [
    ["single frame", "0.555", "0.679", "0.705", "0.654", "220"],
    ["single + mean vote", "0.587", "0.759", "0.756", "0.702", "53"],
    ["median", "0.564", "0.717", "0.730", "0.685", "29"],
    ["glareskip", "0.581", "0.723", "0.538", "0.702", "21"],
    ["noalign", "0.614", "0.607", "0.678", "0.638", "0"],
    ["cross-frame attention", "0.601", "0.726", "0.751", "0.707", "518"],
    ["feature average", "0.595", "0.769", "0.747", "0.731", "25"],
  ], M, 1.45, 12.13, [3.0, 1.55, 1.55, 1.55, 1.55, 2.93], { size: 14, rowH: 0.44,
    highlight: (i, j) => (j === 1 && i === 4) || (j === 2 && i === 6) || (j === 3 && i === 1) || (j === 4 && i === 6) || (j === 5 && i === 4) });
  bullets(s, [
    ["Aligned fusion beats the single frame on every object video", "(glareskip excepted on jaspilito_4); without alignment, noalign falls below it on three videos."],
    ["False alarms stay small:", "even the highest (518 squares) is 0.3% of the ore-only video."],
    ["glareskip's fixed whiteness rule", "fails on real glare (0.538): it drops views that still show the object. Learned fusion avoids this."],
  ], M, 5.2, 12.13, 1.6, 14, { gap: 8 });
}

// ---------------------------------------------------------------- conclusions
section("Conclusions");
{
  const s = content("Limitations and honest caveats",
    "These are the points to keep in mind when quoting the numbers.");
  const items = [
    ["Labels", "LLM boxes are about ±30–50 px; human review corrected them, but square-level IoU is still limited by box precision."],
    ["Seeds", "3 seeds for avg and attention at 3 views; the 8-view run is one seed, so gaps under ~0.03 are not significant."],
    ["Counting", "Counting objects by connected blobs is weak (1.63 regions per object, exact count 55%); a per-object tracker is the remedy."],
    ["Validation vs test", "Validation IoU (~0.75) is above test (~0.65): test videos are harder; validation has only 3 videos."],
    ["Rolling", "Rolling is covered by the test video and the wide window, but there is no separate rolling-only score yet."],
    ["Simulated glare", "Most glare is painted on; jaspilito_4 is the only real-glare test video (multi-view still +0.05 to +0.11 IoU there)."],
  ];
  items.forEach(([h, t], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = M + col * 6.18, y = 1.45 + row * 1.8;
    card(s, x, y, 5.95, 1.6);
    num(s, i + 1, x + 0.2, y + 0.2, 0.45, C.accent2);
    label(s, h, x + 0.8, y + 0.2, 4.95, 0.4, { size: 16, bold: true, color: HEX.ink });
    label(s, t, x + 0.8, y + 0.62, 4.95, 0.95, { size: 13, color: HEX.ink });
  });
}

{
  const s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: sectionName });
  s.addText("Conclusions: the goal is met", { placeholder: "title" });
  s.addNotes("The project goal, detecting foreign objects under glare and rolling using the belt's motion, is met. Cross-frame attention is not a failure: it performs like averaging and is the most stable choice across training runs, and it gives interpretable per-view trust weights.");
  const items = [
    ["0.682 vs 0.622", "Objects found under glare, blur and rolling", "Every trained multi-view model beats the single frame. Best: 8 views, feature average; recall under glare 0.88 vs 0.81."],
    ["+0.04", "Belt alignment works", "The physics shift adds +0.02 (3 views) to +0.04 (8 views) IoU, and +0.12 on canga3, at no training cost."],
    ["3 → 8 views", "More views help every multi-view model", "Feature average 0.620 → 0.682, cross-frame attention 0.626 → 0.666, mean vote 0.620 → 0.671."],
    ["≈ and 3×", "Cross-frame attention: same accuracy, steadier", "Matches averaging (0.626 vs 0.620) with 3× smaller seed spread, fewer false alarms on most videos, and a readable trust weight per view."],
  ];
  items.forEach(([big, head, text], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = M + col * 6.18, y = 1.5 + row * 2.68;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 5.95, h: 2.45, rectRadius: 0.1, fill: { color: "2F3943" }, line: { type: "none" }, objectName: `takeaway ${i + 1}` });
    s.addText(big, { x: x + 0.3, y: y + 0.2, w: 5.35, h: 0.8, fontSize: 34, bold: true, color: "5CC8BD", fontFace: "Cambria", valign: "middle", margin: 0, isTextBox: true });
    s.addText(head, { x: x + 0.3, y: y + 1.05, w: 5.35, h: 0.4, fontSize: 17, bold: true, color: "FFFFFF", valign: "top", margin: 0, isTextBox: true });
    s.addText(text, { x: x + 0.3, y: y + 1.45, w: 5.35, h: 0.9, fontSize: 14, color: "C9D2D8", valign: "top", margin: 0, isTextBox: true });
  });
}

{
  const s = content("Next steps",
    "Ordered by how directly they strengthen the report's claims.");
  const items = [
    ["Per-object decisions", "Track boxes across frames with belt physics and decide once per object: fixes counting and gives one alarm per impurity (step 10)."],
    ["Rolling score (E15)", "Score the flagged rolling intervals separately, to quantify the benefit of the wide window."],
    ["More 8-view seeds", "Two more seeds for averaging and attention at 8 views, to confirm the stability result there."],
    ["Stronger backbone (E14)", "Swap frozen ResNet-18 for DINOv2 features; the fusion models stay the same."],
    ["Lighter model (E16)", "Test whether the belt-context transformer is needed; it holds a third of the parameters."],
    ["Operating point", "Tune the decision threshold per use case: high recall (OR-like) vs few false alarms."],
  ];
  items.forEach(([h, t], i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = M + col * 4.13, y = 1.5 + row * 2.6;
    card(s, x, y, 3.87, 2.3);
    num(s, i + 1, x + 0.2, y + 0.2, 0.5);
    label(s, h, x + 0.85, y + 0.25, 2.85, 0.45, { size: 16, bold: true, color: HEX.ink, valign: "middle" });
    label(s, t, x + 0.2, y + 0.9, 3.47, 1.45, { size: 13, color: HEX.ink });
  });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
})();

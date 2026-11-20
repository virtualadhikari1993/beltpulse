"""Render each SVG figure of docs/pipeline_architecture.html to a PNG (light theme) with headless Chrome."""
import os, re, subprocess
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "figs")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
html = open(os.path.join(ROOT, "docs", "pipeline_architecture.html"), encoding="utf-8").read()
style = re.search(r"<style>.*?</style>", html, re.S).group(0)
fonts = re.search(r'<link rel="stylesheet"[^>]*>', html).group(0)
names = ["pipeline", "sample", "model", "alignment", "trust", "baselines", "per_object"]
svgs = re.findall(r"<svg viewBox=.*?</svg>", html, re.S)
W = 1120
for name, svg in zip(names, svgs):
    vw, vh = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
    h = int(W * vh / vw) + 2
    page = f"""<!doctype html><html data-theme="light"><head><meta charset="utf-8">{fonts}{style}
<style>html,body{{margin:0;background:#fff}} svg{{display:block;width:{W}px;height:auto}}</style></head><body>{svg}</body></html>"""
    src = os.path.join(OUT, f"_{name}.html")
    open(src, "w", encoding="utf-8").write(page)
    png = os.path.join(OUT, f"{name}.png")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    f"--window-size={W},{h}", "--virtual-time-budget=4000", f"--screenshot={png}", "file:///" + src.replace("\\", "/")],
                   check=True, capture_output=True)
    os.remove(src)
    print(name, W, h)

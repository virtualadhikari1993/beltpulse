"""Run the full pipeline unattended, with a log you can watch, and keep Windows awake while it runs.

Usage (from the project root, in a normal PowerShell window, not inside VS Code):
  .venv\\Scripts\\python scripts\\run_full.py                  # 3 views:  05 -> 06 -> 07 -> 07b -> 08   (~1 h)
  .venv\\Scripts\\python scripts\\run_full.py --variant T8w7   # 8 views:  05 -> 06 -> 07 -> 08          (~4 h)
  .venv\\Scripts\\python scripts\\run_full.py --from 07        # skip steps before 07 (e.g. after a crash)

Watch progress in a second window:
  Get-Content outputs\\logs\\full_base.log -Wait -Tail 20
  (outputs\\logs\\full_base_status.txt always holds the current step in one line)

While this script runs, Windows is told not to go to sleep (the same request a video player makes).
Locking the screen is fine. Closing the lid still sleeps the laptop unless the lid action is set to
"Do nothing" (see docs/full_run_plan.md). The request ends by itself when the script exits.
"""
import argparse
import ctypes
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="base", choices=["base", "T8w7"])
ap.add_argument("--from", dest="start", default="05", help="first step to run: 05, 06, 07, 07b, 08")
ap.add_argument("--seed", type=int, default=0, help="extra training seed: runs only 06 + 07, models in <run>_sN")
ap.add_argument("--modes", nargs="+", default=None, help="models to train in step 06 (default: all 6)")
args = ap.parse_args()

v = [] if args.variant == "base" else ["--variant", args.variant]
sd = ["--seed", str(args.seed)] if args.seed else []
md = ["--modes"] + args.modes if args.modes else []
steps = [("05", ["src/05_extract_features.py"] + v), ("06", ["src/06_train.py"] + v + sd + md), ("07", ["src/07_evaluate.py"] + v + sd)]
if not args.seed:
    if args.variant == "base":
        steps.append(("07b", ["src/07b_multi_object.py"]))
    steps.append(("08", ["src/08_explain_attention.py"] + v))
names = [s for s, _ in steps]
assert args.start in names, f"--from must be one of {names}"
steps = steps[names.index(args.start):]

logdir = os.path.join(ROOT, "outputs", "logs")
os.makedirs(logdir, exist_ok=True)
run = f"full_{args.variant}" + (f"_s{args.seed}" if args.seed else "")
log_path = os.path.join(logdir, f"{run}.log")
status_path = os.path.join(logdir, f"{run}_status.txt")


def status(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    open(status_path, "w", encoding="utf-8").write(line + "\n")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== {line}\n")


# keep the system (not the screen) awake: ES_CONTINUOUS | ES_SYSTEM_REQUIRED
if os.name == "nt":
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
t_all = time.time()
open(log_path, "a").close()
for i, (name, cmd) in enumerate(steps, 1):
    status(f"step {name} ({i}/{len(steps)}) running: {' '.join(cmd)}")
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as f:
        code = subprocess.call([sys.executable] + cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    if code != 0:
        status(f"FAILED at step {name} (exit code {code}) after {(time.time() - t0) / 60:.1f} min. "
               f"Fix it, then rerun with --from {name}")
        sys.exit(code)
    status(f"step {name} done in {(time.time() - t0) / 60:.1f} min")
status(f"ALL DONE ({run}) in {(time.time() - t_all) / 60:.0f} min. Results: outputs/07_eval/"
       f"{('full' if args.variant == 'base' else 'full_' + args.variant) + (f'_s{args.seed}' if args.seed else '')}/metrics.md")

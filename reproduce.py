"""Runs the whole pipeline in order and saves each console log to logs/.
Usage (from the repository root):
    python reproduce.py                 # everything
    python reproduce.py --from q2       # resume from a step
    python reproduce.py --only figures  # one step
    python reproduce.py --skip q3       # everything except the slow step
    python reproduce.py --dry-run       # just print the commands
"""
import argparse
import os
import subprocess
import sys
import time

try:                                   # always run from the repo folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except NameError:                      # pasted into a notebook cell: __file__ does not exist
    pass

STEPS = [("split", "data_split.py"),
         ("q1", "q1_tree_bagging.py"),
         ("q1c", "q1c_oob_disagreement_margin.py"),
         ("q1d", "q1d_compare_masked.py"),
         ("q2toy", "q2_toy_samme.py"),
         ("q2", "q2_boosting.py"),
         ("q3", "q3_bias_variance.py"),       # slowest step; resumable
         ("figures", "make_figures.py")]
NAMES = [n for n, _ in STEPS]

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", choices=NAMES)
ap.add_argument("--only", choices=NAMES)
ap.add_argument("--skip", nargs="*", default=[], choices=NAMES)
ap.add_argument("--dry-run", action="store_true")
args, _ignored = ap.parse_known_args()   # ignore extra args such as Jupyter's -f kernel.json

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

todo = STEPS
if args.only:
    todo = [s for s in STEPS if s[0] == args.only]
elif args.start:
    todo = STEPS[NAMES.index(args.start):]
todo = [s for s in todo if s[0] not in args.skip]

for name, script in todo:
    cmd = [sys.executable, script]
    log = f"logs/{name}.log"
    print(f"\n=== {name}: python {script}  (log: {log})", flush=True)
    if args.dry_run:
        continue
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:                 # print to screen and save to the log
            sys.stdout.write(line); f.write(line)
        p.wait()
    print(f"--- {name} finished in {time.time() - t0:.0f}s (exit code {p.returncode})", flush=True)
    if p.returncode != 0:
        sys.exit(f"Step '{name}' failed. Fix the error, then rerun with:  python reproduce.py --from {name}")
print("\nAll requested steps finished.")

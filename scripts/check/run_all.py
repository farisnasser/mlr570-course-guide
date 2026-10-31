"""Run every part's check script; exit non-zero if any fails.  usage: python scripts/check/run_all.py"""
import glob
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
bad = []
for path in sorted(glob.glob(os.path.join(here, "[0-9]_[0-9]*.py"))):
    r = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    print(f"{os.path.basename(path):10s} {last}")
    if r.returncode:
        bad.append(path)
        print(r.stdout + r.stderr)
sys.exit(1 if bad else 0)

"""DEPRECATED. Use `run.py train`. Kept so in-flight SLURM jobs do not fail."""
import os
import sys

print("DEPRECATED: train.py is deprecated; use 'run.py train' instead.", file=sys.stderr)
run_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")
os.execv(sys.executable, [sys.executable, run_py, "train", *sys.argv[1:]])

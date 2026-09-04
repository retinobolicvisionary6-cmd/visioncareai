"""
Root entry point for evaluation.
Delegates to src/evaluate.py.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint models/checkpoints/best_model.pth
"""
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "src" / "evaluate.py"
    subprocess.run([sys.executable, str(script)] + sys.argv[1:])

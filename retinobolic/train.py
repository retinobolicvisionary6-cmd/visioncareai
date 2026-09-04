"""
Root entry point for training.
Delegates to src/train.py — see that file for full CLI options.

Usage:
    python train.py
    python train.py --model resnet50 --epochs 25 --batch_size 8
"""
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "src" / "train.py"
    subprocess.run([sys.executable, str(script)] + sys.argv[1:])

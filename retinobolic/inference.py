"""
Root entry point for single-image inference.
Delegates to src/inference.py.

Usage:
    python inference.py path/to/retina_image.jpg
    python inference.py path/to/image.jpg --no_gradcam
"""
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "src" / "inference.py"
    subprocess.run([sys.executable, str(script)] + sys.argv[1:])

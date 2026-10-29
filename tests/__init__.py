"""Run from the repo root: python -m unittest (uses src/ directly when the package is not installed)."""
import os
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent/'src')
if SRC not in sys.path:sys.path.insert(0, SRC)
# Subprocess tests (python -m emgprobe ...) find the package the same way.
os.environ['PYTHONPATH'] = os.pathsep.join(p for p in [SRC, os.environ.get('PYTHONPATH')] if p)

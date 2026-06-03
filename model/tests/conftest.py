"""Make the model package importable when running ``pytest`` from anywhere."""
import sys
from pathlib import Path

# model/ (parent of tests/) holds vsa.py, bep.py, layers.py, model.py, data.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

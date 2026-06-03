"""Make the model package importable when running ``pytest`` from anywhere."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# model/ holds vsa.py, bold.py, layers.py, model.py, data.py; tests/ holds
# local test helpers.
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

"""BGPT-1: a fully 1-bit decoder-only language model.

See `bgpt1/model.py` for the model class and `bgpt1/train.py` for the
training loop. Both build entirely on brute.nn / brute.optim — no FP
weights, gradients, or optimizer state in the forward / backward path.
"""

from examples.bgpt1.model import BGPT1, BGPT1Config

__all__ = ["BGPT1", "BGPT1Config"]

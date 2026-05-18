"""Optimizers for bit1 (1-bit) parameters.

The standard torch.optim optimizers don't make sense for binary weights —
there are no gradients to apply with a learning rate. Instead we provide:

- :class:`FlipRule`: BGPT-1 streaming flip-rule optimizer with per-layer
  auto-regularization (BOLD-style ``β`` from layer flip rate) and per-row
  confidence counters. Maintains tiny stateful counters; updates weights
  by stochastic XOR against a flip mask computed from per-batch votes.

These optimizers operate directly on packed bit1 buffers, expecting a
caller-driven backward pass (the modules in ``brute.nn`` produce gates and
inputs on demand; the caller assembles them into a tape and walks
backward through the network).
"""

# Surface torch.optim names for the FP/integer parameters too — the LM head
# bias and other small non-binary state may still want adam/sgd updates,
# though BGPT-1 keeps those updates in custom integer accumulators (see
# `brute.nn.binary_lm_head`).
from torch.optim import *  # noqa: F401, F403

from brute.optim.flip_rule import FlipRule, LayerState, propagate_error

__all__ = ["FlipRule", "LayerState", "propagate_error"]

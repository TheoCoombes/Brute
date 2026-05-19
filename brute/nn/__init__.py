from torch.nn import *

# BOLD (Boolean Logic Deep Learning, NeurIPS 2024) — general-purpose Boolean
# layers that can be mixed with any FP torch module. Parameters carrying the
# ``_bold_boolean`` tag are picked up by ``brute.optim.BooleanOptimizer``.
from brute.nn.bit_linear import BitLinear
from brute.nn.bit_conv import BitConv2d
from brute.nn.bit_activation import BitActivation

from torch.nn import *

# Low-level binary linear (kept for backwards compat)
from brute.nn.linear import BruteLinear, BinaryLinear

# BGPT-1 building blocks. Each is generic enough to compose into any
# binary transformer; the BGPT-1-specific glue lives in
# examples/bgpt1/model.py.
from brute.nn.binary_norm import BitBalancedNorm, bit_balance
from brute.nn.binary_embedding import BinaryEmbedding
from brute.nn.binary_positions import BinaryPositionEmbedding, make_binary_positions
from brute.nn.binary_attention import BinaryAttention, alibi_slopes
from brute.nn.binary_ffn import BinaryFFN
from brute.nn.binary_block import BinaryTransformerBlock
from brute.nn.binary_lm_head import (
    BinaryLMHead,
    signed_int_error,
    lm_head_bias_step,
)

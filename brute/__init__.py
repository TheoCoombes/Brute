import brute._C  # triggers TORCH_LIBRARY static-init registration

from brute.tensor import (
    Tensor,
    bit1,
    float32, float16, bfloat16,
    int8, int32, int64,
)
from brute.module import Module
import brute.layers as layers

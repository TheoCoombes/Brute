from torch import *

try:
    from . import _cbrute
except ImportError:
    try:
        import importlib.util
        spec = importlib.util.find_spec('_cbrute')
        if spec and spec.origin:
            _cbrute = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_cbrute)
    except Exception:
        raise ImportError(
            "brute requires the compiled _cbrute extension. "
            "Install with: pip install --no-build-isolation -ve ."
        )

# bit1-aware tensors + dtype
from brute.tensor import Tensor, bit1

# bit1-aware factory functions
from brute.functional import (
    zeros, ones, empty, full,
    tensor, as_tensor, from_numpy,
    rand, randn, randint,
    rand_like, randn_like, zeros_like, ones_like, full_like, empty_like,
    arange, linspace, eye,
)

# Fast bit1 ops take final priority — they overwrite any torch / functional
# versions of the same name (bitwise_xor, bitwise_and, eq, ne, matmul, …).
from brute.fast import *

try:
    from brute import nn  # noqa: F401
except (ImportError, FileNotFoundError):
    nn = None
try:
    from brute import optim  # noqa: F401
except (ImportError, FileNotFoundError):
    optim = None
from brute import cuda_graphs
from brute import streams
from brute import fast

parallel_streams = streams.parallel_streams

# Re-export torch submodules not covered by `from torch import *`.
# These are accessible as brute.linalg, brute.cuda, etc. brute.optim was
# imported above; we don't shadow it here.
from torch import (  # noqa: F401, E402
    linalg,
    fft,
    special,
    autograd,
    cuda,
    backends,
    distributions,
    utils,
    jit,
    hub,
    profiler,
    ao,
    fx,
)

"""NumPy interop."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1


def test_numpy_export_bool():
    src = torch.tensor([True, False, True, False])
    bit = bit1(src)
    arr = bit.numpy()
    assert arr.dtype.kind == "b"
    assert arr.tolist() == [True, False, True, False]


def test_numpy_export_float():
    src = torch.tensor([1.0, 2.0, 3.0])
    t = brute.tensor(src)
    arr = t.numpy()
    assert arr.tolist() == [1.0, 2.0, 3.0]


def test_numpy_import_via_from_numpy():
    import numpy as np
    arr = np.array([True, False, True])
    t = brute.from_numpy(arr)
    assert t.dtype == torch.bool
    assert t.tolist() == [True, False, True]


def test_numpy_array_interface():
    """__array__ protocol should produce a usable numpy array."""
    import numpy as np
    bit = bit1(torch.tensor([True, False, True]))
    arr = np.asarray(bit)
    assert arr.tolist() == [True, False, True]


def test_numpy_to_bit1_roundtrip():
    import numpy as np
    arr = np.array([True, False, True, True, False])
    bit = bit1(torch.from_numpy(arr))
    assert bit.tolist() == [True, False, True, True, False]


@pytest.mark.cuda
def test_numpy_raises_on_cuda():
    """torch refuses to call .numpy() on a CUDA tensor — bit1 must mirror."""
    x = bit1(torch.tensor([True, False], device="cuda"))
    with pytest.raises((TypeError, RuntimeError)):
        x.as_subclass(torch.Tensor).numpy()

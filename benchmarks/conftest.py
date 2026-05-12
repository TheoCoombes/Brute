import pytest
import torch
import brute


def _probe(device: torch.device) -> bool:
    """Check that brute ops are compiled for this device."""
    try:
        a = brute.rand(16, dtype=brute.bit1, device=device)
        torch.ops.brute.packed_popcount(a._packed_buf)
        return True
    except Exception:
        return False


def _available_devices():
    candidates = [torch.device("cpu")]
    if torch.cuda.is_available():
        candidates.append(torch.device("cuda"))
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        candidates.append(torch.device("mps"))
    return [d for d in candidates if _probe(d)]


DEVICES = _available_devices()


@pytest.fixture(params=DEVICES, ids=lambda d: d.type)
def device(request):
    return request.param


@pytest.fixture
def synced(device):
    """Callable that blocks until all pending GPU work on `device` is done."""
    if device.type == "cuda":
        def _s(): torch.cuda.synchronize(device)
    elif device.type == "mps":
        def _s(): torch.mps.synchronize()
    else:
        def _s(): pass
    return _s

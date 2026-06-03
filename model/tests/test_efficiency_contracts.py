"""Source- and runtime-level contracts for the binary-first BEP implementation."""
from pathlib import Path

import torch

import brute
from model import HaemmrConfig, HaemmrLM


ROOT = Path(__file__).resolve().parent.parent


def _text(name):
    return (ROOT / name).read_text()


def test_no_local_megabatch_or_old_flip_cap_api():
    joined = "\n".join(p.read_text() for p in ROOT.glob("*.py"))
    assert "--accum" not in joined
    assert "args.accum" not in joined
    assert "max_flip_frac" not in joined
    assert "--vocab-cap" not in joined
    assert "--offline-codebook" not in joined
    assert "--structured-codebook" not in joined


def test_hot_forward_uses_brute_fast_matmul():
    layers = _text("layers.py")
    assert "z = brute.fast.matmul(a_bit, self.W.bit)" in layers
    assert "return brute.fast.matmul(chat_bit, self.E.bit)" in layers
    assert "sim = brute.fast.matmul(q_bit, self.P.bit)" in layers


def test_v2_removed_position_bound_decode_mode():
    model_py = _text("model.py")
    train_py = _text("train.py")
    cfg = HaemmrConfig(use_position=True)
    assert not hasattr(cfg, "position_decode")
    assert "position_decode" not in model_py
    assert "--position-decode" not in train_py
    assert "decode_pos" not in model_py


def test_no_unpack_or_float_on_hot_path_source():
    """layers.py / bep.py forward+backward never unpack to ±1 or go to float."""
    for name in ("layers.py", "bep.py"):
        src = _text(name)
        assert "to_pm1(" not in src, f"{name} unpacks via to_pm1"
        assert "unpack_pm1(" not in src, f"{name} unpacks via unpack_pm1"
        assert ".float(" not in src, f"{name} drops to float on the hot path"


def test_param_holds_exactly_one_integer_buffer():
    """Each parameter's bytes == Σ H.nbytes (Int16); no float weight copies."""
    m = HaemmrLM(HaemmrConfig(vocab_size=16, D=128, n_layers=1, d_ff=256,
                              n_slots=16, top_k=3, seed=0), device="cpu")
    expected = sum(p.H.nbytes for p in m.parameters())
    assert m.param_bytes() == expected
    # H is int16 ⇒ exactly 2 bytes per weight (≈16×, not ≈80×, the 1-bit size).
    assert m.param_bytes() == 2 * m.num_bit_parameters()
    for p in m.parameters():
        assert p.H.dtype == torch.int16
        assert not hasattr(p, "q") and not hasattr(p, "_pm1")


def test_no_float_parameter_tensors_after_training_step():
    m = HaemmrLM(HaemmrConfig(vocab_size=16, D=128, n_layers=1, d_ff=256,
                              n_slots=16, top_k=3, seed=0), device="cpu")
    ids = torch.randint(0, 16, (2, 8))
    tgt = torch.randint(0, 16, (2, 8))
    m.loss_and_backward(m.forward(ids), tgt)
    for p in m.parameters():
        assert p.H.dtype not in (torch.float32, torch.float64)
        assert p.bit.dtype == brute.bit1


def test_forward_backward_complete_without_unpacking():
    """Patch unpack_pm1/to_pm1 to raise; a full training step must still run."""
    import vsa
    from brute.tensor import Tensor as BT

    orig_unpack = BT.unpack_pm1
    orig_to_pm1 = vsa.to_pm1

    def _boom(*a, **k):
        raise AssertionError("hot path unpacked a bit1 tensor to ±1")

    BT.unpack_pm1 = _boom
    vsa.to_pm1 = _boom
    try:
        m = HaemmrLM(HaemmrConfig(vocab_size=16, D=128, n_layers=1, d_ff=256,
                                  n_slots=16, top_k=3, seed=0), device="cpu")
        ids = torch.randint(0, 16, (2, 8))
        tgt = torch.randint(0, 16, (2, 8))
        matched = torch.full((2, 8), -1)
        matched[:, -2] = 0
        info = m.loss_and_backward(m.forward(ids), tgt, matched=matched)
        assert torch.isfinite(torch.tensor(info["loss"]))
    finally:
        BT.unpack_pm1 = orig_unpack
        vsa.to_pm1 = orig_to_pm1

"""Source-level contracts for the binary-first implementation."""
from pathlib import Path

import torch

from model import HaemmrConfig, HaemmrLM
ROOT = Path(__file__).resolve().parent.parent


def _text(name):
    return (ROOT / name).read_text()


def test_no_local_megabatch_or_old_flip_cap_api():
    joined = "\n".join(p.read_text() for p in ROOT.glob("*.py"))
    assert "--accum" not in joined
    assert "args.accum" not in joined
    assert "max_flip_frac" not in joined


def test_hot_forward_uses_brute_fast_matmul():
    layers = _text("layers.py")
    assert "z = brute.fast.matmul(a_bit, self.W.bit)" in layers
    assert "return brute.fast.matmul(chat_bit, self.E.bit)" in layers
    assert "sim = brute.fast.matmul(q_bit, self.P.bit)" in layers


def test_no_position_bound_decode_mode():
    model_py = _text("model.py")
    train_py = _text("train.py")
    cfg = HaemmrConfig(use_position=True)
    assert not hasattr(cfg, "position_decode")
    assert "position_decode" not in model_py
    assert "--position-decode" not in train_py
    assert "decode_pos" not in model_py


def test_packed_hardware_forward_has_no_unpack_or_pm1_cache():
    cfg = HaemmrConfig(
        vocab_size=128,
        D=128,
        n_layers=1,
        d_ff=256,
        n_slots=32,
        top_k=1,
        epi_read_k=1,
        use_bsr=False,
        use_position=False,
        sem_weight=0.0,
        seed=0,
    )
    model = HaemmrLM(cfg, device="cpu")
    ids = torch.randint(0, cfg.vocab_size, (2, 16))

    from profile_helpers import count_hot_ops

    with count_hot_ops() as counts:
        logits = model.forward(ids)

    assert logits.shape == (32, cfg.vocab_size)
    assert counts["unpack_pm1"] == 0
    assert counts["as_tensor_bit1_pack"] == 0
    assert sum(1 for p in model.parameters() if p._pm1 is not None) == 0

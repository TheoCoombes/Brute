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
    # BooleanLinear fused path now uses matmul_sign; int32 matmul kept for boundary_nu
    assert "brute.fast.matmul_sign(a_bit, self.W.bit" in layers
    assert "brute.fast.matmul(a_bit, self.W.bit)" in layers   # boundary_nu path
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
        epi_registers=0,
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


def test_boolean_linear_fused_path_uses_matmul_sign():
    """BooleanLinear without boundary_nu uses matmul_sign, not matmul+sign."""
    from layers import BooleanLinear
    from vsa import random_hypervectors
    from profile_helpers import count_hot_ops

    lin = BooleanLinear(128, 64, name="lin_test", boundary_nu=None)
    x   = random_hypervectors(8, 128)
    with count_hot_ops() as counts:
        out, z = lin.forward(x)
    assert z is None, "z must be None for fused path"
    assert counts["fast_matmul_sign"] >= 1, "matmul_sign must be called"
    assert counts["fast_matmul"] == 0,      "plain matmul must NOT be called"
    assert counts["threshold_pack"] == 0,   "pack_sign must NOT be called separately"


def test_episodic_forward_is_linear_not_quadratic():
    """Chunk scores must be bounded O(n·C), not O(n²)."""
    from layers import EpisodicSlotMemory
    from vsa import random_hypervectors

    D, n, C = 64, 128, 64
    mem = EpisodicSlotMemory(D, name="epi_contract", epi_chunk=C, epi_registers=0)
    c   = random_hypervectors(n, D).reshape(1, n, D)
    pos = random_hypervectors(n, D)
    mem.forward(c, pos)

    for score_g in mem._cache["chunk_scores"]:
        _, q_cnt, w_cnt = score_g.shape
        assert q_cnt <= C,     f"q_cnt={q_cnt} exceeds chunk size C={C}"
        assert w_cnt <= 2 * C, f"w_cnt={w_cnt} exceeds 2*C={2*C}"
        # No (n, n) matrix: q_cnt*w_cnt must be << n^2
        assert q_cnt * w_cnt <= 2 * C * C, "chunk score matrix is too large (quadratic?)"

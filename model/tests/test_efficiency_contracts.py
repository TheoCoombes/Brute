"""Source-level contracts for the binary-first implementation."""
from pathlib import Path

from model import HaemmrConfig


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
    assert "logits = brute.fast.matmul(chat_bit, self.E.bit)" in layers
    assert "sim = brute.fast.matmul(q_bit, self.P.bit)" in layers


def test_positioned_models_default_to_next_unbind_decode():
    cfg = HaemmrConfig(use_position=True)
    assert cfg.position_decode == "next_unbind"
    cfg = HaemmrConfig(use_position=False)
    assert cfg.position_decode == "none"


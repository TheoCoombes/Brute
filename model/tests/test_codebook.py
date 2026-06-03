"""Offline GPT-2 SimHash codebook (codebook.py) and the TokenCodebook init path."""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brute
import codebook as cb_mod
from bep import pm1_int
from layers import TokenCodebook


@pytest.fixture
def fake_gpt2(monkeypatch):
    """Patch the embedding loader with a deterministic toy matrix (no download)."""
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(50, 16, generator=g)
    monkeypatch.setattr(cb_mod, "_load_gpt2_embeddings", lambda source: emb.clone())
    return emb


def test_codebook_shape_and_pm1(tmp_path, fake_gpt2):
    ids = torch.tensor([1, 5, 9, 9, 2])
    code = cb_mod.gpt2_semantic_codebook(ids, D=64, seed=3, cache_dir=str(tmp_path))
    assert code.shape == (5, 64)
    assert set(code.unique().tolist()) <= {-1.0, 1.0}
    # identical gpt2 ids must map to identical codes
    assert torch.equal(code[2], code[3])


def test_codebook_is_cached_and_deterministic(tmp_path, fake_gpt2):
    ids = torch.tensor([1, 5, 9])
    a = cb_mod.gpt2_semantic_codebook(ids, D=32, seed=7, cache_dir=str(tmp_path))
    files = list(Path(tmp_path).glob("*.pt"))
    assert len(files) == 1                      # one cache blob written
    b = cb_mod.gpt2_semantic_codebook(ids, D=32, seed=7, cache_dir=str(tmp_path))
    assert torch.equal(a, b)                    # cache hit reproduces it exactly


def test_simhash_preserves_semantic_order(tmp_path, monkeypatch):
    # token 0 and 1 share a direction (near-duplicate); token 2 is orthogonal.
    base = torch.zeros(3, 8)
    base[0, 0] = 1.0
    base[1, 0] = 1.0
    base[1, 1] = 0.05
    base[2, 3] = 1.0
    monkeypatch.setattr(cb_mod, "_load_gpt2_embeddings", lambda source: base.clone())
    code = cb_mod.gpt2_semantic_codebook(torch.tensor([0, 1, 2]), D=4096, seed=1,
                                         cache_dir=str(tmp_path))
    sim_close = float((code[0] == code[1]).float().mean())
    sim_far = float((code[0] == code[2]).float().mean())
    assert sim_close > sim_far                  # near tokens get nearer codes


def test_token_codebook_accepts_offline_init():
    V, D = 8, 64
    g = torch.Generator().manual_seed(2)
    init = torch.where(torch.rand(V, D, generator=g) > 0.5, 1.0, -1.0)
    cb = TokenCodebook(V, D, name="E", init_pm1=init)
    # the stored visible weight must match the requested ±1 signs
    expected = brute.as_tensor(init > 0, dtype=brute.bit1)
    assert torch.equal(pm1_int(cb.E.bit), pm1_int(expected))
    ids = torch.tensor([[0, 3, 7]])
    assert cb.embed(ids).shape == (1, 3, D)


def test_token_codebook_init_shape_mismatch_raises():
    with pytest.raises(ValueError):
        TokenCodebook(8, 64, init_pm1=torch.ones(8, 32))

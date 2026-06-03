"""Offline token codebook built from GPT-2's pretrained embeddings.

HÆMMR's :class:`TokenCodebook` is a *fixed prototype classifier* (V × D bits):
it is never trained by the flip rule, so its geometry is decided once, up front.
The inline BEF (``binary_equiangular_frame``) makes codes that are maximally and
uniformly separated — good for decode hygiene, but **semantically blind**: every
token is equidistant from every other, so ``" cat"`` is no nearer ``" dog"`` than
to ``" the"``.  The model then has to learn all lexical structure from scratch.

This module instead reuses the GPT-2 tokenizer's *pretrained* token embeddings
(``wte``) — semantic geometry the user already has — and folds them down to a
balanced ``D``-bit code per token by **SimHash** (sign random projection):

    code(t) = sign(R · (e_t − ē)),   R ~ N(0, 1) of shape (D, 768) fixed by seed.

SimHash is a locality-sensitive hash: ``P[bit_i(a) = bit_i(b)] = 1 − θ(a,b)/π``,
so the expected Hamming similarity between two codes is monotone in the cosine
similarity of their GPT-2 embeddings.  Semantically close tokens therefore get
close codes — the decode sweep starts with real lexical structure for free.

The result is keyed by ``(gpt2_ids, D, seed)`` and cached to disk: building it
needs the GPT-2 embedding matrix (a one-off download via ``transformers``), but
every subsequent run loads a small ``.pt`` blob and never touches the network.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import torch


def _cache_key(gpt2_ids: torch.Tensor, D: int, seed: int, source: str) -> str:
    h = hashlib.sha1()
    h.update(source.encode())
    h.update(f"|D={D}|seed={seed}|".encode())
    h.update(gpt2_ids.to(torch.int64).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _load_gpt2_embeddings(source: str) -> torch.Tensor:
    """Return the (vocab, hidden) pretrained input-embedding matrix for ``source``."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(source)
    with torch.no_grad():
        emb = model.get_input_embeddings().weight.detach().float().clone()
    return emb


def gpt2_semantic_codebook(
    gpt2_ids: torch.Tensor,
    D: int,
    *,
    seed: int = 0,
    cache_dir: str = "./.data/codebook",
    source: str = "gpt2",
    cache: bool = True,
) -> torch.Tensor:
    """Build a ``(len(gpt2_ids), D)`` ±1 float codebook from GPT-2 embeddings.

    ``gpt2_ids`` are the GPT-2 vocabulary ids each *compact* row stands for (i.e.
    ``Corpus.compact_to_gpt2``).  Returns ±1 floats so the result drops straight
    into :class:`layers.TokenCodebook` like the BEF frame does.
    """
    gpt2_ids = gpt2_ids.to(torch.int64).cpu()
    key = _cache_key(gpt2_ids, D, seed, source)
    path = os.path.join(cache_dir, f"{key}.pt")

    if cache and os.path.exists(path):
        return torch.load(path, weights_only=True)

    emb = _load_gpt2_embeddings(source)            # (vocab, H)
    rows = emb[gpt2_ids.clamp_max(emb.shape[0] - 1)]   # (V, H)
    rows = rows - emb.mean(dim=0, keepdim=True)    # centre → balanced SimHash bits

    g = torch.Generator().manual_seed(int(seed))
    R = torch.randn(rows.shape[1], D, generator=g)  # (H, D) fixed random hyperplanes
    proj = rows @ R                                  # (V, D)
    code = torch.where(proj >= 0, 1.0, -1.0)         # SimHash sign

    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(code, path)
    return code


def from_corpus(corpus, D: int, *, seed: int = 0,
                cache_dir: str = "./.data/codebook",
                cache: bool = True) -> torch.Tensor:
    """Convenience: build the offline codebook for a :class:`data.Corpus`.

    Uses ``corpus.compact_to_gpt2`` so codebook row ``i`` is the SimHash of the
    GPT-2 embedding for the token compact-id ``i`` decodes to.  Best results when
    the corpus tokenizer is the same GPT-2 BPE that produced the embeddings.
    """
    source = getattr(getattr(corpus, "tokenizer", None), "name_or_path", "gpt2") or "gpt2"
    return gpt2_semantic_codebook(corpus.compact_to_gpt2, D, seed=seed,
                                  cache_dir=cache_dir, source=source, cache=cache)

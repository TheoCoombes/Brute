"""Datasets used by the BEP RNN language model demo.

* ``tinyshakespeare`` — Karpathy's TinyShakespeare corpus (~1 MB of
  Shakespeare, English-only), tokenized with the HuggingFace GPT-2 BPE
  tokenizer. To keep the BEF output codebook size tractable, we cap the
  vocabulary to the most-frequent ``vocab_cap`` tokens (default 512) and
  fold all rarer ids into a single OOV id (placed at index 0).

The loader returns a ``ShakespeareCorpus`` dataclass containing:
  * ``train_ids`` / ``val_ids`` — int64 1-D tensors of compact vocab indices.
  * ``vocab_size`` — the actual exposed vocab size (= ``vocab_cap`` if
    the corpus has more than that many distinct tokens, otherwise the
    number of distinct tokens + 1 for OOV).
  * ``compact_to_gpt2`` — int64 1-D map ``compact_id → gpt2_id`` for
    detokenisation; OOV maps back to ``tokenizer.eos_token_id``.
  * ``tokenizer`` — the ``GPT2Tokenizer`` instance (kept for decoding).

There is also a ``make_lm_batches`` helper that slices a long token stream
into ``(input, target)`` pairs of length ``T`` for next-token training.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Tuple

import torch


# Canonical TinyShakespeare download (single .txt, ~1 MB). Used by Karpathy
# in the char-rnn / nanoGPT lineage; English-only, public domain.
_TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)

OOV_ID: int = 0  # Position of the OOV bucket in the compact vocabulary.


@dataclass
class ShakespeareCorpus:
    """A tokenised text corpus with a compact (capped) vocabulary."""

    train_ids: torch.Tensor          # 1-D int64 — compact ids
    val_ids: torch.Tensor            # 1-D int64 — compact ids
    vocab_size: int                  # exposed vocab size, including OOV
    compact_to_gpt2: torch.Tensor    # 1-D int64 (vocab_size,)
    tokenizer: object                # GPT2Tokenizer

    def decode(self, ids: torch.Tensor) -> str:
        """Decode a 1-D tensor of compact ids back to a string."""
        gpt2_ids = self.compact_to_gpt2[ids.long().cpu()].tolist()
        return self.tokenizer.decode(gpt2_ids)


def _download_tinyshakespeare(data_root: str) -> str:
    os.makedirs(data_root, exist_ok=True)
    path = os.path.join(data_root, "tinyshakespeare.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(_TINYSHAKESPEARE_URL, path)
    return path


def tinyshakespeare(
    *,
    data_root: str = "./.data",
    vocab_cap: int = 512,
    val_frac: float = 0.1,
    cache_tokens: bool = True,
) -> ShakespeareCorpus:
    """Load and tokenise TinyShakespeare with a capped GPT-2 vocabulary.

    Parameters
    ----------
    data_root : str
        Where to cache the raw text and pre-computed tokenisation.
    vocab_cap : int
        Maximum vocabulary size (including the OOV bucket). Top ``vocab_cap−1``
        most-frequent BPE ids are kept; everything else folds into OOV.
    val_frac : float
        Fraction of the token stream held out as validation (taken
        contiguously from the *end* of the corpus).
    cache_tokens : bool
        If True, cache the raw GPT-2 token ids in a ``.pt`` file so we don't
        re-tokenise on every load.
    """
    from transformers import GPT2Tokenizer  # local import — heavy dep

    text_path = _download_tinyshakespeare(data_root)
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    tok = GPT2Tokenizer.from_pretrained("gpt2")

    tok_cache = os.path.join(data_root, "tinyshakespeare.gpt2_ids.pt")
    if cache_tokens and os.path.exists(tok_cache):
        full_ids = torch.load(tok_cache, weights_only=True)
    else:
        # ``encode`` is fast enough on a 1 MB corpus (a few seconds).
        full_ids = torch.tensor(tok.encode(text), dtype=torch.long)
        if cache_tokens:
            torch.save(full_ids, tok_cache)

    # Build the capped vocabulary on the FULL corpus (train+val) so that the
    # validation set never produces a token that is unknown to the model —
    # we map all rare ids to OOV regardless of split.
    counts = torch.bincount(full_ids)
    # Keep the top (vocab_cap - 1) most frequent ids; index 0 is reserved
    # for the OOV bucket.
    n_keep = max(min(int(vocab_cap) - 1, int((counts > 0).sum().item())), 1)
    _, top_gpt2 = torch.topk(counts, k=n_keep)
    top_gpt2, _ = torch.sort(top_gpt2)   # stable ordering of the compact ids

    # Compact id 0 = OOV (mapped to GPT-2 EOS for round-trip decoding).
    compact_to_gpt2 = torch.full((n_keep + 1,), int(tok.eos_token_id), dtype=torch.long)
    compact_to_gpt2[1:] = top_gpt2

    # Inverse map: gpt2 id → compact id (OOV id by default).
    gpt2_to_compact = torch.full((int(counts.numel()),), OOV_ID, dtype=torch.long)
    gpt2_to_compact[top_gpt2] = torch.arange(1, n_keep + 1, dtype=torch.long)

    compact_ids = gpt2_to_compact[full_ids]                          # (N,)
    n_val = max(int(round(val_frac * compact_ids.shape[0])), 1)
    train_ids = compact_ids[:-n_val]
    val_ids = compact_ids[-n_val:]

    return ShakespeareCorpus(
        train_ids=train_ids,
        val_ids=val_ids,
        vocab_size=n_keep + 1,
        compact_to_gpt2=compact_to_gpt2,
        tokenizer=tok,
    )


# ── Batching ─────────────────────────────────────────────────────────────────

def make_lm_batches(
    ids: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    seed: int = 0,
    shuffle: bool = True,
    mask_target_id: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice a 1-D token stream into ``(input, target)`` pairs of length ``T``.

    For each starting position ``i``, the input is ``ids[i : i+T]`` and the
    target is ``ids[i+1 : i+1+T]`` (standard next-token-prediction layout,
    teacher-forced).

    Returns
    -------
    inputs, targets : torch.LongTensor
        Both of shape ``(N, T)`` where ``N`` is the number of full chunks
        that fit in ``ids`` (with stride ``T``). Rows are shuffled when
        ``shuffle=True``.
    """
    n = ids.shape[0]
    n_chunks = (n - 1) // seq_len
    if n_chunks <= 0:
        raise ValueError(
            f"token stream length {n} is too short for seq_len={seq_len}"
        )
    base = torch.arange(n_chunks, dtype=torch.long) * seq_len
    offsets = torch.arange(seq_len, dtype=torch.long)
    chunk_idx = base.unsqueeze(1) + offsets.unsqueeze(0)              # (N, T)
    inputs = ids[chunk_idx]                                           # (N, T)
    targets = ids[chunk_idx + 1]                                      # (N, T)
    if mask_target_id is not None:
        # Replace masked target ids with IGNORE_INDEX (-1) so the model does
        # not waste updates trying to predict the OOV bucket. The input
        # stream is left untouched — the model still consumes OOV embeddings.
        targets = torch.where(targets == int(mask_target_id),
                              torch.full_like(targets, -1),
                              targets)

    if shuffle:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        perm = torch.randperm(n_chunks, generator=g)
        inputs = inputs[perm]
        targets = targets[perm]

    # batch_size is kept as an argument so callers can compose with their own
    # batch loop. We return the full set of chunks; callers slice in chunks
    # of ``batch_size`` themselves.
    _ = batch_size
    return inputs, targets


# ── Tiny synthetic dataset for tests ─────────────────────────────────────────

def repeating_sequence(
    *,
    cycle: int = 8,
    vocab_size: int = 16,
    n_tokens: int = 4096,
    seed: int = 0,
) -> torch.Tensor:
    """A deterministic length-``cycle`` repeating token stream.

    The next token after ``x_t`` is ``(x_t + 1) mod cycle`` (with a few
    distractor tokens never appearing). A binary RNN with K_h ≥ ``cycle``
    should solve next-token prediction to ~100% — the test suite uses this
    to sanity-check that BPTT learns the recurrent dependency.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    start = int(torch.randint(0, cycle, (1,), generator=g).item())
    ids = torch.tensor([(start + i) % cycle for i in range(n_tokens)],
                       dtype=torch.long)
    # vocab_size may be larger than ``cycle``; values above ``cycle`` never
    # occur which is fine — the BEF still produces a P of size
    # (vocab_size, K_h).
    if vocab_size <= cycle:
        raise ValueError("vocab_size must exceed cycle length")
    return ids

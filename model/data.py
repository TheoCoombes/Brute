"""WikiText data loader with the GPT-2 BPE tokenizer and a capped vocabulary.

HÆMMR decodes by a min-Hamming sweep over the whole codebook ``E`` (V × D bits),
so for a small local demo we cap the (50k-token) GPT-2 vocabulary to the top
``vocab_cap`` most-frequent tokens and fold the rest into a single OOV bucket at
index 0.  This keeps the codebook — and the decode sweep — small and fast while
still using the real GPT-2 tokenizer the user asked for.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

OOV_ID = 0


@dataclass
class Corpus:
    train_ids: torch.Tensor          # 1-D int64 compact ids
    val_ids: torch.Tensor            # 1-D int64 compact ids
    vocab_size: int                  # includes the OOV bucket
    compact_to_gpt2: torch.Tensor    # (vocab_size,) int64
    tokenizer: object

    def decode(self, ids: torch.Tensor) -> str:
        gpt2_ids = self.compact_to_gpt2[ids.long().cpu()].tolist()
        return self.tokenizer.decode(gpt2_ids)

    def encode(self, text: str) -> torch.Tensor:
        """Encode a prompt to compact ids (unknown tokens → OOV)."""
        gpt2_ids = self.tokenizer.encode(text)
        table = torch.full((int(self.compact_to_gpt2.max().item()) + 1,),
                           OOV_ID, dtype=torch.long)
        table[self.compact_to_gpt2] = torch.arange(self.vocab_size, dtype=torch.long)
        gi = torch.tensor(gpt2_ids, dtype=torch.long).clamp_max(table.numel() - 1)
        ids = table[gi]
        if ids.numel() == 0:
            ids = torch.tensor([1], dtype=torch.long)
        return ids


def _load_wikitext_text(name: str, split: str) -> str:
    from datasets import load_dataset
    cfg = {"wikitext-2": "wikitext-2-raw-v1", "wikitext-103": "wikitext-103-raw-v1"}[name]
    ds = load_dataset("wikitext", cfg, split=split)
    return "\n".join(t for t in ds["text"] if t)


def wikitext(
    *,
    name: str = "wikitext-2",
    data_root: str = "./.data",
    vocab_cap: int = 2048,
    cache: bool = True,
    max_train_tokens: Optional[int] = None,
) -> Corpus:
    """Load WikiText, tokenise with GPT-2 BPE, cap the vocabulary."""
    from transformers import GPT2TokenizerFast

    os.makedirs(data_root, exist_ok=True)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    cache_path = os.path.join(data_root, f"{name}.gpt2_ids.pt")
    if cache and os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=True)
        train_gpt2, val_gpt2 = blob["train"], blob["val"]
    else:
        train_txt = _load_wikitext_text(name, "train")
        val_txt = _load_wikitext_text(name, "validation")
        train_gpt2 = torch.tensor(tok.encode(train_txt), dtype=torch.long)
        val_gpt2 = torch.tensor(tok.encode(val_txt), dtype=torch.long)
        if cache:
            torch.save({"train": train_gpt2, "val": val_gpt2}, cache_path)

    if max_train_tokens is not None:
        train_gpt2 = train_gpt2[:max_train_tokens]

    # Build capped vocab on the training split.
    counts = torch.bincount(train_gpt2, minlength=int(tok.vocab_size))
    n_keep = max(min(int(vocab_cap) - 1, int((counts > 0).sum().item())), 1)
    _, top = torch.topk(counts, k=n_keep)
    top, _ = torch.sort(top)

    compact_to_gpt2 = torch.full((n_keep + 1,), int(tok.eos_token_id), dtype=torch.long)
    compact_to_gpt2[1:] = top
    gpt2_to_compact = torch.full((int(tok.vocab_size),), OOV_ID, dtype=torch.long)
    gpt2_to_compact[top] = torch.arange(1, n_keep + 1, dtype=torch.long)

    train_ids = gpt2_to_compact[train_gpt2]
    val_ids = gpt2_to_compact[val_gpt2]

    return Corpus(train_ids=train_ids, val_ids=val_ids, vocab_size=n_keep + 1,
                  compact_to_gpt2=compact_to_gpt2, tokenizer=tok)


def make_lm_batches(ids: torch.Tensor, *, seq_len: int, mask_oov: bool = True,
                    seed: int = 0, shuffle: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice a token stream into next-token ``(input, target)`` chunks of length T."""
    n = ids.shape[0]
    n_chunks = (n - 1) // seq_len
    if n_chunks <= 0:
        raise ValueError(f"stream length {n} too short for seq_len={seq_len}")
    base = torch.arange(n_chunks, dtype=torch.long) * seq_len
    off = torch.arange(seq_len, dtype=torch.long)
    idx = base.unsqueeze(1) + off.unsqueeze(0)
    inputs = ids[idx]
    targets = ids[idx + 1]
    if mask_oov:
        targets = torch.where(targets == OOV_ID, torch.full_like(targets, -1), targets)
    if shuffle:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        perm = torch.randperm(n_chunks, generator=g)
        inputs, targets = inputs[perm], targets[perm]
    return inputs, targets


def repeating_sequence(*, cycle: int = 8, n_tokens: int = 4096, seed: int = 0) -> torch.Tensor:
    """Deterministic length-``cycle`` repeating stream (synthetic learning test)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    start = int(torch.randint(0, cycle, (1,), generator=g).item())
    return torch.tensor([(start + i) % cycle for i in range(n_tokens)], dtype=torch.long)

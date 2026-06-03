"""WikiText data loader with the full GPT-2 BPE tokenizer.

The current model uses the whole tokenizer rather than a capped subset, so the
token stream, codebook rows, and checkpoint vocabulary all stay aligned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class Corpus:
    train_ids: torch.Tensor          # 1-D int64 GPT-2 token ids
    val_ids: torch.Tensor            # 1-D int64 GPT-2 token ids
    vocab_size: int                  # full tokenizer size
    compact_to_gpt2: torch.Tensor    # (vocab_size,) int64
    tokenizer: object

    def decode(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids.long().cpu().tolist())

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to full-tokenizer GPT-2 ids."""
        gpt2_ids = self.tokenizer.encode(text)
        if not gpt2_ids:
            eos = int(getattr(self.tokenizer, "eos_token_id", 0) or 0)
            gpt2_ids = [eos]
        return torch.tensor(gpt2_ids, dtype=torch.long)


def _load_wikitext_text(name: str, split: str) -> str:
    from datasets import load_dataset
    cfg = {"wikitext-2": "wikitext-2-raw-v1", "wikitext-103": "wikitext-103-raw-v1"}[name]
    ds = load_dataset("wikitext", cfg, split=split)
    return "\n".join(t for t in ds["text"] if t)


def wikitext(
    *,
    name: str = "wikitext-2",
    data_root: str = "./.data",
    cache: bool = True,
    max_train_tokens: Optional[int] = None,
) -> Corpus:
    """Load WikiText and tokenise with GPT-2 BPE."""
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

    vocab_size = int(tok.vocab_size)
    compact_to_gpt2 = torch.arange(vocab_size, dtype=torch.long)

    return Corpus(train_ids=train_gpt2, val_ids=val_gpt2, vocab_size=vocab_size,
                  compact_to_gpt2=compact_to_gpt2, tokenizer=tok)


def make_lm_batches(ids: torch.Tensor, *, seq_len: int,
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

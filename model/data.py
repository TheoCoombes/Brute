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

    return Corpus(train_ids=train_gpt2, val_ids=val_gpt2, vocab_size=vocab_size,
                  tokenizer=tok)


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


def tiny_shakespeare(
    *,
    data_root: str = "./.data",
    cache: bool = True,
    val_frac: float = 0.1,
    max_chars: Optional[int] = None,
) -> Corpus:
    """Load tiny_shakespeare.txt and tokenise with GPT-2 BPE (90/10 train/val split).

    ``max_chars`` truncates the source text before encoding to keep tokenisation fast.
    ~4 chars/token on average, so max_chars=200_000 ≈ 50k tokens.
    """
    from transformers import GPT2TokenizerFast

    os.makedirs(data_root, exist_ok=True)
    txt_path = os.path.join(data_root, "tiny_shakespeare.txt")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"tiny_shakespeare.txt not found at {txt_path}")

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    suffix = f"_{max_chars}" if max_chars is not None else ""
    cache_path = os.path.join(data_root, f"tiny_shakespeare{suffix}.gpt2_ids.pt")
    if cache and os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=True)
        train_ids, val_ids = blob["train"], blob["val"]
    else:
        with open(txt_path, "r") as f:
            text = f.read()
        if max_chars is not None:
            text = text[:max_chars]
        all_ids = torch.tensor(tok.encode(text), dtype=torch.long)
        n_val = max(1, int(len(all_ids) * val_frac))
        train_ids = all_ids[:-n_val]
        val_ids = all_ids[-n_val:]
        if cache:
            torch.save({"train": train_ids, "val": val_ids}, cache_path)

    return Corpus(train_ids=train_ids, val_ids=val_ids,
                  vocab_size=int(tok.vocab_size), tokenizer=tok)


def tiny_shakespeare_char(
    *,
    data_root: str = "./.data",
    val_frac: float = 0.1,
) -> Corpus:
    """Load tiny_shakespeare.txt as a character-level corpus (~65 token vocab)."""
    txt_path = os.path.join(data_root, "tiny_shakespeare.txt")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"tiny_shakespeare.txt not found at {txt_path}")
    with open(txt_path, "r") as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    class CharTokenizer:
        def __init__(self, stoi, itos):
            self.stoi = stoi
            self.itos = itos
            self.vocab_size = len(stoi)
        def encode(self, s):
            return [self.stoi[c] for c in s if c in self.stoi]
        def decode(self, ids):
            return "".join(self.itos.get(int(i), "?") for i in ids)

    tok = CharTokenizer(stoi, itos)
    all_ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_val = max(1, int(len(all_ids) * val_frac))
    return Corpus(train_ids=all_ids[:-n_val], val_ids=all_ids[-n_val:],
                  vocab_size=len(chars), tokenizer=tok)


def repeating_sequence(*, cycle: int = 8, n_tokens: int = 4096, seed: int = 0) -> torch.Tensor:
    """Deterministic length-``cycle`` repeating stream (synthetic learning test)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    start = int(torch.randint(0, cycle, (1,), generator=g).item())
    return torch.tensor([(start + i) % cycle for i in range(n_tokens)], dtype=torch.long)

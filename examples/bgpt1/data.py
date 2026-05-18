"""Tokenizer + tiny text corpus for the BGPT-1 Phase-1 experiment.

We use the GPT-2 byte-level BPE tokenizer (50257 tokens) — the same one
Karpathy's nanoGPT uses, well-tested and easy to load via the
``tokenizers`` library. For Phase-1 toy runs we cap the vocabulary at a
small subset of the most frequent tokens to keep the model tiny.

Default corpus is the public-domain *Tiny Shakespeare* dataset (~1 MB,
~330k tokens after GPT-2 BPE), downloaded once and cached locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Tuple
import os
import urllib.request

import torch

from tokenizers import Tokenizer

_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
_CACHE_DIR.mkdir(exist_ok=True)

_TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
_TINY_SHAKESPEARE_PATH = _CACHE_DIR / "tiny_shakespeare.txt"

# Pre-trained byte-level BPE tokenizer (GPT-2, 50k vocab) hosted on
# HuggingFace.  The tokenizers library can load it directly from the hub.
_GPT2_TOKENIZER_NAME = "gpt2"


def download_tiny_shakespeare(path: Path = _TINY_SHAKESPEARE_PATH) -> Path:
    """Download Tiny Shakespeare once into the local cache."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading tiny shakespeare → {path} ...")
    urllib.request.urlretrieve(_TINY_SHAKESPEARE_URL, path)
    return path


def load_corpus_text(path: Path = _TINY_SHAKESPEARE_PATH) -> str:
    """Return the raw corpus text (downloading if necessary)."""
    download_tiny_shakespeare(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_wikitext(
    name: str = "wikitext-2-raw-v1",
    split: str = "train",
) -> str:
    """Load a WikiText split via HuggingFace ``datasets`` and concatenate
    all rows into a single string. Cached at first call.

    ``name`` is one of:
      ``wikitext-2-raw-v1`` (~10 MB, ~2 M tokens after GPT-2 BPE)
      ``wikitext-103-raw-v1`` (~500 MB, ~100 M tokens)
    """
    from datasets import load_dataset
    cache_path = _CACHE_DIR / f"{name}.{split}.txt"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    print(f"  loading wikitext '{name}' [{split}] via HuggingFace datasets …")
    ds = load_dataset("wikitext", name, split=split)
    # Join all non-empty rows. WikiText is pre-tokenized at the article
    # level — each row is a paragraph or section header.
    text = "\n".join(row["text"] for row in ds if row["text"].strip())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def load_tokenizer(name: str = _GPT2_TOKENIZER_NAME, text: str | None = None
                   ) -> "Tokenizer | CharTokenizer":
    """Load a tokenizer.

    If ``name == "char"`` (or HuggingFace is unreachable and ``text`` is
    supplied), returns a precomputed char-level tokenizer. Otherwise loads
    the named HuggingFace tokenizer (default: GPT-2 BPE).
    """
    if name == "char":
        if text is None:
            raise ValueError("char tokenizer requires the corpus text up-front")
        return CharTokenizer.from_text(text)
    try:
        return Tokenizer.from_pretrained(name)
    except Exception as e:
        print(f"  warning: failed to load HF tokenizer {name!r} ({e})")
        if text is not None:
            print(f"  falling back to char tokenizer")
            return CharTokenizer.from_text(text)
        raise


class CharTokenizer:
    """Char-level tokenizer with a fixed vocab learned from a sample text.

    Closest analog to nanoGPT's character-level baseline. Compact (single-
    digit dozens to hundreds of tokens) so the model's vocab embedding
    stays small for Phase-1 toy runs.
    """

    def __init__(self, stoi: dict[str, int], itos: list[str]):
        self._stoi = stoi
        self._itos = itos

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        return cls(stoi, list(chars))

    def get_vocab_size(self) -> int:
        return len(self._itos)

    def encode(self, text: str) -> "_CharEncoding":
        ids = [self._stoi.get(c, 0) for c in text]
        return _CharEncoding(ids)

    def decode(self, ids: List[int]) -> str:
        return "".join(self._itos[i] if 0 <= i < len(self._itos) else "?" for i in ids)


class _CharEncoding:
    def __init__(self, ids: list[int]):
        self.ids = ids


def tokenize_corpus(
    text: str,
    tokenizer: Tokenizer | _CharTokenizer,
    max_vocab: int | None = None,
) -> Tuple[torch.Tensor, int]:
    """Tokenize the corpus into a long tensor.

    If ``max_vocab`` is set and the tokenizer produces ids > max_vocab,
    those ids are remapped to a special unknown id 0 (and the actual
    returned vocab size is ``max_vocab``).
    """
    enc = tokenizer.encode(text)
    ids = enc.ids
    vocab_size = tokenizer.get_vocab_size()
    if max_vocab is not None and max_vocab < vocab_size:
        # OOV → 0. This keeps the toy model's parameter budget small.
        ids = [i if i < max_vocab else 0 for i in ids]
        vocab_size = max_vocab
    return torch.tensor(ids, dtype=torch.long), vocab_size


def batch_iter(
    ids: torch.Tensor,
    batch_size: int,
    context: int,
    seed: int = 0,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Endless random-window batch iterator.

    Yields ``(inputs, targets)`` where targets are inputs shifted by 1.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    n = ids.numel()
    while True:
        starts = torch.randint(0, n - context - 1, (batch_size,), generator=g)
        x = torch.stack([ids[s : s + context] for s in starts])
        y = torch.stack([ids[s + 1 : s + 1 + context] for s in starts])
        yield x, y


__all__ = [
    "load_tokenizer",
    "load_corpus_text",
    "load_wikitext",
    "tokenize_corpus",
    "batch_iter",
    "download_tiny_shakespeare",
]

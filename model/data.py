"""WikiText and TinyShakespeare data loaders for HÆMMR training.

WikiText uses GPT-2 BPE with a capped vocabulary — suitable for word-level LM
experiments at any scale but has a high OOV rate at small vocab caps.

TinyShakespeare is character-level (~1 MB, 65 unique chars).  It is much cleaner
for small-scale binary LM experiments: no BPE fragmentation artefacts, tiny
vocab that fits perfectly in a small codebook, continuous prose without Wikipedia
markup noise.  Download is cached on first run.

HAEMMR decodes by a min-Hamming sweep over the whole codebook E (V x D bits),
so for a small local demo we cap the 50k-token GPT-2 vocabulary to the top
vocab_cap most-frequent tokens and fold the rest into a single OOV bucket at
index 0.  This keeps the codebook small and fast while still using the real
GPT-2 tokenizer.
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
        return decode_compact(ids, self.tokenizer, self.compact_to_gpt2)

    def encode(self, text: str) -> torch.Tensor:
        """Encode a prompt to compact ids (unknown tokens → OOV)."""
        return encode_compact(text, self.tokenizer, self.compact_to_gpt2)

    def sample_ban_ids(self, *, clean: bool = False) -> list[int]:
        return sample_ban_ids(self.compact_to_gpt2, self.tokenizer, clean=clean)

    def sentence_end_ids(self) -> list[int]:
        return sentence_end_ids(self.compact_to_gpt2, self.tokenizer)


def _compact_table(compact_to_gpt2: torch.Tensor) -> torch.Tensor:
    table = torch.full((int(compact_to_gpt2.max().item()) + 1,), OOV_ID, dtype=torch.long)
    table[compact_to_gpt2.long().cpu()] = torch.arange(compact_to_gpt2.numel(), dtype=torch.long)
    return table


def encode_compact(text: str, tokenizer, compact_to_gpt2: torch.Tensor,
                   *, prefix_fallback: bool = True) -> torch.Tensor:
    """Encode text to compact GPT-2 ids, preferring fewer OOV prompt tokens.

    GPT-2 has separate ids for initial words (``"The"``) and space-prefixed words
    (``" The"``).  WikiText mostly trains on the latter, so capped demos often
    know ``" The"`` but not ``"The"``.  For prompts only, a leading-space retry is
    a better context than silently feeding OOV.
    """
    table = _compact_table(compact_to_gpt2)

    def convert(txt: str) -> torch.Tensor:
        gpt2_ids = tokenizer.encode(txt)
        gi = torch.tensor(gpt2_ids, dtype=torch.long).clamp_max(table.numel() - 1)
        ids = table[gi]
        if ids.numel() == 0:
            ids = torch.tensor([1], dtype=torch.long)
        return ids

    ids = convert(text)
    if prefix_fallback and text and not text[0].isspace():
        prefixed = convert(" " + text)
        if int((prefixed == OOV_ID).sum().item()) < int((ids == OOV_ID).sum().item()):
            ids = prefixed
    return ids


def decode_compact(ids: torch.Tensor, tokenizer, compact_to_gpt2: torch.Tensor) -> str:
    gpt2_ids = compact_to_gpt2[ids.long().cpu()].tolist()
    return tokenizer.decode(gpt2_ids)


def sample_ban_ids(compact_to_gpt2: torch.Tensor, tokenizer, *, clean: bool = False) -> list[int]:
    bad = {OOV_ID}
    if not clean:
        return sorted(bad)
    allowed_punct = set(".,!?;:'\"()-")
    allowed_word_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"()-")
    for i, gpt2_id in enumerate(compact_to_gpt2.long().cpu().tolist()):
        text = tokenizer.decode([gpt2_id])
        stripped = text.strip()
        if not stripped:
            bad.add(i)
        elif "<|" in text or "\n" in text or "\r" in text or "\t" in text:
            bad.add(i)
        elif any(ord(ch) < 32 or ord(ch) > 126 for ch in text):
            bad.add(i)
        elif text.startswith(" "):
            if not any(ch.isalnum() for ch in text) or any(ch not in allowed_word_chars for ch in text):
                bad.add(i)
        elif stripped not in allowed_punct:
            # Non-space-prefixed alphabetic tokens are usually GPT-2 continuation
            # fragments in this capped demo; banning them improves sample hygiene.
            bad.add(i)
    return sorted(bad)


def sentence_end_ids(compact_to_gpt2: torch.Tensor, tokenizer) -> list[int]:
    ids = []
    for i, gpt2_id in enumerate(compact_to_gpt2.long().cpu().tolist()):
        stripped = tokenizer.decode([gpt2_id]).strip()
        if stripped in {".", "!", "?"} or stripped.endswith((".", "!", "?")):
            ids.append(i)
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

    # Build the capped vocab on the full training split.  ``max_train_tokens`` is
    # a speed knob for the returned stream, not a different vocabulary policy.
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
    if max_train_tokens is not None:
        train_ids = train_ids[:max_train_tokens]

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


# ── TinyShakespeare (character-level) ──────────────────────────────────────────

_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn"
    "/master/data/tinyshakespeare/input.txt"
)


class _CharTokenizer:
    """Minimal character-level tokenizer that mirrors the GPT-2 tokenizer interface
    used by ``Corpus.decode``, ``Corpus.encode``, ``sample_ban_ids``, and
    ``sentence_end_ids``.

    Character indices start at ``_offset`` (default 1) so that index 0 is reserved
    for the ``OOV_ID`` slot consumed by ``make_lm_batches(mask_oov=True)``.  Since
    all characters in the dataset are in-vocab, OOV_ID never appears and the mask
    is a no-op — but the index convention must match.
    """

    def __init__(self, chars: list, offset: int = 1):
        self._offset = offset
        self._chars = chars           # chars[0] → compact id offset, etc.
        self._c2i = {c: i + offset for i, c in enumerate(chars)}
        self.vocab_size = len(chars) + offset
        self.eos_token_id = self._c2i.get("\n", offset)

    def encode(self, text: str) -> list:
        return [self._c2i.get(c, OOV_ID) for c in text]

    def decode(self, ids) -> str:
        return "".join(
            self._chars[int(i) - self._offset]
            if self._offset <= int(i) < self._offset + len(self._chars)
            else ("?" if int(i) != OOV_ID else "")
            for i in ids
        )


def tiny_shakespeare(
    *,
    data_root: str = "./.data",
    train_frac: float = 0.9,
    max_train_tokens: Optional[int] = None,
) -> Corpus:
    """Character-level Tiny Shakespeare (~1 M chars, 65 unique characters).

    Ideal for small-scale binary LM experiments: clean prose, tiny vocab (65 chars),
    no BPE fragmentation, no OOV tokens.  The file is downloaded once and cached.

    Returns a :class:`Corpus` compatible with :func:`make_lm_batches` and
    :func:`train.py`'s training loop.  ``compact_to_gpt2`` is an identity map
    (compact id == tokenizer id).
    """
    import urllib.request

    os.makedirs(data_root, exist_ok=True)
    cache_path = os.path.join(data_root, "tiny_shakespeare.txt")
    if not os.path.exists(cache_path):
        print(f"  downloading TinyShakespeare → {cache_path}")
        urllib.request.urlretrieve(_SHAKESPEARE_URL, cache_path)

    with open(cache_path, "r", encoding="utf-8") as fh:
        text = fh.read()

    chars = sorted(set(text))
    tok = _CharTokenizer(chars, offset=1)   # index 0 reserved for OOV_ID
    V = tok.vocab_size                       # len(chars) + 1

    all_ids = torch.tensor(tok.encode(text), dtype=torch.long)
    split = int(train_frac * len(all_ids))
    train_ids = all_ids[:split]
    val_ids   = all_ids[split:]

    if max_train_tokens is not None:
        train_ids = train_ids[:max_train_tokens]

    compact_to_gpt2 = torch.arange(V, dtype=torch.long)  # identity map

    return Corpus(
        train_ids=train_ids, val_ids=val_ids,
        vocab_size=V, compact_to_gpt2=compact_to_gpt2, tokenizer=tok,
    )

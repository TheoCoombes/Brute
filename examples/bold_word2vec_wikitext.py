"""
bold_word2vec_wikitext.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Scaled-up cousin of ``bold_word2vec.py`` — same BOLD math, but on a real
English corpus (WikiText-2 via HuggingFace ``datasets``), with d = 1024
1-bit embeddings, an **int8** Boolean accumulator, alias-method negative
sampling, Mikolov frequent-word subsampling, streaming pair generation,
and brute's native packed popcount on the hot path.

Math is identical to ``bold_word2vec.py`` (see that file's header) — only
the data pipeline and per-step infrastructure change here.

Storage / compute summary at the default d = 1024, V ≈ 30k:
    E, C   bit1         (V × d)         ~ 4 MB each            ← model
    m_E,m_C int8        (V × d)         ~ 30 MB each           ← optimiser
                                          (would be 240 MB at int64)
    score  brute.matmul on bit1 → ``K − 2·Hamming`` from fused XNOR-popcount
    per-pair Hamming via  (E ⊕ C).word_popcount().sum(-1)
                                          ← no bool fallback, pure packed
"""

from __future__ import annotations

import argparse
import random
import re
import time
from collections import Counter

import torch

import brute


# ----------------------------------------------------------------------------
# Corpus: WikiText-2 streamed from HuggingFace datasets
# ----------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z][a-z'\-]+")          # alpha tokens, ≥ 2 chars


def stream_corpus(name: str = "wikitext-2-raw-v1",
                  split: str = "train") -> list[list[str]]:
    """Load and tokenise a WikiText split.

    Tokeniser is intentionally simple — lowercase ASCII words ≥ 2 chars,
    apostrophes/hyphens kept. Real BPE is overkill here: word2vec is a
    word-level objective, and a tight word vocab is what we want to test
    the 1-bit Hamming-similarity claim against.
    """
    from datasets import load_dataset
    ds = load_dataset("wikitext", name, split=split)
    tokenised: list[list[str]] = []
    for row in ds:
        toks = _TOKEN_RE.findall(row["text"].lower())
        if len(toks) >= 2:
            tokenised.append(toks)
    return tokenised


def build_vocab(sentences: list[list[str]], min_count: int = 5) -> tuple[
        list[str], dict[str, int], torch.Tensor]:
    """Build vocab + per-word counts. Returns (vocab, word→index, counts)."""
    cnt: Counter[str] = Counter()
    for s in sentences:
        cnt.update(s)
    vocab = [w for w, c in cnt.most_common() if c >= min_count]
    w2i = {w: i for i, w in enumerate(vocab)}
    counts = torch.tensor([cnt[w] for w in vocab], dtype=torch.float64)
    return vocab, w2i, counts


def subsample_mask(counts: torch.Tensor, t: float = 1e-4
                   ) -> torch.Tensor:
    """Mikolov subsampling **keep probability** per word.

        p_keep(w)  =  min(1, sqrt(t / f(w)))   with f(w) = count / total

    Frequent stop-words ("the", "of") get aggressively downsampled; rare
    content words stay at p=1.  Returned as a float32 tensor (V,)."""
    total = counts.sum()
    f = counts / total
    p_keep = torch.sqrt(t / f).clamp(max=1.0)
    return p_keep.to(torch.float32)


# ----------------------------------------------------------------------------
# Streaming pair generator
# ----------------------------------------------------------------------------

def pair_stream(sentences: list[list[str]],
                w2i: dict[str, int],
                p_keep: torch.Tensor,
                window: int,
                seed: int = 0):
    """Yields (centre_idx, context_idx) integer pairs one at a time.

    Frequent-word subsampling is applied per token at sample time (not
    pre-baked into the corpus) so each epoch sees a freshly thinned
    sentence — this is the Mikolov convention.
    """
    rng = random.Random(seed)
    p_keep_list = p_keep.tolist()
    while True:
        rng.shuffle(sentences)
        for sent in sentences:
            ids = []
            for t in sent:
                i = w2i.get(t)
                if i is None:
                    continue
                if rng.random() < p_keep_list[i]:
                    ids.append(i)
            n = len(ids)
            if n < 2:
                continue
            for c_pos in range(n):
                # Dynamic window — sample win ∈ {1, ..., window} per centre
                # (also from the original Mikolov recipe — gives nearer
                # words a higher effective weight).
                win = rng.randint(1, window)
                lo  = max(0, c_pos - win)
                hi  = min(n, c_pos + win + 1)
                centre = ids[c_pos]
                for j in range(lo, hi):
                    if j == c_pos:
                        continue
                    yield centre, ids[j]


def batch_iter(stream, batch: int):
    """Pull `batch` (centre, context) pairs from `stream` and return them
    as two long tensors of shape (B,)."""
    while True:
        cs = []
        os = []
        for _ in range(batch):
            c, o = next(stream)
            cs.append(c)
            os.append(o)
        yield torch.tensor(cs, dtype=torch.long), torch.tensor(os, dtype=torch.long)


# ----------------------------------------------------------------------------
# Vose / Walker alias method for O(1) categorical sampling
# ----------------------------------------------------------------------------

class AliasSampler:
    """Walker alias table for O(1)-per-sample weighted draws.

    Replaces ``torch.multinomial(p, n, replacement=True)``, which is O(V)
    per call — fine at V=135, but on a 30k-word vocab with 5 negatives
    per pair × thousands of batches it dominates wall time.

    Setup is O(V); each draw is one uniform float + one uniform int + a
    comparison + a lookup. Implemented vectorised in torch so a single
    `draw(n)` produces a length-n int64 tensor in one shot.
    """

    def __init__(self, probs: torch.Tensor, seed: int = 0):
        probs = probs.to(torch.float64)
        probs = probs / probs.sum()
        V = probs.numel()
        self.V = V
        self.prob = torch.zeros(V, dtype=torch.float32)
        self.alias = torch.zeros(V, dtype=torch.int64)

        scaled = probs * V
        small, large = [], []
        for i in range(V):
            (small if scaled[i] < 1.0 else large).append(i)

        while small and large:
            s = small.pop()
            l = large.pop()
            self.prob[s]  = float(scaled[s])
            self.alias[s] = l
            scaled[l] = scaled[l] + scaled[s] - 1.0
            if scaled[l] < 1.0:
                small.append(l)
            else:
                large.append(l)
        for i in large + small:
            self.prob[i] = 1.0
            self.alias[i] = i

        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

    def draw(self, n: int, device: str = "cpu") -> torch.Tensor:
        col  = torch.randint(0, self.V, (n,), generator=self.gen)
        coin = torch.rand((n,),               generator=self.gen)
        keep = coin < self.prob[col]
        out  = torch.where(keep, col, self.alias[col])
        return out.to(device=device)


# ----------------------------------------------------------------------------
# Bit-level helpers
# ----------------------------------------------------------------------------

def hamming_per_row_packed(a_bit1, b_bit1) -> torch.Tensor:
    """Per-row Hamming distance using brute's fused packed ops only.

      (a ⊕ b)          -> bit1, packed XOR on the integer buffer
      .word_popcount() -> per-packed-word popcount via libpopcnt
      .sum(dim=-1)     -> per-row total

    Never materialises the bool view. For d=1024 the packed buffer is
    16 uint64 words per row, so this is 16 popcounts + a tiny reduction.
    """
    xor = brute.bitwise_xor(a_bit1, b_bit1)                 # bit1
    return xor.word_popcount().to(torch.int64).sum(dim=-1)  # (..., ) int64


def flip_bits(bit1_tensor, bool_mask) -> "brute.Tensor":
    """Return ``bit1 XOR mask`` — flips every bit where the bool mask is True."""
    mask_bit1 = brute.as_tensor(bool_mask, dtype=brute.bit1,
                                device=bit1_tensor.device)
    return brute.bitwise_xor(bit1_tensor, mask_bit1)


def pm1_int8(bit1_tensor) -> torch.Tensor:
    """±1 (int8) view: e=0 → -1, e=1 → +1."""
    return bit1_tensor.bool().to(torch.int8) * 2 - 1


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train(
    *,
    d:           int   = 1024,
    window:      int   = 5,
    k_neg:       int   = 5,
    batch:       int   = 4096,
    steps:       int   = 4000,
    margin:      int   = 256,        # ≈ d / 4
    m_clip:      int   = 100,        # int8 headroom (max 127, leave room for grad)
    min_count:   int   = 5,
    subsample_t: float = 1e-4,
    seed:        int   = 0,
    device:      str   = "cpu",
):
    torch.manual_seed(seed)
    random.seed(seed)

    print("loading corpus...")
    sentences = stream_corpus("wikitext-2-raw-v1", "train")
    vocab, w2i, counts = build_vocab(sentences, min_count=min_count)
    V = len(vocab)
    n_tokens = int(counts.sum().item())
    print(f"  corpus: {len(sentences)} sentences,  {n_tokens:,} tokens, "
          f"vocab {V:,}  (min_count={min_count})")

    # Mikolov subsampling probability vector
    p_keep = subsample_mask(counts, t=subsample_t)

    # Unigram^0.75 alias table for fast negative sampling
    neg_probs = counts.to(torch.float64).pow(0.75)
    neg_sampler = AliasSampler(neg_probs, seed=seed)

    # Sliding-window pair generator (infinite, shuffled each epoch)
    pairs = pair_stream(sentences, w2i, p_keep, window=window, seed=seed)
    batches = batch_iter(pairs, batch)

    # ─────────── 1-bit model parameters ───────────
    E = brute.randint(0, 2, (V, d), dtype=brute.bit1, device=device)
    C = brute.randint(0, 2, (V, d), dtype=brute.bit1, device=device)
    print(f"  E.nbytes = {E.nbytes / 1e6:.1f} MB,  "
          f"C.nbytes = {C.nbytes / 1e6:.1f} MB  (1 bit per weight)")

    # ─────────── BOLD optimiser state (int8 accumulators) ───────────
    m_E = torch.zeros(V, d, dtype=torch.int8, device=device)
    m_C = torch.zeros(V, d, dtype=torch.int8, device=device)
    print(f"  m_E.nbytes = {m_E.element_size() * m_E.numel() / 1e6:.1f} MB "
          f"(int8 — would be {8 * m_E.numel() / 1e6:.0f} MB at int64)")

    # Per-step scratch (avoid reallocs)
    print(f"\ntraining: batch={batch}  k_neg={k_neg}  steps={steps}  "
          f"margin={margin}  m_clip=±{m_clip}\n")

    t0 = time.time()
    log_every = max(1, steps // 20)
    for step in range(steps):
        ci, oi = next(batches)
        ci = ci.to(device); oi = oi.to(device)
        ni = neg_sampler.draw(batch * k_neg, device=device)               # Bk
        ci_neg = ci.unsqueeze(1).expand(batch, k_neg).reshape(-1)         # Bk

        # ----------------- forward (packed XOR + word popcount) -----------------
        E_pos = E[ci]                                                     # B  × d
        C_pos = C[oi]                                                     # B  × d
        E_neg = E[ci_neg]                                                 # Bk × d
        C_neg = C[ni]                                                     # Bk × d

        h_pos = hamming_per_row_packed(E_pos, C_pos)                      # B
        h_neg = hamming_per_row_packed(E_neg, C_neg)                      # Bk
        s_pos = d - 2 * h_pos                                             # B  int64
        s_neg = (d - 2 * h_neg).view(batch, k_neg)                        # B × k

        # ----------------- hinge loss + integer gradient -----------------
        diff   = s_pos.unsqueeze(1) - s_neg                               # B × k
        active = (margin - diff > 0).to(torch.int64)                      # B × k {0,1}

        # ∂L/∂s_pos  = −Σ_j a_{i,j}    ∂L/∂s_neg = +a_{i,j}
        z_pos = -active.sum(dim=-1)                                       # B  int64
        z_neg =  active.view(-1)                                          # Bk int64

        # ±1 int8 embedding views (used only for grad math, not stored)
        Epm  = pm1_int8(E_pos)
        Cpm  = pm1_int8(C_pos)
        Enpm = pm1_int8(E_neg)
        Cnpm = pm1_int8(C_neg)

        # Per-pair gradient contributions are integer with magnitudes ≤ k_neg
        # (z) × 1 (±1 embed) = bounded ints — keep in int16 to scatter cheaply.
        gE_pos = (z_pos.to(torch.int16).unsqueeze(-1) * Cpm.to(torch.int16))
        gE_neg = (z_neg.to(torch.int16).unsqueeze(-1) * Cnpm.to(torch.int16))
        gC_pos = (z_pos.to(torch.int16).unsqueeze(-1) * Epm.to(torch.int16))
        gC_neg = (z_neg.to(torch.int16).unsqueeze(-1) * Enpm.to(torch.int16))

        # Scatter-aggregate to (V × d), then add into the int8 accumulator
        # under saturation — clamp(±m_clip) keeps everything inside int8 range.
        g_E = torch.zeros(V, d, dtype=torch.int16, device=device)
        g_E.index_add_(0, ci,     gE_pos)
        g_E.index_add_(0, ci_neg, gE_neg)
        g_C = torch.zeros(V, d, dtype=torch.int16, device=device)
        g_C.index_add_(0, oi, gC_pos)
        g_C.index_add_(0, ni, gC_neg)

        # m ← clip(m + g, ±κ).  Promote to int16 for the add, then saturate
        # back into int8. This is the BOLD optimiser step (Eq. 10) with the
        # FP β·m decay replaced by the bounded-accumulator regularisation
        # of Assumption A.5.
        m_E = (m_E.to(torch.int16) + g_E).clamp_(-m_clip, m_clip).to(torch.int8)
        m_C = (m_C.to(torch.int16) + g_C).clamp_(-m_clip, m_clip).to(torch.int8)

        # ----------------- flip rule (Eq. 9) -----------------
        # flip iff sign(m) agrees with sign(w)  ⇔  m · (2w − 1) > 0
        E_pm1 = pm1_int8(E)
        C_pm1 = pm1_int8(C)
        flip_E = (m_E.to(torch.int16) * E_pm1.to(torch.int16)) > 0       # bool V×d
        flip_C = (m_C.to(torch.int16) * C_pm1.to(torch.int16)) > 0

        if flip_E.any():
            E = flip_bits(E, flip_E)
            m_E[flip_E] = 0
        if flip_C.any():
            C = flip_bits(C, flip_C)
            m_C[flip_C] = 0

        if step % log_every == 0 or step == steps - 1:
            loss = (margin - diff).clamp(min=0).float().mean().item()
            print(f"  step {step:5d}  loss {loss:7.2f}  "
                  f"⟨s⁺⟩ {s_pos.float().mean().item():+7.1f}  "
                  f"⟨s⁻⟩ {s_neg.float().mean().item():+7.1f}  "
                  f"flips E/C {int(flip_E.sum()):7d}/{int(flip_C.sum()):7d}  "
                  f"active {active.float().mean().item():.2f}  "
                  f"({time.time() - t0:5.1f}s)")

    print(f"\ntraining time: {time.time() - t0:.1f}s")
    return E, C, vocab, w2i


def differential_analysis(E, vocab, w2i, d: int, w1: str, w2: str) -> list[int]:
    """
    Finds the specific bits that encode the semantic difference between two words.
    Returns a list of the bit indices that differ.
    """
    if w1 not in w2i or w2 not in w2i:
        print(f"  Error: '{w1}' or '{w2}' is out-of-vocab.")
        return []

    e1 = E[w2i[w1]:w2i[w1] + 1]
    e2 = E[w2i[w2]:w2i[w2] + 1]

    # Find where the bits differ. 
    # Assuming the brute.bit1 type supports standard PyTorch equality checks.
    diff_mask = (e1 != e2).squeeze(0) 
    diff_indices = torch.where(diff_mask)[0].tolist()

    print(f"\n=== Differential Analysis: '{w1}' vs '{w2}' ===")
    print(f"  Total bits differing: {len(diff_indices)} / {d}")
    print(f"  Feature axes responsible for the difference: {diff_indices}")
    
    return diff_indices

def global_bit_meaning(E, vocab, d: int, target_bit: int, sample_size: int = 15) -> None:
    """
    Reveals the global meaning of a specific bit by sampling the vocabulary
    on both sides of its hyperplane.
    """
    print(f"\n=== Global Decoding for Bit {target_bit} ===")
    
    # Extract the column for the target bit across the entire vocabulary.
    # Assuming standard PyTorch slicing works on the custom type.
    bit_column = E[:, target_bit]
    
    # Group vocabulary indices by their bit state.
    # Adjust the condition (True/False, 1/0, 1/-1) based on how brute.bit1 exposes values to PyTorch.
    state_1_indices = torch.where(bit_column == True)[0].tolist() 
    state_0_indices = torch.where(bit_column == False)[0].tolist()

    # Sample from the indices to get a broad view of the category.
    # If vocab is sorted by frequency, sampling from the first few thousand yields more recognizable words.
    sample_1 = random.sample(state_1_indices, min(sample_size, len(state_1_indices)))
    sample_0 = random.sample(state_0_indices, min(sample_size, len(state_0_indices)))

    words_1 = [vocab[i] for i in sample_1]
    words_0 = [vocab[i] for i in sample_0]

    print(f"  State 1 (Size: {len(state_1_indices)} words):")
    print("    " + ", ".join(words_1))
    print(f"\n  State 0 (Size: {len(state_0_indices)} words):")
    print("    " + ", ".join(words_0))


def demo(E, vocab, w2i, d: int) -> None:
    """
    Main orchestration function to run the new subspace analysis.
    """
    print(f"Starting 1-Bit Topology Analysis (d = {d} bits)")
    
    # 1. Analogy testing to find semantic axes
    pairs_to_test = [
        ("king", "queen"),      # Expected to isolate Gender / Consort axes
        ("paris", "france"),    # Expected to isolate City / Country axes
        ("computer", "software") # Expected to isolate Hardware / Abstract axes
    ]
    
    # Store the differing bits for the first pair to analyze globally
    target_bits_to_explore = []
    
    for w1, w2 in pairs_to_test:
        diff_bits = differential_analysis(E, vocab, w2i, d, w1, w2)
        if w1 == "king" and w2 == "queen":
            target_bits_to_explore = diff_bits

    # 2. Global decoding
    # If "king" and "queen" differed by 12 bits, let's look at what the first 3 
    # of those bits actually mean across the entire English language.
    if target_bits_to_explore:
        print("\n\n--- Deep Dive: Decoding the King vs. Queen Subspace ---")
        # Just pick the first 3 differing bits to avoid terminal spam
        for bit_idx in target_bits_to_explore[:3]: 
            global_bit_meaning(E, vocab, d, target_bit=bit_idx, sample_size=12)

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--d",       type=int,   default=1024)
    p.add_argument("--steps",   type=int,   default=4000)
    p.add_argument("--batch",   type=int,   default=4096)
    p.add_argument("--k-neg",   type=int,   default=5)
    p.add_argument("--window",  type=int,   default=5)
    p.add_argument("--margin",  type=int,   default=256)
    p.add_argument("--m-clip",  type=int,   default=100)
    p.add_argument("--device",  type=str,   default="cpu")
    p.add_argument("--seed",    type=int,   default=0)
    args = p.parse_args()

    E, C, vocab, w2i = train(
        d=args.d, steps=args.steps, batch=args.batch, k_neg=args.k_neg,
        window=args.window, margin=args.margin, m_clip=args.m_clip,
        device=args.device, seed=args.seed,
    )
    demo(E, vocab, w2i, d=args.d)

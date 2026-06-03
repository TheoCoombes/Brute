# bGPT — BEP, in the language domain

A reference implementation of **Binary Error Propagation** (Colombo et al.,
ICLR 2026 — [arXiv:2512.04189](https://arxiv.org/abs/2512.04189)) applied
to a fully-binary **RNN language model**. Built on top of `brute.bit1`
tensors.

> BEP is the first end-to-end binary training algorithm: every forward
> activation, every backward error signal, and every weight is binary.
> The only piece of non-binary state is an integer-valued *metaplasticity*
> accumulator. All matmuls — forward and through-time backward — run on
> the XNOR + popcount fast path provided by `brute.bit1`.

The paper validates BEP on binary MLPs and binary RNNs; the
`examples/bep` folder covers the MLP. This folder is the **RNN** case,
specialised to next-token prediction on a small English-only corpus
(TinyShakespeare) tokenised with the HuggingFace `GPT2Tokenizer`.

## Files

| file              | what's in it                                                              |
| ----------------- | ------------------------------------------------------------------------- |
| `bep.py`          | `BEPLanguageModel`, the BPTT BEP forward + Eq. 7 backward + Eq. 9 update. |
| `bef.py`          | Greedy coordinate-flip Binary Equiangular Frame generator (Appendix C).   |
| `data.py`         | TinyShakespeare loader + GPT-2 tokenisation + capped vocab + batching.    |
| `train.py`        | CLI training script.                                                      |
| `test_bep.py`     | Local pytest suite — runs as `python -m pytest test_bep.py`.              |

## Architecture

```
inputs:   tokens x_1 .. x_T  ∈ [0, V)
embedding: e_t = P[x_t]                                    ∈ {±1}^{K_h}
state init: a_0 = +1 vector                                ∈ {±1}^{K_h}
recurrent:  z_t = W_xh e_t + W_hh a_{t-1}                  ∈ ℤ^{K_h}    (sum of two XNOR-popcount dots)
            a_t = sign(z_t)                                ∈ {±1}^{K_h}
output:     ŷ_t = P a_t                                    ∈ ℤ^V        (logits)
```

The same BEF codebook `P` of shape `(V, K_h)` is used as **both** the input
embedding lookup and the fixed output classifier. This is the natural
binary analogue of weight-tied embedding/output projections used in
standard transformer LMs, but achieved here by construction: prototypes
are already maximally separated ±1 codes.

Both `W_xh` and `W_hh` are visible binary weights stored as `brute.bit1`,
derived from integer hidden weights `H_xh, H_hh ∈ ℤ^{K_h × K_h}` via
`W = sign(H)`.

## BPTT BEP

**Trigger** (Eq. 1, per timestep): position `(b, t)` is triggered if the
margin `correct_logit − max_other` is below `r · K_h`.

**Backward** (Eq. 7) walks `t = T-1 ... 0`:

```
direct_t   = ρ^{target_t}                          if triggered_t
recur_t    = sign(W_hh^T (g_{t+1} ⊙ a*_{t+1}))     if a*_{t+1} exists
              with g_{t+1, i} = 1 iff |z_{t+1, i}| ≤ ν · 2K_h
a*_t       = sign(direct_t + recur_t)              # sign(0) = +1
```

The gate threshold is `ν · 2K_h` because each `z_t` aggregates `K_h + K_emb`
= `2K_h` ±1 inputs.

**Update** (Eq. 9, winner-takes-update inside neuron groups, broadcast
over all active `(b, t)` positions):

```
stability[μ, j] = a*_t[μ, j] · z_t[μ, j]            # joint stability over both matrices
mask M^μ        = per-group argmin of |stability| over neurons with stability < 0
H_xh ← H_xh + 2 · Σ_{μ, t}  (M^μ_t ⊙ a*_t^μ)^T  e_t^μ
H_hh ← H_hh + 2 · Σ_{μ, t}  (M^μ_t ⊙ a*_t^μ)^T  a_{t-1}^μ
```

Using the **joint** pre-activation `z_t = W_xh e_t + W_hh a_{t-1}` for
stability means a neuron is "misclassified" iff its full integrated
signal disagrees with the desired output — in which case BOTH incoming
matrices receive the same nudge. A reinforcement step (CP+R, Sec. 3.3)
drifts each non-zero weight further from zero with probability
`p_r √(2 / (π K_h))`.

## Running the demo

```bash
# Default config (~25 s/epoch on CPU; ~4 s/epoch on Apple-silicon MPS).
python train.py --epochs 6 --hidden 256 --vocab-cap 256 --seq-len 32 --r 0.2
```

Useful flags:

| flag             | default     | meaning                                              |
| ---------------- | ----------- | ---------------------------------------------------- |
| `--hidden`       | `256`       | recurrent state width `K_h` (must be divisible by group) |
| `--vocab-cap`    | `512`       | top-N most-frequent GPT-2 tokens (OOV folded into id 0) |
| `--seq-len`      | `64`        | BPTT context window `T`                              |
| `--r`            | `0.5`       | trigger margin (Eq. 1), as a fraction of `K_h`       |
| `--nu`           | `0.05`      | backward gating threshold (Eq. 5)                    |
| `--group-size`   | `4`         | neurons per group for winner-takes-update            |
| `--p-reinforce`  | `0.5`       | base reinforcement probability                       |
| `--weight-clip`  | `2048`      | clip on `|H|` (paper uses `2^{B-1}-1` with B=16)     |
| `--bef-iters`    | `auto`      | coord-flip iterations for the BEF codebook           |
| `--sample-every` | `0`         | if >0, greedy-decode a short sample every N epochs   |

## What "BEP works in the language domain" looks like

### Validation accuracy

Validation next-token accuracy on TinyShakespeare with a tiny config
(`--hidden 256 --vocab-cap 256 --seq-len 32 --batch-size 64 --r 0.2`,
6 epochs, ~20 s total on Apple-silicon MPS):

| metric                            | value     |
| --------------------------------- | --------- |
| uniform baseline (1 / 256)        | 0.0039    |
| unigram baseline (most-freq tok)  | 0.3267    |
| **BEP RNN, val top-1**            | **~0.39** |

So BEP clearly beats both the uniform and unigram baselines —
confirming that the BPTT BEP rule transfers from MLPs to RNNs in the
language setting.

### Sampled text — structured English

Scaling up to `--hidden 1024 --vocab-cap 1024 --seq-len 48 --r 0.2
--group-size 8 --top-k 8 --temperature 1.0` (15 epochs, ~3 min on
MPS), the model produces unmistakably Shakespeare-flavoured output.
Verbatim sample from a training run, prompted with `"First Citizen:"`:

> `\nth,illo bound yth,\nULIUS:\nULIier,\nth bound setillo,\nULIUS:\nth, thoughts bound\nULI bound bring setillo,...ULET:\nULIUS:\nULI y right,\n bound trust set little bound`

What the model has learned, *with 1-bit weights and 1-bit activations
throughout*:

* **Speaker-tag formatting** — `\nULIUS:`, `\nULET:`, `\nULIier`:
  Shakespeare character names (JULIUS, CAPULET) reconstructed from
  GPT-2 BPE pieces.
* **Act/scene markers** — `IV:` (Act IV).
* **Common Shakespearean lexicon** — "thoughts", "come hearts",
  "answer", "canst", "right", "bound", "trust", "set", "little",
  "thou", "your".
* **Sentence structure** — commas, paragraph breaks, colons in
  plausible positions.

The greedy attractor / repetition you see in samples is the
well-known low-capacity RNN failure mode; we ship a cheap fix
(`repetition_window=3` in :func:`_sample`) that bans the last few
emitted tokens from re-appearing.

### Caveats

* **Greedy decoding is harsh on binary RNNs.** Logits live in
  `[-K_h, K_h]` integer space and are very spiky. We default to
  ``--top-k 5 --temperature 1.0`` and ban the most-recent tokens from
  reappearing; pure argmax decoding falls into `\n\n\n…` attractors
  almost immediately.
* **OOV bucket dominates if not masked.** We fold rare tokens into
  compact id `0` for input, and replace OOV *targets* with
  ``IGNORE_INDEX`` so they don't drive BEP updates. The sampler also
  bans the OOV id by default. Disable with ``--no-mask-oov`` /
  ``--no-ban-oov-sampling``.
* **Coherence vs. baseline.** Bigger configs (1024×1024) produce more
  English-looking output but val top-1 hovers ~5% above the unigram
  baseline. Most of the headroom would come from things the paper
  also lists as future work — multi-layer stacks, attention, and
  adaptive `γ_l` / reinforcement scheduling.

## Running the tests

```bash
cd examples/bgpt
python -m pytest test_bep.py -v
```

The test suite covers:

* `forward` — output shapes / dtypes, ±1 invariant, agreement with a
  float ±1 reference, init state is the all-ones vector, `a_prev[t+1] ==
  a_curr[t]`;
* `step` — no-op when no position triggers, mutates `H_xh` / `H_hh`
  when triggered, respects the `IGNORE_INDEX = -1` mask;
* end-to-end "learns better than chance" on a tiny deterministic
  repeating-token task (>50% on an 8-cycle vocab=16 stream);
* `brute.bit1` integration — packed bit1 weight storage, int32 matmul
  output, weight cache invalidates after `H` changes.

## Why this is useful

Once you build on `brute.bit1` and BEP, you get a path to a full
language-model training stack where:

* **Packed storage**: every binary weight & activation lives as 1 bit in
  a 64-bit int64 word — 32× less memory than `float32`, 8× less than
  `int8`.
* **XNOR-popcount matmul**: every per-step recurrent update is a few
  packed popcounts — orders of magnitude fewer FLOPs than any STE-based
  QAT pipeline.
* **No autograd, no surrogate gradients**: BEP is a faithful binary
  analogue of the BPTT chain rule; nothing in this folder needs PyTorch
  autograd.

## Architecture notes — relation to transformers

The paper validates BEP on MLPs and binary RNNs and explicitly leaves
convolutional and *transformer-style* models for future work. This folder
covers the RNN case; the MLP variant in `examples/bep` is the building
block most directly analogous to the per-token FFN inside a transformer
block. Extending BEP to attention requires a binary analogue of the
attention softmax — still open research — but the per-cell BEP rule here
is already drop-in for the recurrent `nn.Linear`-equivalents of a
transformer's FFN, and the shared `(V, K_h)` BEF codebook is the binary
analog of weight-tied input/output embeddings.

# HÆMMR — a binary-first, concept-native language model on `brute`

A self-contained implementation of the **HÆMMR** architecture (the attached
whitepaper) — *Hyperdimensional Autoregressive Memory with Majority-vote
Reasoning* — trained natively in the Boolean domain with **BOLD** (Boolean Logic
Deep Learning, NeurIPS 2024). Every weight and activation is a single bit; the
forward pass is XNOR / popcount / majority-vote / Hamming-distance, stored and
computed on packed `brute.bit1` (uint64) tensors. No floating-point latent
weights anywhere — training is a logic test and a bit flip.

This is the paper's **Stage 1** target: *"build a single small binary expert
(BOLD reference implementation) … train on CPU."* The distributed
tracker / federated-MoE machinery (whitepaper §8) is intentionally **out of
scope**, as requested.

> Note on the dtype name: the request said `brute.int1`; the library's packed
> 1-bit dtype is actually `brute.bit1` — that is what is used throughout.

---

## What's faithful, and what stays bitwise

The whole point is to **never round-trip a 1-bit tensor through a byte
bool-tensor**. The hot path operates directly on the packed uint64 buffers:

| HÆMMR op | math | `brute` realisation | stays packed? |
|---|---|---|---|
| binding `a ⊗ b` | XNOR = bipolar `·` | `brute.fast.eq` / packed XOR+NOT | ✅ |
| residual merge | per-coord MUX | packed `&`, `\|`, `~` | ✅ |
| `<u,v> = D − 2·Ham` | signed dot | `xnor_popcount_matmul` (`a @ w`) | ✅ |
| threshold `sign(z)` | activation | int32 `z ≥ 0` → pack | ✅ (int→bit) |
| min-Hamming decode | argmax `<ĉ,E(t)>` | one `bit1 @ E` sweep | ✅ |

Integers appear **only** where the architecture itself is integer-valued, exactly
as the paper says ("the only non-bitwise element is the transient integer
accumulator"): the signed matmul pre-activations, the BSR vote accumulator
`A_i`, and the BOLD optimiser signals. **All stored parameters are
`brute.bit1`.** The ±1 *float views* used by the backward-pass signal matmuls are
transient caches recomputed after each flip — they are never the source of
truth (BOLD's signals are inherently real, so this is faithful, not a shortcut).

## Architecture (whitepaper §3–§6)

```
token t_i ──▶ E(t_i)              # 1-bit codebook lookup (V×D bits)
          ⊗ ρ^i(POS)             # cyclic-shift position binding
          ── input-bind (⊗ W) ──▶ concept c_i⁰
   ┌──────────────────────────────────────────────────┐   ×L blocks
   │  r = BSR(c)            # linear-time binary mixer  │
   │  c ← residual-merge(c, r)                          │
   │  h = HopfieldBank(c)   # binary latent attention   │
   │  c ← residual-merge(c, h)                          │
   │  m = channel-mix(c)    # binary MLP (XNOR matmuls) │
   │  c ← residual-merge(c, m)                          │
   └──────────────────────────────────────────────────┘
          ── out-bind (⊗ W) ──▶ next-token concept ĉ
          ── argmax_t <ĉ, E(t)> ──▶ token        # min-Hamming / Boltzmann decode
```

* **Concept ≠ embedding** (§4): `E` is the context-free 1-bit identity; the
  concept `ĉ` is a *computed* hypervector that may lie between codebook rows.
  Decoding projects it to the nearest realisable token.
* **BSR** (§5.1): `A_i = γ·A_{i-1} + (k_i⊗v_i)`, `S_i = sign(A_i)`, `r_i = q_i⊗S_i`
  with dense 1-bit Boolean projections for `k`, `v`, and `q`. This is a
  linear-time (O(n)) binary reimagining of linear attention / RetNet. Trained by
  the **exact O(n) reverse-scan adjoint** of the recurrence.
* **Latent Hopfield Bank** (§5.2): `M` learned key/payload slots; read =
  `sign(Σ_topk U_m)` over the top-k most similar keys (winner-take-all, not
  softmax) — softmax-free latent attention.
* **Decode** (§6): `argmax_t <ĉ,E(t)>`; to sample, `P(t) ∝ exp(<ĉ,E(t)>/T)`.

## Training — BOLD (§7)

Pure Boolean variation calculus, no gradients:

* signals `q` (weights) and `g` (backprop) via the XNOR chain rule (Eqs. 5–8),
  variance-scaled by `√(2/fan_out)`;
* flip rule (Eq. 9): flip `w` iff the accumulated signal **agrees in sign** with
  it;
* accumulator + auto-regularising β-plasticity (Eqs. 10–11) with reset-on-flip
  error feedback.

**One practical stabiliser:** from a random init the literal sign-only flip rule
can thrash, so the optimiser integrates signal in a small integer accumulator
and flips only after the agreeing evidence crosses `--threshold`. This preserves
BOLD's bit-flip update while letting small, consistent signals build over
ordinary single-machine batches.

## Files

```
model/
  vsa.py      bind / bundle / permute + bit1↔±1 helpers (packed)
  bold.py     BoldParam (1-bit weight + accumulator) and the flip-rule optimiser
  layers.py   BooleanLinear, DiagBind, gated ResidualMerge, TokenCodebook,
              HopfieldBank, BSR  — each with bitwise forward + BOLD backward
  model.py    HaemmrConfig / HaemmrLM — block stack, CE loss + backward, sampling
  data.py     WikiText + GPT-2 tokenizer, capped vocabulary, LM batches
  train.py    training CLI
  sample.py   sampling CLI (loads a checkpoint)
  tests/      pytest suite (forward-vs-reference, BOLD correctness, learning)
```

## Run

Assumes `brute` is importable (installed system-wide). Needs `transformers` and
`datasets` for the GPT-2 tokenizer and WikiText.

```bash
# train a small demo on WikiText-2 (CPU, a few minutes)
python train.py --steps 1500 --D 1024 --layers 2 --vocab-cap 2048 --seq-len 64

# faster smoke run
python train.py --steps 500 --D 512 --layers 2 --vocab-cap 1024 \
                --max-train-tokens 500000 --seq-len 48

# sample from the checkpoint
python sample.py --ckpt haemmr.pt --prompt "The history of" --n 60 --temperature 0.8

# tests (≈2 s)
python -m pytest tests/ -q
```

Key flags: `--D` (concept dim — bigger ⇒ better VSA geometry, the paper uses
8k–16k), `--layers`, `--d-ff`, `--slots`/`--top-k` (Hopfield), `--eta` /
`--threshold` (BOLD), `--device {cpu,mps,cuda}`.

## What to expect

This is a **small local demo of a speculative architecture**, not a strong LM.
On a ~3 M-bit (≈0.36 MB) model, 500 CPU steps on WikiText-2 takes it from
perplexity ~1700 → ~260 and accuracy 0.0015 → ~0.07 — i.e. it decisively beats
the uniform baseline (≈70×) and reaches the unigram baseline, generating the
corpus's high-frequency tokens. Coherent text needs what the whitepaper itself
flags as the open empirical questions: larger `D` (the quasi-orthogonality
geometry), more blocks/slots, and much more training. The deliverable here is a
**correct, fully bitwise, end-to-end-trainable** reference — verified by the
test suite (forward bit-exact against a ±1 reference; BOLD provably reduces loss;
the stack learns a deterministic next-token map to 100 %).

## Deliberate simplifications

* No federated tracker / Mixture-of-Experts / ternary voting (§8) — out of scope.
* BSR decay `γ` is a **fixed** multi-timescale spread (the paper's one
  low-precision concession; §11 lists its bit-width as open). Not learned.
* Residual merges are **gated MUXes** (zero-init-style identity at start) rather
  than raw 3-way majority — this is what makes a deep binary residual stack
  trainable from scratch.
* Hopfield **keys** train by a Hebbian rule and **payloads** by the loss signal;
  the top-k selection is hard (zero query gradient a.e.), so the query path
  learns through the other branches. Dynamic Hopfield writes (§5.2) are omitted
  (the paper flags them as a stability risk; static slots are the safe default).
* The input concept uses `E(t)⊗ρ^i(POS)`; the running-state binding `⊗ S_{i-1}`
  from §4 is provided functionally by the BSR read + residual rather than as a
  mutual recurrence (keeps the graph clean and acyclic).

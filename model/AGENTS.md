# Haemmr Agent Guide

## Project

`model/` is the Haemmr binary language-model reference implementation. It is a
consumer of the root `brute` packed-bit tensor library. Treat `brute` kernel
changes and Haemmr model changes as related but separate layers.

The model is binary-first: parameters and forward activations are
`brute.bit1`, Boolean linears use XNOR/popcount, and training uses BOLD
bit-flip updates rather than floating-point latent weights.

## Layout

- `model.py`: `HaemmrConfig`, `HaemmrLM`, block stack, loss/backward, metrics,
  generation, and checkpoint state.
- `layers.py`: Boolean linear projections, token codebook, residual gates,
  BSR, episodic slot memory, Hopfield bank, and channel mixing.
- `bold.py`: BOLD Boolean parameters and optimizer.
- `vsa.py`: packed VSA operations, position codes, Hamming similarity, and BEF
  codebook initialization.
- `data.py`: WikiText/GPT-2 compact-vocabulary data utilities.
- `train.py`: WikiText training CLI.
- `sample.py`: checkpoint sampling CLI.
- `tests/`: maintained pytest coverage for the current model.

Generated reports, historical comparison artifacts, ad hoc probes, and demo
scripts are intentionally not part of the cleaned model tree.

## Architecture

The decoded stream is a position-free concept vector. Positions are used only
inside memory address lanes. A block combines:

- delta BSR for compressed discourse state;
- episodic slot memory for exact active-window lookup;
- a static Hopfield bank for learned priors;
- a binary channel-mix MLP;
- gated binary residual merges.

The decoder projects the output concept into the lexical frame and scores it by
Hamming similarity against the shared codebook, with optional semantic rerank.

## Packed Hot Paths

- `vsa.sign_to_bit1` uses `brute.fast.sign` and `brute.pack_sign` for int32 and
  float32 thresholds (BSR accumulator, general conversions).
- `BooleanLinear` uses `brute.fast.matmul_sign` (fused, no int32 intermediate)
  when `boundary_nu` is `None`; falls back to `brute.fast.matmul` + sign when
  `boundary_nu` is set (needs `|z|` for the eligibility gate).
- Multi-slot `HopfieldBank` uses `brute.fast.majority` (bit-sliced packed vote)
  instead of integer tally + sign.
- BSR forward uses `brute.fast.bsr_scan`, which keeps read/state/gate packed.
- Episodic batched training uses the two-tier chunked forward (O(n·C)) via
  `_banded_sim` per chunk window; no (B, n, n) score matrix.
- Episodic streaming uses `brute.fast.episodic_causal_search` (fused Hamming
  scan + top-1 + payload gather over the Tier-1 ring buffer).

BOLD backward still materialises real-valued signal views where needed. That is
training state, not canonical parameter storage.

## Tests

Run from the repository root:

```bash
./.venv/bin/python -m pytest model/tests -q
```

The current suite covers BOLD, VSA, layer behavior, packed BSR parity,
episodic streaming parity, model forward/loss/generation/checkpoint behavior,
and efficiency contracts. Add tests near the behavior being changed; do not
restore deleted probe scripts as test substitutes.

## Known Limits

- Decode rerank scores the full compact vocabulary.
- The inline BEF initializer is for small/local runs; large vocabularies need
  a precomputed codebook.
- CUDA kernels are implemented but not locally tested in this Mac workspace.
- Streaming generation (`generate()`) still recomputes the full context each
  token; wiring block-level BSR/episodic streaming state into autoregressive
  inference is deferred.

## Working Rules

Keep `/model` focused on the current Haemmr architecture. Do not reintroduce
historical version comparisons, transformer baselines, generated CSV/JSON
artifacts, or one-off demo scripts. Prefer small, explicit model changes and
run `model/tests` before handoff.

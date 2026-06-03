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
  float32 thresholds.
- `BooleanLinear` uses `brute.fast.matmul`.
- BSR forward uses `brute.fast.bsr_scan`, which keeps read/state/gate packed.
- Episodic streaming stores key, position, and payload ring-buffer state as
  packed bit buffers.

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

- Batched episodic training constructs `(B, n, n)` scores and is quadratic in
  sequence length.
- Multi-slot Hopfield and episodic votes still materialise integer tallies.
- Decode rerank scores the full compact vocabulary.
- The inline BEF initializer is for small/local runs; large vocabularies need
  a precomputed codebook.
- CUDA kernels are implemented but not locally tested in this Mac workspace.

## Working Rules

Keep `/model` focused on the current Haemmr architecture. Do not reintroduce
historical version comparisons, transformer baselines, generated CSV/JSON
artifacts, or one-off demo scripts. Prefer small, explicit model changes and
run `model/tests` before handoff.

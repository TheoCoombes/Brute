# Haemmr Agent Guide

## Project

`model/` is the Haemmr binary language-model reference implementation. It is a
consumer of the root `brute` packed-bit tensor library. Treat `brute` kernel
changes and Haemmr model changes as related but separate layers.

The model is binary-first: parameters and forward activations are
`brute.bit1`, Boolean linears use XNOR/popcount, and training uses BOLD
bit-flip updates rather than floating-point latent weights.

## Layout

- `model.py`: `HaemmrConfig`, `HaemmrLM`, the clipped integer concept highway,
  the block stack, loss/backward, metrics, generation, and checkpoint state.
- `layers.py`: Boolean linear projections, the fixed/learned token codebook,
  delta erase/write BSR, the episodic causal-ring memory, the Hopfield prior
  bank, and channel mixing.
- `bold.py`: BOLD Boolean parameters and optimizer.
- `vsa.py`: packed VSA operations, position codes, Hamming similarity, and BEF
  codebook initialization.
- `data.py`: WikiText/GPT-2 compact-vocabulary data utilities.
- `train.py`: WikiText training CLI.
- `sample.py`: checkpoint sampling CLI.
- `tests/`: maintained pytest coverage for the current model.

Generated reports, historical comparison artifacts, ad hoc probes, and demo
scripts are intentionally not part of the cleaned model tree.

## Architecture (v3)

The decoded stream is a position-free concept vector. Positions are used only
inside the episodic address lane. The three non-negotiables are: one canonical
linear-time exact-memory path, a non-destructive vertical transport path, and a
strict split between compressed / exact / prior memory.

Every block carries two vertical states: a public binary concept `c = sign(u)`
and a private **clipped integer concept highway** `u ∈ [-S, S]`. All branches
read the same `c` and emit a binary proposal; the block accumulates them into the
highway with fixed power-of-two scales and binarises once:

    u' = clip_S( λ·u + Σ_r α_r · proposal_r ),   c' = sign(u')

There is **no majority-vote residual merge** — that is explicitly forbidden as
the inter-layer transport. The branches are:

- delta erase/write BSR — compressed discourse (separate per-channel-group erase
  and write strengths over a power-of-two decay palette);
- episodic causal-ring memory — exact in-window recall (one canonical fixed-width
  ring with read-before-write; chunked batched search == streaming search);
- a static Hopfield bank — learned global priors (never a runtime write target);
- a binary channel-mix MLP.

The episodic branch carries a dominant `alpha_epi` so that exact recall can
override the carried concept when it fires. Multiscale horizons grow the episodic
window and slow the BSR decay with depth. The decoder projects `c^L` into the
lexical frame, scores it by Hamming distance against the **fixed** lexical
codebook, reranks with the **learned** semantic prototype bank, and optionally
seeds the shortlist with the final-block episodic retrieval.

## Packed Hot Paths

- `vsa.sign_to_bit1` uses `brute.fast.sign` / `brute.pack_sign` for int32 and
  float32 thresholds (concept highway, BSR accumulator, general conversions).
- `BooleanLinear` uses `brute.fast.matmul_sign` (fused, no int32 intermediate)
  when `boundary_nu` is `None`; falls back to `brute.fast.matmul` + sign when
  `boundary_nu` is set (needs `|z|` for the eligibility gate).
- Multi-slot `HopfieldBank` uses `brute.fast.majority` (bit-sliced packed vote).
- The clipped integer concept highway is the **one** activation concession: each
  block unpacks its branch reads to ±1 to accumulate the int score field
  (`O(layers·branches)` unpacks, never `O(n)`/`O(V)`). Learned weights never
  leave the packed domain in the forward path.
- Delta BSR forward uses `brute.fast.bsr_delta_scan` on CPU: K/V/Q and the
  state/gate/read outputs stay packed while the kernel walks bit lanes with a
  bounded integer accumulator. The older `brute.fast.bsr_scan` single-gate op is
  retained as a library primitive.
- Episodic batched training is the chunked ring search (O(n·C)) via `_banded_sim`
  per chunk window; no (B, n, n) score matrix and no Tier-2 register cache.
- Episodic streaming uses `brute.fast.episodic_causal_search` (fused Hamming
  scan + top-1 + payload gather over the causal ring buffer).
- BooleanLinear and decode backward on CPU use `brute.fast.ternary_matmul` for
  ternary signal × packed bit1 rows, avoiding full ±1 weight/codebook views.
- BOLD updates use `brute.fast.bold_update_packed`: `q` is a transient integer
  ternary variation buffer, `m` is the integer momentum/accumulator, and weight
  bits are flipped in packed storage with reset-on-flip.

Real-valued tensors remain only at explicit surrogate boundaries (CE reporting,
episodic soft-attention adjoints, and non-CPU fallbacks), not as latent weight or
optimizer state. The lexical codebook is fixed (not a BOLD parameter); only the
concept projection and semantic bank train.

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

- Exact recall is window-bounded by design (fixed ring). A sparse beyond-window
  extension (e.g. fast-weight product-key memory) is deliberately out of scope
  for v3; the v2.5 Tier-2 LSH register cache was removed.
- Decode rerank scores the full compact vocabulary.
- The inline BEF initializer is for small/local runs; large vocabularies need
  a precomputed (fixed) codebook.
- The concept highway stores `u` as an int tensor clamped to ±S; production
  should pack it bit-sliced (int4/int5). The math is unchanged.
- CUDA kernels are implemented but not locally tested in this Mac workspace.
- Streaming generation (`generate()`) recomputes the full context each token;
  wiring block-level highway/BSR/episodic streaming state into autoregressive
  inference is deferred.

## Working Rules

Keep `/model` focused on the current Haemmr architecture. Do not reintroduce
historical version comparisons, transformer baselines, generated CSV/JSON
artifacts, or one-off demo scripts. Prefer small, explicit model changes and
run `model/tests` before handoff.

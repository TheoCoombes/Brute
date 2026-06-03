# HÆMMR Model (v3)

This directory contains the current HÆMMR reference implementation: a
binary-first, concept-native autoregressive language model built on
`brute.bit1`. The code is intentionally small enough to iterate on locally
while keeping the core constraints visible:

- packed 1-bit weights and a **clipped integer concept highway** as the one
  vertical-transport activation concession (`c = sign(u)`, `u ∈ [-S, S]`);
- XNOR/popcount Boolean linears through `brute.fast.matmul`;
- BOLD bit-flip optimisation with no floating-point latent weights;
- position-free decoded concepts, with positions restricted to the episodic
  address lane;
- one canonical linear-time exact-memory path (a fixed-width causal ring),
  delta erase/write BSR for compressed discourse, static Hopfield priors, and
  binary channel mixing — combined by the highway, **not** a majority-vote merge.

Federated, MoE, tracker, and swarm machinery are out of scope for this package.

## Files

| File | Purpose |
|---|---|
| `model.py` | `HaemmrConfig`, clipped integer concept highway, block stack, forward/loss/backward, metrics, generation, checkpoint state |
| `layers.py` | Boolean linears, fixed/learned codebook, delta erase/write BSR, episodic causal-ring memory, Hopfield bank |
| `bold.py` | BOLD parameters and integer flip-accumulator optimiser |
| `vsa.py` | Packed VSA binding, position codes, Hamming similarity, BEF codebook initialisation |
| `data.py` | WikiText/GPT-2-tokenizer compact-vocab data utilities |
| `train.py` | WikiText training entry point |
| `sample.py` | Checkpoint sampling entry point |
| `tests/` | Model, layer, VSA, BOLD, and efficiency-contract tests |

## Architecture

```text
token ids
  -> FIXED lexical bit codebook E_lex(t)
  -> input bind                       # position-free concept c^0,  u^0 = ±1
  -> [ c = sign(u)                    # public concept; every branch reads it
       delta erase/write BSR          # compressed discourse
       episodic causal ring           # exact in-window recall (read-before-write)
       Hopfield priors                # static learned priors
       binary channel mix
       u' = clip_S(λ·u + Σ α_r·r_r)   # clipped integer concept highway ] x L
  -> c^L = sign(u^L)
  -> lexical Boolean projection -> Hamming decode against FIXED E_lex
  -> + semantic rerank against LEARNED E_sem
  -> + optional episodic shortlist seeding
```

`EpisodicSlotMemory` is one canonical fixed-width causal ring: read-before-write,
direct Hamming search over the active window, no dense `(B, n, n)` score tensor
and no Tier-2 register cache. Batched training is a chunked implementation of the
**same** ring search (O(n·C)); streaming uses `episodic_causal_search` over the
ring. Exact recall is window-bounded by design.

`BSR` uses dense Boolean projections for key/value/query and edits its own
prediction with a decoupled **erase/write delta** (separate per-channel-group
erase and write strengths) over the power-of-two `decay_shifts` palette. The
CPU forward path uses a packed `bsr_delta_scan` kernel over a bounded integer
accumulator, returning packed read/state/gate buffers. BOLD backward uses
ternary Boolean variation and packed ternary-matmul kernels for Boolean linears
and decoders on CPU; real-valued tensors are kept to explicit surrogate
boundaries, not to latent weights or optimizer state.

The **concept highway** carries an integer score field `u`; `c = sign(u)` is the
public concept. Each block's branches all read `c` and emit a binary proposal,
which the highway accumulates with fixed power-of-two scales (`alpha_*`) and an
identity carry (`highway_carry`), clipping to `±highway_clip` and binarising once
per block. The episodic branch uses a dominant `alpha_epi` so exact recall can
override the carried concept. Multiscale horizons grow the episodic window and
slow the BSR decay with depth.

## Training

Run from the repository root after installing `brute` in editable mode:

```bash
./.venv/bin/python -m pytest model/tests -q

./.venv/bin/python model/train.py \
  --steps 500 --D 512 --layers 2 --vocab-cap 1024 \
  --max-train-tokens 500000 --seq-len 48 --no-position
```

Common flags:

- `--D`, `--layers`, `--d-ff`: concept dimension, depth, and channel-mix width.
- `--slots`, `--top-k`: static Hopfield prior bank.
- `--epi-window`, `--epi-chunk`, `--epi-read-k`: episodic ring width, chunk size,
  and read width.
- `--epi-bonus`: episodic shortlist-seeding weight at decode time.
- `--highway-clip`, `--highway-carry`, `--highway-nu`: concept-highway bound `S`,
  identity carry `λ`, and boundary gate.
- `--alpha-epi`/`--alpha-bsr`/`--alpha-hop`/`--alpha-ff`: fixed power-of-two
  branch scales (`alpha-epi` defaults dominant for reliable exact recall).
- `--no-multiscale`: disable per-depth window/decay scaling.
- `--no-bsr`: disables BSR for packed-only ablation profiles.
- `--no-position`: uses neutral all-ones episodic position codes.
- `--no-structured-codebook`: uses random (fixed) token codes instead of BEF.
- `--sem-weight`, `--sem-flip-scale`: semantic rerank weight (`0` disables) and
  semantic-bank flip rate.
- `--boundary-nu`, `--label-smoothing`, `--flip-dropout`: training stabilisers.
- `--eta`, `--threshold`, `--m-clip`: BOLD accumulator controls.
- `--ckpt`, `--last-ckpt`: best-validation and final checkpoint paths.

Sample from a checkpoint:

```bash
./.venv/bin/python model/sample.py \
  --ckpt haemmr.pt --prompt "The history of" --n 60 --temperature 0.8
```

## Tests

The maintained suite is:

```bash
./.venv/bin/python -m pytest model/tests -q
```

Coverage includes VSA primitives, BOLD flip rules, layer forward/backward
wiring, packed BSR recurrence parity, episodic streaming parity, model
forward/loss/generation/checkpoint behavior, and source-level efficiency
contracts.

## Known Limits

- Exact recall is window-bounded by design (fixed causal ring); there is no
  beyond-window dynamic store in v3.
- Decode rerank scores the full compact vocabulary rather than a fused
  shortlist.
- The concept highway stores `u` as an int tensor clamped to ±S; production
  should pack it bit-sliced (int4/int5).
- The inline BEF initialiser is capped for local startup; production-scale
  vocabularies should load a precomputed (fixed) codebook.

# HÆMMR Model

This directory contains the current HÆMMR reference implementation: a
binary-first, concept-native autoregressive language model built on
`brute.bit1`. The code is intentionally small enough to iterate on locally
while keeping the core constraints visible:

- packed 1-bit weights and activations for the forward path;
- XNOR/popcount Boolean linears through `brute.fast.matmul`;
- BOLD bit-flip optimisation with no floating-point latent weights;
- position-free decoded concepts, with positions restricted to memory address
  lanes;
- recurrent BSR discourse state, episodic in-window recall, static Hopfield
  priors, and binary channel mixing.

Federated, MoE, tracker, and swarm machinery are out of scope for this package.

## Files

| File | Purpose |
|---|---|
| `model.py` | `HaemmrConfig`, block stack, forward/loss/backward, metrics, generation, checkpoint state |
| `layers.py` | Boolean linears, codebook, residual gates, BSR, episodic memory, Hopfield bank |
| `bold.py` | BOLD parameters and integer flip-accumulator optimiser |
| `vsa.py` | Packed VSA binding, position codes, Hamming similarity, BEF codebook initialisation |
| `data.py` | WikiText/GPT-2-tokenizer compact-vocab data utilities |
| `train.py` | WikiText training entry point |
| `sample.py` | Checkpoint sampling entry point |
| `tests/` | Model, layer, VSA, BOLD, and efficiency-contract tests |

## Architecture

```text
token ids
  -> lexical bit codebook E(t)
  -> input bind
  -> [ delta BSR
       episodic slot memory
       Hopfield priors
       binary channel mix ] x L
  -> output bind
  -> lexical Boolean projection
  -> Hamming decode against E
  -> optional semantic rerank
```

`EpisodicSlotMemory` writes one packed payload per token and retrieves from the
active causal window by Hamming address. The current batched training form still
forms a `(B, n, n)` score tensor, so it remains quadratic in sequence length.
The streaming form stores key, position, and payload state as packed bit buffers
and reads only the active ring-buffer window.

`BSR` uses dense Boolean projections for key/value/query, packed association
bits, and a fused `brute.bsr_scan` kernel for the forward recurrence. The scan
uses the power-of-two shift palette in `decay_shifts` and returns packed
read/state/gate buffers. BOLD backward lazily materialises the real-valued
views it needs for flip signals.

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
- `--epi-slots`, `--epi-read-k`: episodic recall window and read width.
- `--no-bsr`: disables BSR for packed-only ablation profiles.
- `--no-position`: uses neutral all-ones episodic position codes.
- `--no-structured-codebook`: uses random token codes instead of BEF.
- `--sem-weight`: semantic rerank weight; `0` disables it.
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

- Batched episodic training is still quadratic in sequence length because it
  constructs dense causal score matrices.
- Multi-slot Hopfield and episodic reads still use integer vote materialisation.
- Decode rerank scores the full compact vocabulary rather than a fused
  shortlist.
- The inline BEF initialiser is capped for local startup; production-scale
  vocabularies should load a precomputed codebook.

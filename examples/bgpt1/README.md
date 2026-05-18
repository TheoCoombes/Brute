# BGPT-1 (Phase-1 implementation)

A fully 1-bit decoder-only transformer language model trained under the
streaming flip-rule optimizer. Implemented entirely on top of the
**brute** 1-bit tensor library — every weight, every activation, and
every attention score is one bit. No FP weights, no gradients, no
optimizer moments.

## What lives where

The reusable building blocks are kept in **brute itself** (so the
library stays general and the BGPT-1 example is the only place
model-specific composition happens):

| File | Purpose |
|------|---------|
| `brute/nn/binary_norm.py` | Bit-balanced sign + per-element near-boundary gate flag |
| `brute/nn/binary_embedding.py` | Bit1 lookup table over a vocabulary |
| `brute/nn/binary_positions.py` | Multi-scale parity-of-prefix positional codes |
| `brute/nn/binary_attention.py` | Two-stage XNOR-popcount attention with ALiBi bias + 2-bit zero-band map |
| `brute/nn/binary_ffn.py` | Two-layer bit1 FFN |
| `brute/nn/binary_block.py` | Parallel-residual transformer block with stochastic depth |
| `brute/nn/binary_lm_head.py` | Tied LM head + INT8 bias + signed-integer error signal |
| `brute/nn/linear.py` | `BruteLinear` (low-level) + `BinaryLinear` (with bit-balance) |
| `brute/optim/flip_rule.py` | Streaming flip-rule optimizer with β regularization |

BGPT-1-specific glue:

| File | Purpose |
|------|---------|
| `examples/bgpt1/model.py` | `BGPT1` model class with custom `forward` + `backward_step` |
| `examples/bgpt1/data.py` | Tokenizer (GPT-2 BPE or char-level) + Tiny Shakespeare loader |
| `examples/bgpt1/train.py` | Training script with diagnostics and CSV logging |

## Quick start

```bash
# Smoke test (50 steps, char-level Tiny Shakespeare)
python -m examples.bgpt1.train \
    --dim 64 --n-heads 4 --n-layers 2 --context 32 \
    --batch-size 16 --steps 200 --log-every 20 \
    --tokenizer char --t-init 0.2 --target-flip-rate 0.002 \
    --decisive-threshold 0.1 --n-ref-scale 16
```

Logs are written to `examples/bgpt1/.runs/<timestamp>.csv` with the
metrics specified in BGPT-1 spec §9 (loss, entropy, tie_rate, T_base,
mu_mean, f_mean, flips per step, total flips, bias updates).

## What works (Phase 1 verified)

- End-to-end forward pass through embedding → multi-scale positions →
  parallel-residual transformer blocks → tied LM head — all bit1 with
  one-bit storage and XNOR-popcount kernels under the hood.
- Streaming flip-rule optimizer with per-layer flip-rate EMA, per-output-row
  2-bit confidence counter, cosine-annealed base temperature, per-layer
  multiplicative temperature feedback.
- INT8 LM head bias with bounded-accumulator updates (spec §3.7).
- Signed-integer (4-bit-magnitude) output error signal (spec §3.8).
- Custom backward that walks the model in reverse, computes votes per
  layer in row-blocks, applies stochastic XOR flips, and propagates a
  bit-balanced error to the layer below.
- 27 dedicated unit tests for the new modules; all 1976 brute tests
  pass.
- Loss decreases monotonically on the toy memorize-a-fixed-batch task
  in `test_copy_task_loss_decreases`. Loss drops from ~19 → ~7 in 200
  steps on Tiny Shakespeare (char-level, 100k binary params).

## What stays on packed bit1

The forward and backward keep all binary state on packed int64 buffers.
Concretely:

- **Stage 1 (QK similarity)** — single ``bit1 @ bit1`` matmul →
  ``S = K − 2·Hamming`` int32 (kernel output). ALiBi bias is added as
  int32 from a precomputed per-head ``(C, C)`` table. Causal mask
  applied via ``brute.where``.
- **Stage 2 (aggregation)** — *exact* via the algebraic identity
  ``Y = (matmul(A_sign, V) − matmul(A_active XOR A_sign, V)) / 2``.
  Both matmuls are pure XNOR-popcount on packed buffers; the only
  intermediate is the int32 result of each. No int8 ternary, no
  unpack-to-bool. Proof:

  ``B := A_active XOR A_sign`` equals ``A_sign`` at inactive
  positions and ``¬A_sign`` at active. So
  ``matmul(A_sign, V) − matmul(B, V)`` is exactly twice the sum over
  active positions, with inactive contributions cancelling. See
  ``brute/nn/binary_attention.py::BinaryAttention._stage2`` and the
  test ``test_stage2_exact_matches_ternary_matmul``.
- **Parallel residual** — ``sign(x + attn + ffn) =
  majority(x, attn, ffn)`` computed as three packed AND + two packed
  OR. No bool round-trip in the forward residual; the residual
  backward uses the same primitive.
- **Token ⊙ position embedding** — single ``~XOR`` on packed bit1.
- **Flip step** — the weight stays packed. We snapshot the bool view
  once per layer-step (lazy unpack), compute a bool flip-mask in row
  blocks, pack the mask once, and XOR into the packed weight buffer
  in one shot.
- **τ (zero-band threshold)** — int32 per head, learned via a bounded
  integer accumulator: each batch the per-head ``att_dec − target`` is
  added to the accumulator; on crossing ±``tau_accum_threshold`` the
  buffer ticks ``τ`` by ±1. See
  ``BinaryAttention.tau_feedback_step`` and the test
  ``test_tau_feedback_moves_threshold``.

The only places we deliberately leave the bit1 domain are:

- Integer pre-activations feeding ``bit_balance`` (we need the median,
  an integer op).
- The LM-head signed-int error path (``signed_int_error`` produces
  int8 in [-7, 7], which goes through one plain int matmul on its way
  back to the binary body).
- The BOLD-style int8 LM-head bias (the only non-binary trainable
  weight in the model, by spec design).

## BEP gate masking (Phase 3, applied)

Each binary linear (`q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`,
`fc2`) emits a per-output bit1 gate flag at training time — True iff the
output's centered pre-activation magnitude is below ``ν · d`` (i.e.
saturated, near the decision boundary). The forward tape carries those
flags as bit1 (one bit per neuron). The backward `flip_step` and
`propagate_error` apply the gate via the same
``Y = (matmul(err_sign, x) − matmul((~gate) XOR err_sign, x)) / 2``
identity used by attention stage 2 — two packed XNOR-popcount matmuls,
no unpack, no full-(B, m) bool intermediate. See
``brute/optim/flip_rule.py::FlipRule._vote_block_bit1`` and the tests:
- ``test_gate_zeroes_saturated_positions_in_vote`` — fully-saturated
  gate ⇒ zero vote everywhere.
- ``test_gate_passes_through_unsaturated`` — gate False everywhere ⇒
  identical to ungated vote.
- ``test_propagate_error_with_gate_zeroes_saturated`` — gate fully on
  ⇒ zero pre-sign score.

## Remaining Phase-3/4 approximations (vs. spec)

1. **Attention backward through stage 1 + stage 2 is a straight-through
   STE** — we propagate the err at the head-aggregated output to all
   three Q / K / V projection backward paths. A faithful sign-
   aggregation backward (separate dQ / dK / dV through the attention
   map) is a Phase-4 extension; the propagation primitive itself is in
   place.
2. **ALiBi slopes are fixed** — by spec; only ``τ`` is trainable (via
   the bounded-accumulator feedback on per-head ``att_dec``).

## Memory budget — training within ~10 % of inference

Everything in the forward+backward stays on packed bit1 except:
- integer pre-activations feeding `bit_balance` (need int median);
- the signed-int output error (sparse int8);
- the int8 LM-head bias and its int32 accumulator (BOLD-style).

Concrete memory-friendly choices:
- **Vote**: streamed per row-block of width `B_v`. Only one
  `(B_v, n) int32` vote scratch lives at a time.
- **Flip mask**: packed to bit1 per row-block and XOR'd into the
  row-block of `W._packed_buf` in one shot — no full `(m, n) bool`
  allocation.
- **Weight read in the descent comparison**: `W[row_slice].bool()`
  unpacks only the current row-block, not the full weight.
- **Signed-int err matmul** (LM-head boundary only): chunked over
  weight rows (`signed_chunk`, default 1024) and the batch axis to cap
  the float32 unpack temporary at a few MB.
- **Causal mask + attention map**: bit1 (1/8 of bool).
- **Activation tape**: per block we store `x_in` (block input, bit1),
  the FFN hidden `h` (bit1), and six gate-flag tensors (all bit1) —
  no int32 pre-activations, no `attn_out` / `ffn_out` duplicates.

The integration test ``test_training_memory_within_budget`` asserts the
activation+gate tape stays within `6×` of the total weight bytes, which
is the natural overhead from carrying one block input + one hidden +
six gates per block (all bit1).

## Hyper-parameter guidance (Phase 1, toy scale)

The defaults in `BGPT1Config` are tuned for a ~100K-param toy model on
Tiny Shakespeare. For your own runs:

- **`t_init = 0.2, t_final = 0.02`** — lower than the spec's 0.5
  default. The spec's value over-flips at toy scale; halving it gives
  stable loss curves.
- **`target_flip_rate = 0.002, decisive_threshold = 0.1`** — adjust
  these together. A higher `decisive_threshold` means the per-row
  confidence counter ticks up only when vote magnitude is large
  relative to batch size; useful with small batches.
- **`n_ref_scale = 16`** — the script computes `n_ref = numel /
  n_ref_scale` to keep per-batch flip probability around
  `T_base / n_ref_scale`. Halve the scale to flip twice as fast.

## Diagnostics

`brute.nn.BinaryAttention.forward(return_diag=True)` returns the spec
§9 metrics `att_dec`, `att_sparsity`, `scores_abs_mean`. The training
script logs `mu_mean`, `f_mean`, flip counts, bias updates, and the
cross-entropy / softmax entropy / top-2 tie rate.

Inspect a run with:

```python
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("examples/bgpt1/.runs/<your_run>.csv")
df.plot(x="step", y="loss")
```

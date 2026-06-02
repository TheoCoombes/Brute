# HÆMMR Benchmark Matrix Report

Generated from:

```bash
python model/benchmark_matrix.py \
  --haemmr-steps 80 \
  --transformer-steps 300 \
  --wikitext-csv /private/tmp/haemmr_runs/final_best_200.csv \
  --json-out /private/tmp/haemmr_benchmark_matrix.json
```

All numbers below are from that run unless stated otherwise. The benchmark is
small by design: it is intended to expose architectural failure modes, not claim
final model quality.

## Executive Summary

HÆMMR's binary/VSA substrate is behaving correctly: binding, unbinding, position
recovery, small VSA records, BSR streaming, and controlled Hopfield lookup all
pass cleanly. The architecture also solves a transformer-style induction task at
length 64.

The major gap is selective content retrieval. HÆMMR fails marker/copy tasks that
a tiny causal Transformer solves exactly. This is the clearest evidence that the
current BSR + static Hopfield path is not yet equivalent to transformer
attention. It can learn associative sequence patterns, but not robust arbitrary
key-value retrieval under distractors.

## Benchmark Matrix

### Geometry and Decode Space

| Test | Result | Interpretation |
|---|---:|---|
| Bind then unbind bit accuracy | 1.000 | VSA binding algebra is exact. |
| 5-record bundle retrieval accuracy | 1.000 | Small VSA content-addressed records work at D=1024. |
| Position bind/unbind bit accuracy | 1.000 | Position roles are invertible when explicitly unbound. |
| Positioned raw decode accuracy | 0.000 | Directly decoding `E(t)⊗pos` against raw `E` is wrong. |
| Positioned unbound decode accuracy | 1.000 | `next_unbind` fixes the position/codebook mismatch. |
| Mean abs random similarity, D=512 | 0.0336 | Cross-talk/noise floor at smaller D. |
| Mean abs random similarity, D=4096 | 0.0134 | Noise falls with dimension as expected. |

### Components

| Component | Metric | Result | Interpretation |
|---|---|---:|---|
| BSR | streaming vs batched mismatch | 0.000 | Streaming recurrence matches batched forward. |
| BSR | persistent-prefix late-read difference | 0.516 | BSR state remains content-sensitive over length 64 when signal persists. |
| Hopfield | controlled top-1 payload bit accuracy | 1.000 | Static slot retrieval works when keys/payloads are set directly. |
| Model | bit parameters, D=256 one-block bench | 525,568 bits | About 0.066 MB stored parameters. |

### Synthetic Attention: HÆMMR vs Tiny Transformer

| Task | Length | HÆMMR Acc | Transformer Acc | Gap | Interpretation |
|---|---:|---:|---:|---:|---|
| Induction `A B ... A -> B` | 16 | 1.000 | 1.000 | 0.000 | Matched. |
| Induction `A B ... A -> B` | 64 | 1.000 | 1.000 | 0.000 | Matched at long toy context. |
| Marker retrieval | 16 | 0.525 | 1.000 | -0.475 | Near chance for binary marker. |
| Marker retrieval | 64 | 0.504 | 1.000 | -0.496 | No robust long retrieval. |
| Copy retrieval | 16 | 0.126 | 1.000 | -0.874 | Chance for 8-way copy. |
| Copy retrieval | 64 | 0.126 | 1.000 | -0.874 | No arbitrary payload copying. |

Transformer losses were near zero on all synthetic tasks. HÆMMR losses were:
induction length 64 = 0.0075, marker length 64 = 0.7515, copy length 64 =
2.7310.

### Efficiency

Forward profile, D=256, one layer, batch=4, seq=32, 5 iterations:

| Metric | Result |
|---|---:|
| Forward tokens/s | 105,058 |
| `brute.fast.matmul` calls | 35 |
| `unpack_pm1` calls | 90 |
| `to_bit1` pack calls | 35 |

The forward path is using packed matmul. The remaining efficiency issue is that
BSR and Hopfield still unpack to ±1 for integer state/majority construction.
That is currently correct but not optimal.

### Tiny WikiText

Best small run available:

| Metric | Step 100 | Step 200 |
|---|---:|---:|
| Validation loss | 5.4999 | 5.5251 |
| Validation perplexity | 244.67 | 250.92 |
| Validation accuracy | 0.0106 | 0.0073 |

Config: D=256, one block, vocab cap 512, `position_decode=next_unbind`,
normal batches, no local megabatching.

Qualitative sample is phrase-fragment level: it contains local word patterns
such as "on the", "a city", "first", "in that", but not coherent sentences.

## Changes From The Original Whitepaper

These deviations were made because the paper formula either failed in code or
created a testable pathology.

| Area | Paper | Current implementation | Reason |
|---|---|---|---|
| BSR K/V/Q | Diagonal binding masks `c⊗W_K`, `c⊗W_V`, `c⊗W_Q` | Dense 1-bit Boolean projections via XNOR/popcount/sign | Diagonal K/V cancels content: `(c⊗W_K)⊗(c⊗W_V)=W_K⊗W_V`. |
| BSR timing | Write current association then read `S_i` | Read prior causal state, then write current association | Prevents current token from dominating the context branch. |
| Residual merge | Majority residuals | Learned gated MUX residuals initialized mostly closed | Raw majority residuals destroyed signal at init; MUX gives identity start. |
| Position decode | Concept bound to position, decoded against raw `E` | `position_decode=next_unbind` unbinds next position before codebook decode | Direct positioned decode is measured at 0.000 accuracy. |
| BOLD local optimizer | Literal sign flip / distributed voting discussion | Int8 per-weight accumulator + threshold + β-plasticity | Preserves sub-threshold evidence locally without top-k discard. |
| Megabatch | Distributed droplet/tracker mechanism | Removed from local CLI | Serial megabatching is not the intended single-machine equivalent. |
| Hopfield memory | Static plus optional dynamic writes | Static learned slots only | Dynamic writes are explicitly a stability risk. |
| Gamma | Learned low-precision decay | Fixed multi-timescale spread | Keeps implementation simple; gamma remains the non-1-bit concession. |

## Work Attributable To The Original Claude Session

The original handoff implemented the self-contained `model/` package:

- `vsa.py`: packed bit1 VSA operations.
- `bold.py`: BOLD parameter and optimizer machinery.
- `layers.py`: BooleanLinear, DiagBind, TokenCodebook, HopfieldBank, BSR,
  gated residual merges.
- `model.py`: HÆMMR stack, loss/backward, generation.
- `data.py`, `train.py`, `sample.py`: WikiText/GPT-2 tokenization, training,
  sampling.
- Initial tests and smoke training.

Important fixes from that phase:

- Replaced raw majority residuals with gated MUX residuals after observing depth
  collapse.
- Identified local serial `--accum` as conceptually wrong for single-machine
  training.
- Moved away from top-k local flip filtering toward the integer accumulator.

Subsequent work in this iteration:

- Removed old `--accum` and `max_flip_frac` APIs.
- Added dense causal BSR.
- Added safe position decode via `next_unbind`.
- Replaced hot forward matmuls with `brute.fast.matmul`.
- Added component tests, efficiency contracts, synthetic probes, WikiText probes,
  and this benchmark matrix.

## Where HÆMMR Excels

- Binary algebra is exact and cheap: bind/unbind, position unbind, and small VSA
  records are all at 1.000 accuracy.
- Memory footprint is excellent: 525,568 bit parameters is roughly 0.066 MB.
- Packed forward is fast for this small CPU profile: about 105k tokens/s.
- It can learn induction-style associative patterns at length 64, matching the
  tiny Transformer on that synthetic task.
- Causal BSR streaming matches batched recurrence exactly, so the O(1)-per-token
  inference path is structurally viable.

## Where It Falls Short

- It does not yet match transformer attention on selective retrieval:
  marker retrieval is chance-level and copy retrieval is chance-level.
- WikiText generation remains phrase-fragment level. It has local token
  distribution knowledge, but not robust sentence construction.
- BSR/Hopfield still require unpacking to ±1 for vote/majority state, which
  undermines the ideal packed-only forward path.
- The static Hopfield bank works when keys are controlled, but current training
  does not make it solve arbitrary copy/retrieval.
- Position binding is safe only with explicit unbinding. Direct raw decode of
  positioned concepts is demonstrably invalid.

## Research Priorities

1. **Binary key-value retrieval.** Add a stronger retrieval path that preserves
   payload identity under distractors. Marker/copy tasks should be the gate:
   target >=0.90 at lengths 16 and 64 before scaling WikiText.
2. **Trainable query/key alignment.** Current hard top-k and Hebbian Hopfield
   update are not enough for arbitrary retrieval. Test loss-driven key gradients,
   straight-through top-k alternatives, or explicit role-coded KV memory.
3. **Packed vote/majority kernels.** Add `brute` fused ops for packed
   association-to-int accumulator and packed majority/readout. This directly
   targets the 90 `unpack_pm1` calls in the small profile.
4. **Position policy.** Compare `use_position=False`, `next_unbind`, and possible
   relative/role-only position schemes on the same matrix. Direct raw positioned
   decode must remain forbidden.
5. **Matched baselines.** Extend the matrix with a small RetNet/RWKV-style FP
   baseline, not only a Transformer, to separate attention quality from
   recurrence quality.
6. **Training stability.** The tiny WikiText run peaked early and drifted worse.
   Add checkpoint-on-best-val, threshold schedules, and per-parameter flip-scale
   sweeps before increasing D.

## Bottom Line

HÆMMR's binary substrate is defensible. The current blocker is not basic VSA
geometry, packed computation, or recurrence implementation. The blocker is
selective content-addressed retrieval. Until marker/copy retrieval reaches
transformer-like accuracy, scaling the language model will mostly improve
frequency modeling and induction-like patterns, not robust sentence-level
context use.


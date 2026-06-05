# Benchmark Report — native binary transformer

> Regenerate with `.venv/bin/python model/benchmark_matrix.py` (add `--json
> out.json` for machine-readable output). Numbers below are a representative run
> (`seed=0`, CPU, Apple MPS available).

The matrix reports, side by side, the exact algebra the model is built on, the
attention mechanism's bit-exactness, end-to-end learning vs a float transformer
baseline, and the packed-bit efficiency.

## geometry — exact bit1 / VSA identities

| identity | result |
|----------|--------|
| `XNOR(a,b)` == bipolar multiply `e(a)·e(b)` | ✓ exact |
| `⟨u,v⟩` == `D − 2·Hamming(u,v)` | ✓ exact |

## mechanism — attention exactness

| property | result |
|----------|--------|
| integer score `ℓ_ij` == `d_h − 2·H(q_i,k_j)` | ✓ exact |
| forward stays packed `brute.bit1` | ✓ |
| hardmax transports the argmax value bit-for-bit | ✓ (see `tests/test_attention.py`) |

## learning — previous-token copy (non-local attention task, chance = 0.0625)

Previous-token copy is genuinely non-local: a position-free concept `x_i` cannot
contain `token[i-1]`, so the head must select the previous token (ALiBi recency +
strict causal) and the value path must transport it.

| model | best accuracy |
|-------|---------------|
| **binary transformer** (BEP, fully packed) | **1.00** |
| float transformer baseline (fp32, Adam) | 1.00 |

The binary transformer **matches the float transformer** on a non-local attention
task — the headline result. (The binary model uses the gentle-update retrieval
recipe; see `tests/_helpers.py::retrieval_config`.)

## efficiency — packed-bit footprint

For a `D=256, n_layers=2, n_heads=4, d_ff=512, vocab=256` model:

| metric | value |
|--------|-------|
| forward unpacks (hardmax) | **0** (fully packed hot path) |
| parameter footprint (int8 `H`) | ≈ 1.38 MB (1.38 M 1-bit params) |
| equivalent fp32 + Adam footprint (weight + m + v) | ≈ 16.53 MB |
| **memory ratio** | **≈ 12× smaller** |

BEP carries **no per-parameter optimiser state** (no momentum, no Adam moments),
so the train-time parameter memory is exactly the int8 hidden weights. The soft
value-combine accumulator is **int8**. These contracts are enforced in
`tests/test_memory.py` and `tests/test_efficiency_contracts.py`.

## test suite

`.venv/bin/python -m pytest model/tests -q --run-slow` — 173 tests (CPU + MPS),
including end-to-end learnability probes and the Stage-B `signed_bundle` kernel
parity (CPU + MPS; CUDA compile-checked only).

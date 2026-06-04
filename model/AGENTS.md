# AGENTS.md — working in `model/`

Native binary decoder transformer on packed `brute.bit1`, trained with BEP. Read
`README.md` for the architecture; this file is the operational guide.

## Environment & commands

Always use the project venv (Python 3.14, torch 2.12, MPS available, **no CUDA**):

```bash
.venv/bin/python -m pytest model/tests -q                 # full suite (CPU+MPS), ~1.5s
.venv/bin/python -m pytest model/tests -q --run-slow      # + end-to-end learnability probes (~30s)
.venv/bin/python -m pytest model/tests/test_attention.py -q   # the core mechanism tests
.venv/bin/python model/train.py --steps 500 --D 256 --layers 2 --n-heads 4   # run as a script,
.venv/bin/python model/benchmark_matrix.py --quick            #   not -m (dir shadows model.py)

# Rebuild the brute extension after touching cbrute/ (Stage-B kernel etc.):
.venv/bin/pip install --no-build-isolation -ve .
.venv/bin/python -m pytest tests/unit -q                  # brute regression (must stay green)
```

`model/` is put on `sys.path` by `tests/conftest.py` and by `train.py`, so modules
import as `model`, `attention`, `layers`, `bep`, `vsa`, `data` (no package prefix).

## File map

| file | contents | status |
|------|----------|--------|
| `attention.py` | `BinaryMultiHeadAttention`, `signed_bundle`, `alibi_slopes`, packed head split/merge | **new** |
| `layers.py` | `BooleanLinear`, `DiagBind`, `TokenCodebook` (kept); `BinaryGLU`, `MajorityResidual` (new) | edited |
| `model.py` | `TransformerConfig`, `Block`, `BinaryTransformerLM` | rewritten |
| `bep.py`, `vsa.py` | BEP trainer & VSA primitives | reused verbatim |
| `train.py`, `data.py` | WikiText training loop & loader | adapted |
| `tests/` | full suite + `_helpers.py` (synthetic tasks + float baseline) | new |

The non-transformer mixers (`BSR`, `HopfieldBank`, `EpisodicSlotMemory`,
`ResidualMerge`) were removed; their useful backward patterns were lifted into
`attention.py` (value path) and `MajorityResidual` (gate update).

## bit1 conventions & gotchas (read before editing the hot path)

* **Last axis is the packed axis.** A `bit1` tensor of shape `(…, K)` stores
  `ceil(K/64)` uint64 words in `_packed_buf`. Keep `K` (and head dim `d_h`) a
  multiple of 64 to stay on the packed fast paths.
* **`brute.fast.matmul(a, b_t)` is the signed dot:** `K − 2·Hamming`, i.e.
  `⟨a, b⟩` over ±1, returning a plain int32 `(M, N)`. `b` must be stored `(N, K)`.
* **Leading-axis gather is packed:** `x[idx]` / `x.index_select(0, idx)` slices
  whole packed rows (used for the hardmax value gather and BEP active-row gather).
  Last-axis indexing (`x[..., j]`) is *not* packed.
* **No `cat`/`stack` on the packed axis** — they fall through to the bool path and
  unpack. To assemble/split heads, operate on the uint64 `_packed_buf` words
  directly (`_split_head`, `_merge_heads` in `attention.py`), then rewrap with
  `Tensor._make_bit1_from_packed`.
* **`permute`/`reshape` stay packed only if the last axis stays last and 64-aligned.**
* **`.bool()` / `pm1_int(...)` unpack** (via `unpack_bool`). They are fine in the
  BEP *backward* (weight-gradient granularity) but must never appear in the
  forward sequence-mixing path. `sign_to_bit1`, `to_bit1`, `bind`, `bundle3`,
  `mux`, `combine_desired`, `brute.fast.*` are all packed.
* **`sign(0) = +1`** everywhere (`z >= 0`), matching `sign_to_bit1`.

## The no-unpack contract (enforced by tests)

`tests/test_efficiency_contracts.py` instruments the C++ `unpack_bits` (→ dense ±1
float) and `unpack_bool` (→ bool bytes) ops:

* **Forward** (hardmax) must perform **zero** unpacks of any kind.
* A **full training step** must perform **zero** `unpack_bits` — no dense ±1 float
  anywhere. `unpack_bool` is allowed only at parameter / active-row granularity
  (bounded `< B·n²`), never per token-pair.
* Soft-attention forward unpacks the value transiently **only** in the Stage-A
  reference; once the kernel is built it is zero. Keep the hardmax path pristine.

If you add to the forward, run this test — it is the canary for accidental
unpacking.

## The `signed_bundle` kernel (Stage B)

`cbrute/{cpu,metal,cuda}` + `cbrute/ext.cpp` (schema `signed_bundle(Tensor W,
Tensor V, int D) -> Tensor`) + `brute/fast.py` wrapper. `attention._kernel_available()`
checks the *registered op* (not just the Python wrapper) so the reference is used
until you rebuild. The internal accumulator is a transient int32 register
(register integers are allowed); it is bit-identical to the int8-clamped reference
because clamping preserves sign. **CUDA is compile-checked only on this machine —
not run.** After editing any `cbrute/` file: rebuild, then run both
`model/tests/test_signed_bundle_kernel.py` and `tests/unit`.

## Hyper-parameters — what each does & sane ranges

| knob | effect | sane range |
|------|--------|-----------|
| `D` / `n_heads` | model dim / heads; `D//n_heads` packed (×64) | `D∈[128,1024]`, `d_h∈{64,128,256}` |
| `d_ff` | GLU hidden width | `2·D` typical |
| `attn_mode` | `hardmax` (exact gather) / `soft` (int8 vote bundle) | hardmax for copy, soft for blend |
| `attn_band` | soft attention temperature (integer margin) | `1` (≈hardmax) … `~8` |
| `causal_strict` | exclude self (j<i) ⇒ recency = previous token | `True` for retrieval probes |
| `value_proj` | learned `W_V/W_O` (`True`) vs raw-concept copy (`False`) | `False` = cleaner retrieval |
| `residual_mode` | `mux` (replace, copy) / `majority` (add, BOLD-faithful) | `mux` for copy tasks |
| `gate_open` | residual admittance init openness | `0.05` (LM) … `0.9` (clean retrieval) |
| `alibi` / `alibi_slopes_override` | integer recency bias; slopes scale to `d_h` | last head ≈ `2·d_h+1` = recency |
| `r` / `margin_r_final` / `margin_anneal_steps` | BEP margin trigger fraction (anneal) | `r∈[0.02,0.15]` |
| `readout_warmup_steps` | codebook-only warm start | `15–40` |
| `max_trigger_rate` | cap hidden BEP update rows / step | `0.05–0.3` (gentle) |
| `update_clip` / `block_update_clip` | per-step `ΔH` clamp | `1–3` |
| `init_inertia` / `block_init_inertia` | initial `|H|` (plasticity vs stability) | head `1`, blocks `4–16` |
| `bits` | `H` clip width `±2^{bits-1}` | `15` |
| `sem_weight` | semantic rerank lane weight | `0`–`0.5` |
| `margin_weight` / `self_supervised_induction` | train `W_Q/W_K` from induction | on for content tasks |

**Gentle-update recipe for retrieval** (`tests/_helpers.py::retrieval_config`):
`value_proj=False, residual_mode='mux', causal_strict=True, gate_open=0.9,
recency slope, r=0.04, max_trigger_rate=0.08, readout_warmup_steps=30,
update_clip=3, block_init_inertia=8, block_update_clip=1`.

## Adding tests

Mirror the existing files (`pytest`, the `gen` fixture for a seeded CPU
generator, `@pytest.mark.slow` for end-to-end probes, `@pytest.mark.mps` +
`torch.backends.mps.is_available()` for device parity). Prefer **exact** mechanism
assertions (unpack to ±1 and compare to a dense reference) over thresholds; use
thresholds only for the slow learnability probes.

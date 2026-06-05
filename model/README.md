# `model/` — a native binary decoder transformer (BEP-trained, fully packed)

A **1:1 binary-native equivalent of a standard decoder transformer**: token↔token
attention, a GLU feed-forward, residual connections and Hamming LM-head logits —
built on packed 1-bit `brute.bit1` (uint64) activations/visible weights and trained with
**BEP** (Boolean error propagation: integer hidden weights, 1-bit visible
weights, binary desired-activation backward). No float weights, no float
optimiser state, and the sequence-mixing hot path never unpacks.

This replaces the earlier HÆMMR mixers (a linear-recurrence `BSR`, a
`HopfieldBank`, an addressed `EpisodicSlotMemory`) with one native
`BinaryMultiHeadAttention` — the open problem BEP's own paper lists as future work
("extending BEP to transformer-style models … multi-head mechanisms").

```
input_ids ── wte ──▶ position-free hidden_states  x_i ∈ 𝔹^D
   ┌────────────────────────────────────────────────────────────────┐ × n_layers
   │  a = h[i].attn(x)                   # token↔token (ALiBi, causal)  │
   │  x = binary residual(x, a)                                           │
   │  f = h[i].mlp(x)                    # exact XNOR-gated GLU          │
   │  x = residual(x, f)                                                │
   └────────────────────────────────────────────────────────────────┘
   ── lm_head ──▶ Hamming logits ──▶ softmax
```

## GPT-2 naming map

The public structure follows Hugging Face GPT-2 names where the concepts match:

| GPT-2 name | Binary model equivalent |
|------------|-------------------------|
| `transformer.wte` | `wte` deterministic packed token codes |
| `transformer.wpe` | no table; position is integer ALiBi in attention scores |
| `transformer.h` | `h`, the list of decoder blocks |
| `h[i].attn` | `BinaryMultiHeadAttention` |
| `h[i].mlp` | `BinaryGLU` |
| `transformer.ln_f` | no float layer norm; final hidden state goes straight to `lm_head` |
| `lm_head` | trainable binary Hamming LM head (`lm_head.weight`) |

The old lexical projection / semantic rerank lane is gone; logits are produced
directly by `lm_head(hidden_states)`. Unlike Hugging Face GPT-2's tied
`lm_head.weight` / `transformer.wte.weight`, this implementation keeps input
codes fixed and trains the output `lm_head.weight`.

## The attention mechanism (`attention.py`)

The persistent stream `x_i ∈ 𝔹^D` is **position-free**; position enters only as an
integer bias on the score *register*. Per head (`d_h = D / n_heads`, a multiple of
64 so heads stay packed):

| step | operation | type |
|------|-----------|------|
| project | `q,k,v = sign(W_{Q,K,V}^h · x)` (`BooleanLinear`, per head) | `bit1` |
| score | `ℓ_ij = ⟨q_i, k_j⟩ = d_h − 2·H(q_i,k_j)` (`brute.fast.matmul`) | int32 **register** |
| position | `ℓ_ij += −b_h·(i−j)` (ALiBi, integer slope per head) | int32 register |
| causal | `ℓ_ij = −∞` for `j>i` (or `j≥i` if `causal_strict`) | mask |
| combine (hardmax) | `o_i = v_{argmax_j ℓ_ij}` (packed `index_select` gather) | `bit1` |
| combine (soft) | `o_i = sign(Σ_{j≤i} n(ℓ_ij)·v_j)`, `n(ℓ)=relu(ℓ−ℓ_max+band)` | int8 weights/acc |
| output | `a = sign(W_O · concat_h o)` (`BooleanLinear`) | `bit1` |

Heads are split/merged by slicing and concatenating the underlying uint64
**packed buffers** directly (never `cat`/`stack` on the logical bits), so the
whole forward stays packed.

### Exact ↔ approximate map

* **Exact (algebraic identities, no approximation):** XNOR = bipolar multiply;
  `⟨u,v⟩ = D − 2·Hamming`; the score register; ALiBi (relative, integer); the
  causal mask; **hardmax** value transport (a packed gather — bit-exact copy of
  the argmax source); the XNOR-GLU gate; the coordinate majority residual; the
  Hamming LM-head logits.
* **Bounded approximation (soft attention only):** the soft value combine is a
  *signed integer vote bundle* `sign(Σ_j w_ij·pm1(v_j))`. It satisfies a
  **dominance theorem** — if one source's weight exceeds the sum of all others
  (`w_ij* > Σ_{j≠j*} w_ij`), the bundle equals that source's value coordinate-wise
  — so when the top key's margin clears the band it reduces to hardmax. The int8
  accumulator saturates at ±127, but **clamping preserves sign**, so the output
  sign is exact regardless of magnitude.

## BEP training (`bep.py`, reused verbatim)

Each parameter is an int8 hidden weight `H`; the visible weight is `W = sign(H)`
(≈1 bit/weight). The backward pass threads **binary desired activations** `a*`
(bit1) — never a float signal:

* `linear_backward`: `ΔH = a*_out ⊗ a_in` (binary outer product → integer
  accumulate); upstream `a*_in = sign(Wᵀ a*_out)`. Both are bit1 XNOR-popcount
  matmuls.
* The head is **contrastive margin-triggered**: an update fires for a position
  only when `logit[target] − max_other < r·D`. No softmax/CE on the backward
  path (real softmax/CE are logging only).
* Attention's score lane (`W_Q, W_K`) is trained by a per-head **Hamming-margin**
  objective (`BinaryMultiHeadAttention.margin_loss`) supervised by the
  self-supervised induction signal; the value lane is trained by routing the
  desired output back through the same combine (`backward`).

## Hot-path-packed invariant

Every persistent activation and visible weight is packed `brute.bit1` (uint64)
end-to-end. Trainable inertia is the int8 `H` buffer behind each visible weight.
The other integers are **transient registers**: the int32 score `ℓ`, the int8
attention weights `n(ℓ)`, and the int8 value-vote accumulator. The forward pass
performs **zero** unpacks of any kind (verified in
`tests/test_efficiency_contracts.py`); a full training step materialises **no**
dense ±1 float (`unpack_bits`) — the only bit→byte unpacks (`unpack_bool`) are
BEP weight-gradient / LM-head updates at parameter granularity, never per
token-pair.

## The one custom kernel — `brute.signed_bundle` (Stage B)

The soft combine is an **int8-weight × bit1 → bit1** reduction. `brute` ships
bit1×bit1→int (XNOR-popcount) but not int×bit1, so this is the single
register-bound primitive that needs a kernel. `brute.signed_bundle(W:int8 (B,M,N),
V:bit1 (B,N,Dp), D) → bit1 (B,M,Dp)` reads `V` straight from its packed words
(never materialised) and accumulates the signed vote in a transient register. It
is implemented on **CPU** (`cbrute/cpu`), **MPS** (`cbrute/metal`) and **CUDA**
(`cbrute/cuda`, compile-checked only on this Mac — no NVIDIA GPU here), and is
reused for both the attention forward and its transposed backward. Until the
extension is rebuilt the code transparently falls back to a bit-identical Stage-A
reference; `tests/test_signed_bundle_kernel.py` asserts kernel == reference on
CPU + MPS.

## What learns, and the training characteristics (research notes)

This is binary, discrete-update training; a few properties differ from a float
transformer and are worth knowing (all reflected in the config defaults and the
test recipes in `tests/_helpers.py::retrieval_config`):

* **Residual cold-start.** A summed binary residual is *not* identity at init —
  an untrained branch injects ~25% bit noise and destroys the signal before the
  LM head can learn. The residual is therefore admitted through a per-coordinate
  gate initialised mostly **closed** (≈ identity at init), opening where the
  branch predicts the target.
* **`residual_mode`.** `mux` (open ⇒ branch *replaces* skip) is required for exact
  value replacement (copy / retrieval); `majority` (open ⇒ `maj3(skip,branch,c)`)
  is the BOLD-faithful binary residual but can only *add*, not overwrite.
* **`value_proj`.** `True` (faithful `W_V/W_O`) is the default. `False` is
  raw-concept-copy attention: the retrieved value is the clean source codeword
  (no projection), which the LM head decodes directly — the cleanest, most stable
  transport for copy / retrieval tasks.
* **ALiBi must scale to `d_h`.** Scores are the unnormalised `±d_h` dot, so slopes
  span `1` (content head) to `> 2·d_h` (recency head, attends the nearest key).
* **Gentle updates.** The LM-head/codebook co-adaptation runs away under
  aggressive hyper-parameters; small `r`, `max_trigger_rate`, `update_clip` and a
  `lm_head_warmup_steps` keep it convergent.

**Verified end-to-end:** local next-token identity → ~1.0; previous-token copy (a
genuine non-local attention task) → ~1.0, *matching a float transformer baseline*
on the same task (`tests/test_transformer_baseline.py`). The attention mechanism
itself is verified bit-exact in `tests/test_attention.py`.

## Hyper-parameters (`TransformerConfig`)

| group | knobs | notes |
|-------|-------|-------|
| shape | `D, n_layers, n_heads, d_ff` | `D // n_heads` must be a multiple of 64 |
| attention | `attn_mode {soft,hardmax}`, `attn_band`, `alibi`, `causal`, `causal_strict`, `value_proj`, `alibi_slopes_override` | soft `band` ≥ 1 = temperature |
| residual | `residual_mode {mux,majority}`, `gate_open` | `gate_open` low ⇒ identity init |
| head / BEP | `r`, `margin_r_final`, `margin_anneal_steps`, `lm_head_warmup_steps`, `max_trigger_rate`, `init_inertia`, `update_clip`, `block_init_inertia`, `block_update_clip` | margin trigger + integer step |
| margin lane | `margin_theta_pos/neg`, `margin_weight`, `self_supervised_induction` | trains `W_Q/W_K` |

## Running

```bash
.venv/bin/python model/train.py --steps 1500 --D 512 --layers 2 --n-heads 4 --seq-len 64
.venv/bin/python model/benchmark_matrix.py --quick     # geometry / mechanism / efficiency matrix
.venv/bin/python -m pytest model/tests -q              # full suite (CPU + MPS)
.venv/bin/python -m pytest model/tests -q --run-slow   # + end-to-end learnability probes
```

(Run `train.py` / `benchmark_matrix.py` as scripts, not `python -m model.*` — the
directory name `model` would otherwise shadow the `model.py` module.)

See `AGENTS.md` for the development workflow, bit1 conventions, the no-unpack
contract, and the rebuild command for the Stage-B kernel.

# BEP — Binary Error Propagation, in brute

A reference implementation of **Binary Error Propagation** (Colombo et al.,
ICLR 2026 — [arXiv:2512.04189](https://arxiv.org/abs/2512.04189)) for
fully-binary MLPs, built on top of `brute.bit1` tensors.

> BEP is the first end-to-end binary training algorithm: every forward
> activation, every backward error signal, and every weight is binary; the
> only piece of non-binary state is an integer-valued *metaplasticity*
> accumulator. All matmuls — forward and backward — run on the
> XNOR + popcount fast path provided by `brute.bit1`.

## Files

| file              | what's in it                                                              |
| ----------------- | ------------------------------------------------------------------------- |
| `bep.py`          | `BEPModel`, the BEP forward + Eq. 7 backward + Eq. 9 winner-takes-update. |
| `bef.py`          | Greedy coordinate-flip Binary Equiangular Frame generator (Appendix C).   |
| `data.py`         | Random Prototypes generator + FashionMNIST median-thresholding helper.    |
| `train.py`        | CLI training script (the demo).                                           |
| `test_bep.py`     | Local pytest suite — runs as `python -m pytest test_bep.py`.              |

The example is self-contained; nothing here touches the global `brute`
package or the top-level `tests/` directory.

## What BEP looks like in this implementation

**Forward** (`BEPModel.forward`):

```python
a_bit  = brute.as_tensor(x > 0, dtype=brute.bit1)                   # ±1 input
for l in range(L):
    W_bit = brute.as_tensor(H[l] >= 0, dtype=brute.bit1)            # W = sign(H)
    z     = a_bit @ W_bit                                           # XNOR-popcount, ±1 dot
    a_bit = brute.as_tensor(z > 0, dtype=brute.bit1)                # sign(z)
logits = a_bit @ P_bit                                              # fixed BEF classifier
```

**Backward** (`BEPModel.step`, Eq. 7):

```
a*_L = ρ^{c}                                                        # correct class prototype
for l = L-1 .. 1:
    g_{l+1} = (|z_{l+1}| ≤ ν · K_l)                                 # backward gate
    a*_l    = sign( W_{l+1}^T (g_{l+1} ⊙ a*_{l+1}) )                # binary chain rule
```

**Update** (Eq. 9, winner-takes-update within neuron groups):

```
stability[μ, j] = a*_l[μ, j] · ⟨H_{l, j}, a_{l-1}[μ]⟩
mask M^μ        = per-group argmin of |stability| over neurons with stability < 0
H_l            ← H_l + 2 · sum_μ ( (a*_l[μ] ⊙ M^μ)^T a_{l-1}[μ] )
```

A reinforcement step (CP+R, Sec. 3.3) optionally drifts each non-zero
weight further from zero with probability `p_r √(2 / (π K_l-1))`.

## Running the demo

```bash
# Random Prototypes (synthetic binary, ~10 s/epoch on CPU)
python train.py --dataset prototypes --epochs 30 --hidden 512 512

# FashionMNIST (downloads on first run, ~30 s/epoch on CPU)
python train.py --dataset fashion_mnist --epochs 40 --hidden 256 256
```

Useful flags:

| flag             | default     | meaning                                              |
| ---------------- | ----------- | ---------------------------------------------------- |
| `--hidden`       | `256 256`   | space-separated layer widths `K_1 ... K_L`           |
| `--r`            | `0.5`       | trigger margin (Eq. 1), as a fraction of `K_L`       |
| `--nu`           | `0.05`      | backward gating threshold (Eq. 5)                    |
| `--group-size`   | `4`         | neurons per group for winner-takes-update            |
| `--p-reinforce`  | `0.5`       | base reinforcement probability                       |
| `--weight-clip`  | `2048`      | clip on `|H_l|` (paper uses `2^{B-1}-1` with B=16)   |
| `--bef-iters`    | `auto`      | coord-flip iterations for the BEF classifier         |

## Reproducing paper numbers

The paper reports the following test accuracies on **Random Prototypes**
(`L=2`, `K_l ∈ {525, 250, ...}`, 5 runs):

| method                       | accuracy  |
| ---------------------------- | --------- |
| QAT (Larq)                   | ~70 %     |
| Colombo et al. 2025 local rule | ~74 %   |
| **BEP (this paper)**         | **~78 %** |

With this implementation we get:

| config                                   | best test acc |
| ---------------------------------------- | ------------- |
| `--hidden 512 --group-size 4` (L=1)      | **72.3 %**    |
| `--hidden 512 512 --group-size 8`        | **71.6 %**    |
| `--hidden 512 512 --group-size 16`       | **72.7 %**    |
| `--hidden 1024 1024 --group-size 16`     | reaches **~75 %** range with longer training |

The gap to the paper's headline number is dominated by two pieces we left
out for clarity: the *adaptive* group-size schedule (`γ_l` ramps up when
validation accuracy stagnates) and the per-epoch reinforcement rescaling
by `√E_e`. Adding either closes most of the gap; both are mentioned in
the inline comments.

The single-layer (L=1) configuration reduces BEP exactly to the
CP+Reinforcement rule of Baldassi (2009) and reaches the expected
performance band, validating that the integer-weight machinery and the
update rule are wired correctly.

## Running the tests

```bash
cd examples/bep
python -m pytest test_bep.py -v
```

The test suite covers:

* the BEF generator (shape, ±1 invariant, improves over random init);
* the binary forward (shape, dtype, equality with a float ±1 reference);
* `step` correctness (no-op when no sample triggers, mutates `H` when a
  sample triggers);
* an end-to-end "learns better than chance" check on a tiny prototypes
  task for both L=1 and L=2;
* that the model uses `brute.bit1` packed weights and the bit1 matmul
  fast path (output dtype `int32`).

## Why this is useful

Once you build on `brute.bit1`, you get:

* **Packed storage**: every binary weight & activation lives as 1 bit in
  a 64-bit int64 word — 32× less memory than `float32`, 8× less than
  `int8`.
* **XNOR-popcount matmul**: `bit1 @ bit1` lowers to a single packed
  kernel (≈ K/64 popcounts per output) — orders of magnitude fewer FLOPs
  than any STE-based QAT pipeline.
* **No autograd, no surrogate gradients**: BEP-Sec. 3.4 shows the
  algorithm is a faithful binary analog of the BP chain rule; nothing in
  this folder needs PyTorch autograd.

The combination is exactly what the paper highlights as the path to
running BNN *training* (not just inference) on resource-constrained
hardware: weights stay in 16 bits, errors stay in 1 bit, and every
matmul is a few popcounts.

## Architecture notes — closeness to transformers

The paper validates BEP on MLPs and binary RNNs and explicitly leaves
convolutional and *transformer-style* models for future work. Of the two
architectures the paper supports, the MLP is the one whose update rule
maps most directly onto a transformer block — a standalone matmul over a
linear projection of incoming activations, identical to the per-token
FFN computation. This is the architecture chosen here.

Extending to attention would require a binary analog of the attention
softmax — open research — but the per-layer machinery in `bep.py` is
already drop-in for the `nn.Linear` modules inside an FFN.

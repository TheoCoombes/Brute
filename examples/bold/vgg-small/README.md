# BOLD: VGG-SMALL on CIFAR-10

Faithful re-implementation of the **VGG-SMALL / CIFAR-10** experiment from
**BOLD: Boolean Logic Deep Learning** (Nguyen et al., NeurIPS 2024), built on
top of [`brute`](../../../README.md).

| Method                 | Forward (W/A) | Test Acc.            | Source         |
| ---------------------- | ------------- | -------------------- | -------------- |
| Full-precision         | 32 / 32       | **93.80 %**          | Table 2        |
| BOLD **w/ BN**         | 1 / 1         | **92.37 ± 0.01 %**   | Table 2 (paper)|
| BOLD **w/o BN**        | 1 / 1         | **90.29 ± 0.09 %**   | Table 2 (paper)|

## What this example demonstrates

* `brute.nn.BitConv2d`, `brute.nn.BitLinear`, `brute.nn.BitActivation` —
  drop-in Boolean replacements for `nn.Conv2d`, `nn.Linear`, `nn.ReLU`,
  whose parameters live natively in `{0, 1}`.
* `brute.optim.BooleanOptimizer` — the deterministic per-weight flip-rule
  optimizer from BOLD's Algorithm 8.
* `brute.optim.split_parameters` — automatic split between Boolean and
  full-precision parameters so the BOLD-style mixed training recipe
  (`Adam(FP) + BooleanOptimizer(Bool)`) is two lines of glue.

The training script reproduces both the **with-BN (92.37 %)** and
**without-BN (90.29 %)** variants from a single CLI flag.

## Architecture

VGG-SMALL ([Simonyan & Zisserman 2015](https://arxiv.org/abs/1409.1556),
BNN variant from BinaryConnect, modified by BOLD to end in a single FC layer
— see Table 9 footnote 2):

```
RGB 3×32×32
  ↓
[FP Conv 3→128 3×3, pad 1]    BN?   BitActivation
[BitConv  128→128 3×3]   MaxPool 2  BN?   BitActivation
[BitConv  128→256 3×3]               BN?   BitActivation
[BitConv  256→256 3×3]   MaxPool 2  BN?   BitActivation
[BitConv  256→512 3×3]               BN?   BitActivation
[BitConv  512→512 3×3]   MaxPool 2  BN?   BitActivation
Flatten → 512·4·4 = 8192
[FP Linear 8192 → 10]
```

* The first conv is full-precision (it sees raw real-valued RGB pixels) and
  the last linear is full-precision (it produces real-valued class logits).
  Together these are ~85k of the ~4.57M parameters (< 2 %). Both are
  optimised by Adam.
* Every other conv / linear is Boolean: weights ∈ `{0, 1}`, forward is
  XOR-count, backward follows BOLD Algorithm 7, optimised by
  `BooleanOptimizer`.
* `BitConv2d(maxpool_after=True)` sets the BOLD backprop-variance scaling
  factor (Appendix C.2, Remark C.2) to account for the 1/4 variance drop
  induced by the 2×2 max-pool downstream.
* `BitActivation(m=in_ch·9)` picks the `α = π / (2·√(3m))` from Eq. 47,
  matching the receptive-field size feeding the activation.

## Training recipe (paper Appendix D.1)

| Hyperparameter            | Value                                            |
| ------------------------- | ------------------------------------------------ |
| Epochs                    | 300                                              |
| Batch size                | 300 (paper) / 128 (default here)                 |
| Real-param optimiser      | Adam, lr `1e-3`                                  |
| Boolean-param optimiser   | `BooleanOptimizer`, lr `150` (BN) / `12` (no BN) |
| LR schedule (both)        | Cosine to 0 over the full run                    |
| Data augmentation         | Random crop (pad 4) + horizontal flip            |
| Strong aug (`--strong-aug`) | + RandAugment + ColorJitter + Mixup            |
| Loss                      | Cross-entropy                                    |

## Usage

```bash
# Full reproduction targeting the 92.37 % paper number (~hours on a GPU).
python train.py --use-bn --epochs 300 --batch-size 300 --strong-aug

# Or the 90.29 % no-BN variant.
python train.py --epochs 300 --batch-size 300 --strong-aug

# Smoke test that everything wires up (~3 min on CPU).
python train.py --use-bn --epochs 2 --batch-size 128 --max-train-batches 50
```

Available flags (see `python train.py --help` for the full list):

| Flag                    | Meaning                                             |
| ----------------------- | --------------------------------------------------- |
| `--use-bn`              | Include BatchNorm (targets 92.37 %, vs 90.29 % off) |
| `--strong-aug`          | RandAugment + ColorJitter + Mixup                   |
| `--lr-bool 150`         | Override the auto-picked Boolean-optimizer lr       |
| `--lr-real 1e-3`        | Override the Adam lr for FP params                  |
| `--max-train-batches N` | Cap iterations per epoch (smoke-test mode)          |
| `--checkpoint PATH`     | Save the best-test-accuracy checkpoint              |

## Files

| File                | Purpose                                           |
| ------------------- | ------------------------------------------------- |
| `vgg_small.py`      | The model — VGG-SMALL with BOLD Boolean layers    |
| `train.py`          | Training / evaluation loop with mixed optimisers  |
| `test_recreation.py`| Smoke tests: parity with paper Algorithms 5–8     |
| `requirements.txt`  | Pin list                                          |

## Verification of paper fidelity

`test_recreation.py` covers:

* `BitLinear.forward`   ≡   Algorithm 5 (`logical_xor`-based reference).
* `BitLinear.backward`  ≡   Algorithm 6 (Boolean) and Algorithm 7 (real).
* `BitConv2d.forward`   ≡   naive nested-loop XOR-count.
* `BitConv2d.backward`  ≡   `conv_transpose2d` / `conv2d_weight` reference.
* `BooleanOptimizer.step`  ≡  Algorithm 8 line-by-line.

Run with:

```bash
python test_recreation.py
```

## Citation

```
@inproceedings{nguyen2024bold,
  title={BOLD: Boolean Logic Deep Learning},
  author={Nguyen, Van Minh and Ocampo, Cristian and Askri, Aymen and
          Leconte, Louis and Tran, Ba-Hien},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

"""VGG-SMALL architecture for BOLD's CIFAR-10 image classification benchmark.

Reference
---------
- Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale
  Image Recognition" (ICLR 2015). The "VGG-SMALL" variant follows the
  established BNN-literature variant (BinaryConnect, Courbariaux et al. 2015,
  §4.1) adapted to 32×32 inputs.
- Nguyen et al., "BOLD: Boolean Logic Deep Learning" (NeurIPS 2024),
  §4.1 / Appendix D.1.2 / Table 2 + Table 9. The paper reports
  ``90.29 ± 0.09%`` (without BN, 6 reps) and ``92.37 ± 0.01%`` (with BN, 5
  reps) for the *modified* VGG-SMALL ending with a single FC layer
  (Table 9 footnote 2). The full-precision baseline is ``93.80%``.

Topology
--------
Convolutional spine — 6 blocks, channels 128/128 → 256/256 → 512/512,
spatial down-sampled to 4×4 via a 2×2 max-pool at the end of each pair.
Classifier head — single FC (8192 → 10).

Per the BOLD recipe (§4 "Experimental Setup"):

  * The first conv layer (3 RGB channels in) and the final FC layer stay
    in full precision and are optimised by Adam.
  * Every other conv / linear is a Boolean :class:`brute.nn.BitConv2d` /
    :class:`brute.nn.BitLinear`, optimised by :class:`brute.optim.BooleanOptimizer`.
  * BatchNorm is optional; without BN the activation thresholds carry the
    bias themselves.
  * Activation is the BOLD threshold step with the ``tanh'(α·Δ)`` STE
    (Appendix C.3), α chosen per Eq. 47.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from bit_conv import BitConv2d
from bit_activation import BitActivation


__all__ = ["VGGSmall"]


# Channel widths per VGG-SMALL convolutional block. Each entry is repeated
# twice (two conv layers per block), then 2×2 max-pooled.
_CHANNELS = (128, 256, 512)


class VGGSmall(nn.Module):
    """Modified VGG-SMALL with a single FC head (BOLD Table 9 footnote 2)."""

    def __init__(
        self,
        num_classes: int = 10,
        use_bn: bool = True,
        threshold: float = 0.0,
    ):
        super().__init__()
        self.use_bn = use_bn

        layers = []
        in_ch = 3
        first_layer = True

        # 6 conv layers organised in 3 (c, c) blocks; max-pool after each block.
        for block_idx, ch in enumerate(_CHANNELS):
            for conv_idx in range(2):
                is_block_end = (conv_idx == 1)
                if first_layer:
                    # FP first conv (the only layer that sees real RGB input).
                    layers.append(nn.Conv2d(in_ch, ch, 3, padding=1, bias=not use_bn))
                    first_layer = False
                else:
                    # Boolean conv. The BOLD bp_scale assumes ``maxpool_after``
                    # iff this conv is the last of its block.
                    layers.append(
                        BitConv2d(
                            in_ch, ch, 3, padding=1, bias=not use_bn,
                            maxpool_after=is_block_end,
                        )
                    )
                if use_bn:
                    layers.append(nn.BatchNorm2d(ch))
                if is_block_end:
                    layers.append(nn.MaxPool2d(2, 2))
                # Boolean activation. m = receptive-field size feeding the
                # activation (Eq. 47): for a 3×3 conv with in_ch input
                # channels that's ``in_ch · 9``. Threshold is 0 because the
                # XOR-count is 0-centred by BitConv2d.
                m_receptive = in_ch * 9
                layers.append(BitActivation(threshold=threshold, m=m_receptive))
                in_ch = ch

        self.features = nn.Sequential(*layers)
        # 32×32 → halve 3× → 4×4 spatial. Flatten to (B, 512·4·4 = 8192).
        self.classifier = nn.Linear(_CHANNELS[-1] * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def count_parameters(model: nn.Module):
    """Return (n_boolean, n_real) parameter counts. Useful for sanity-checking
    that boolean / real splits match what the BOLD paper reports."""
    n_bool = sum(p.numel() for p in model.parameters() if getattr(p, "_bold_boolean", False))
    n_real = sum(p.numel() for p in model.parameters() if not getattr(p, "_bold_boolean", False))
    return n_bool, n_real


if __name__ == "__main__":
    for use_bn in (True, False):
        m = VGGSmall(use_bn=use_bn)
        n_b, n_r = count_parameters(m)
        # Sanity: forward a synthetic batch
        x = torch.randn(2, 3, 32, 32)
        y = m(x)
        print(f"use_bn={use_bn}: y.shape={tuple(y.shape)}  boolean={n_b:,}  real={n_r:,}")

"""Datasets used by the BEP example.

* ``random_prototypes`` — the synthetic binary benchmark from Colombo et al.
  (Sec. 4.1 of the BEP paper). Each class has a random ±1 prototype in
  R^{K_0}; samples are produced by flipping each bit independently with
  probability ``p`` (default 0.46). This is the dataset on which BEP shows
  its largest absolute improvement over the SotA local-rule baseline
  (+6.89%).

* ``fashion_mnist`` — the standard 28x28 grey-scale image dataset, binarised
  via per-feature median thresholding (Sec. 3.1, "binarisation function").

Both helpers return a ``(x_train_pm1, y_train, x_test_pm1, y_test)`` tuple
with ``x`` already encoded as ±1 ``int8`` tensors ready to drop into
``BEPModel.forward``.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch


def _split(x: torch.Tensor, y: torch.Tensor, test_frac: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    n = x.shape[0]
    perm = torch.randperm(n, generator=g)
    n_test = int(round(test_frac * n))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


# ── Random Prototypes ────────────────────────────────────────────────────────

def random_prototypes(
    *,
    n_samples: int = 12000,
    n_features: int = 1000,
    n_classes: int = 10,
    flip_prob: float = 0.46,
    test_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate the Random Prototypes dataset (Colombo et al., Sec. 4.1).

    Returns
    -------
    x_train, y_train, x_test, y_test : torch.Tensor
        ``x`` is int8 ±1 of shape ``(N, n_features)`` ; ``y`` is int64 ``(N,)``.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    prototypes = (torch.randint(0, 2, (n_classes, n_features), generator=g) * 2 - 1).to(torch.int8)

    per_class = n_samples // n_classes
    extra = n_samples - per_class * n_classes
    x_chunks: list[torch.Tensor] = []
    y_chunks: list[torch.Tensor] = []
    for c in range(n_classes):
        m = per_class + (1 if c < extra else 0)
        flips = (torch.rand(m, n_features, generator=g) < flip_prob).to(torch.int8)
        # ±1 sample: prototype with bits flipped wherever ``flips`` is 1.
        sample = prototypes[c].unsqueeze(0) * (1 - 2 * flips)
        x_chunks.append(sample)
        y_chunks.append(torch.full((m,), c, dtype=torch.long))
    x = torch.cat(x_chunks, dim=0)
    y = torch.cat(y_chunks, dim=0)

    # Shuffle deterministically before the train/test cut.
    perm = torch.randperm(n_samples, generator=g)
    return _split(x[perm], y[perm], test_frac, seed=seed + 1)


# ── FashionMNIST ─────────────────────────────────────────────────────────────

def fashion_mnist(
    *,
    data_root: str = "./.data",
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load FashionMNIST and binarise each feature with the train-set median.

    Returns flat ±1 ``int8`` tensors of shape ``(N, 28*28=784)``.
    """
    try:
        from torchvision import datasets, transforms
    except ImportError as e:
        raise ImportError(
            "FashionMNIST requires torchvision. Install with `pip install torchvision`."
        ) from e
    os.makedirs(data_root, exist_ok=True)
    tf = transforms.Compose([transforms.ToTensor()])
    tr = datasets.FashionMNIST(data_root, train=True, download=True, transform=tf)
    te = datasets.FashionMNIST(data_root, train=False, download=True, transform=tf)

    x_tr = torch.stack([tr[i][0].view(-1) for i in range(len(tr))]).float()
    y_tr = torch.tensor([tr[i][1] for i in range(len(tr))], dtype=torch.long)
    x_te = torch.stack([te[i][0].view(-1) for i in range(len(te))]).float()
    y_te = torch.tensor([te[i][1] for i in range(len(te))], dtype=torch.long)

    # Median thresholding per feature on the training set (Sec. 3.1).
    median = x_tr.median(dim=0).values
    x_tr_pm1 = ((x_tr > median).to(torch.int8) * 2 - 1)
    x_te_pm1 = ((x_te > median).to(torch.int8) * 2 - 1)

    # Shuffle the training set deterministically.
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(x_tr_pm1.shape[0], generator=g)
    return x_tr_pm1[perm], y_tr[perm], x_te_pm1, y_te

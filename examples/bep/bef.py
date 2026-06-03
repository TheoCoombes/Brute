"""Binary Equiangular Frame (BEF) generator.

BEP uses a *fixed* output classifier whose prototype rows are maximally and
uniformly separated in the binary hypercube {-1,+1}^D. The paper builds one
via greedy coordinate-descent on the cost function (Appendix C, Eq. 13):

    J({rho_c}) = sum_{i<j} <rho_i, rho_j> + alpha * Var_{i<j}(<rho_i, rho_j>)

The first term pushes pairwise inner products to be as negative (i.e. as
separated in Hamming) as possible; the second pushes them toward a common
value, yielding the *equiangular* property.

The procedure is exactly the local search described in the paper:
  * start from random ±1 vectors,
  * repeatedly pick a (class, coordinate) pair at random,
  * flip the bit iff that decreases J,
  * stop after a fixed budget of iterations.

The resulting prototypes are returned as a float ±1 tensor of shape
``(C, D)``; pass through ``brute.as_tensor(..., dtype=brute.bit1)`` if you
want the packed representation.
"""

from __future__ import annotations

import torch


def _pairwise_inner_products(P: torch.Tensor) -> torch.Tensor:
    """Return the (C, C) Gram matrix of ±1 row vectors."""
    return P @ P.t()


def _cost(P: torch.Tensor, alpha: float) -> torch.Tensor:
    """Cost J = sum_{i<j} <rho_i, rho_j> + alpha * Var_{i<j}(<rho_i, rho_j>)."""
    G = _pairwise_inner_products(P)
    C = P.shape[0]
    iu = torch.triu_indices(C, C, offset=1, device=P.device)
    off = G[iu[0], iu[1]]
    return off.sum() + alpha * off.var(unbiased=False)


def generate_bef(
    n_classes: int,
    n_features: int,
    *,
    alpha: float = 1.0,
    iters: int | None = None,
    seed: int | None = 0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Greedy coordinate-flip search for a Binary Equiangular Frame.

    Returns
    -------
    P : torch.Tensor
        Float ±1 tensor of shape ``(n_classes, n_features)``.
    """
    if iters is None:
        iters = max(20 * n_classes * n_features, 5000)

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(int(seed))

    P = (torch.randint(0, 2, (n_classes, n_features), generator=gen).float() * 2 - 1).to(device)
    G = _pairwise_inner_products(P)  # incremental — flipping P[c,k] toggles 2*P[c,k]*P[c',k] in G[c,c']

    C = n_classes
    iu = torch.triu_indices(C, C, offset=1, device=device)
    off_idx_i, off_idx_j = iu[0], iu[1]

    def cost_from_offdiag(off: torch.Tensor) -> torch.Tensor:
        return off.sum() + alpha * off.var(unbiased=False)

    off = G[off_idx_i, off_idx_j]
    cur_cost = cost_from_offdiag(off)

    rand_c = torch.randint(0, C, (iters,), generator=gen)
    rand_k = torch.randint(0, n_features, (iters,), generator=gen)

    for it in range(iters):
        c = int(rand_c[it])
        k = int(rand_k[it])

        old_bit = P[c, k].item()
        # When we flip P[c, k]: G[c, c'] for c' != c changes by -2 * old_bit * P[c', k].
        delta_row = -2.0 * old_bit * P[:, k]
        delta_row[c] = 0.0
        new_col_c = G[:, c] + delta_row
        new_col_c[c] = G[c, c]
        # Build the new off-diagonal vector by updating entries involving c.
        mask_i_eq_c = (off_idx_i == c)
        mask_j_eq_c = (off_idx_j == c)
        new_off = off.clone()
        new_off[mask_i_eq_c] = new_col_c[off_idx_j[mask_i_eq_c]]
        new_off[mask_j_eq_c] = new_col_c[off_idx_i[mask_j_eq_c]]

        new_cost = cost_from_offdiag(new_off)
        if new_cost < cur_cost:
            P[c, k] = -old_bit
            # Update Gram matrix
            G[c, :] += delta_row
            G[:, c] += delta_row
            off = new_off
            cur_cost = new_cost

    return P


def bef_stats(P: torch.Tensor) -> dict:
    """Helper for tests: return mean / std / min / max pairwise inner product."""
    C = P.shape[0]
    iu = torch.triu_indices(C, C, offset=1, device=P.device)
    off = (P @ P.t())[iu[0], iu[1]]
    return {
        "mean": off.mean().item(),
        "std": off.std(unbiased=False).item(),
        "min": off.min().item(),
        "max": off.max().item(),
        "n_pairs": int(off.numel()),
    }

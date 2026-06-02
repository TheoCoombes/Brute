"""VSA primitive tests — binding/bundling/permutation on packed bit1."""
import torch
import pytest

import brute
from vsa import (binary_equiangular_frame, bind, bind_pm1, bundle3,
                 hamming_similarity, hierarchical_position_codes, position_codes,
                 random_hypervectors, sign_to_bit1, to_bit1, to_pm1)


def test_bit1_pm1_roundtrip():
    pm1 = (torch.randint(0, 2, (4, 130)) * 2 - 1).float()
    b = to_bit1(pm1)
    assert b.dtype == brute.bit1
    assert torch.equal(to_pm1(b), pm1)


def test_xnor_is_multiplication():
    """e(xnor(a,b)) = e(a)·e(b) (HÆMMR §2 / BOLD Def. 3.1)."""
    a = random_hypervectors(3, 200, generator=torch.Generator().manual_seed(1))
    b = random_hypervectors(3, 200, generator=torch.Generator().manual_seed(2))
    out = bind(a, b)
    assert out.dtype == brute.bit1
    assert torch.equal(to_pm1(out), to_pm1(a) * to_pm1(b))


def test_binding_self_inverse():
    a = random_hypervectors(2, 256, generator=torch.Generator().manual_seed(3))
    # a ⊗ a = 𝟙 (all +1)
    assert torch.equal(to_pm1(bind(a, a)), torch.ones(2, 256))


def test_bind_unbind_recovers_operand():
    g = torch.Generator().manual_seed(31)
    value = random_hypervectors(6, 512, generator=g)
    role = random_hypervectors(6, 512, generator=g)
    recovered = bind(bind(value, role), role)
    assert torch.equal(to_pm1(recovered), to_pm1(value))


def test_signed_dot_is_popcount_derived():
    """<u,v> = D − 2·Hamming(u,v), computed by XNOR+popcount."""
    g = torch.Generator().manual_seed(4)
    u = random_hypervectors(5, 320, generator=g)
    v = random_hypervectors(7, 320, generator=g)
    sim = hamming_similarity(u, v)               # (5,7) int32
    ref = to_pm1(u) @ to_pm1(v).t()              # ±1 dot product
    assert sim.dtype == torch.int32
    assert torch.equal(sim.float(), ref)


def test_bundle3_is_majority():
    g = torch.Generator().manual_seed(5)
    a, b, c = (random_hypervectors(3, 192, generator=g) for _ in range(3))
    out = bundle3(a, b, c)
    ref = torch.sign(to_pm1(a) + to_pm1(b) + to_pm1(c))  # no ties (odd count)
    assert torch.equal(to_pm1(out), ref)


def test_bundle_retrieval_survives_distractors():
    """A bound role can recover its filler from a small VSA record."""
    g = torch.Generator().manual_seed(33)
    D = 2048
    fillers = random_hypervectors(5, D, generator=g)
    roles = random_hypervectors(5, D, generator=g)
    records = to_pm1(bind(fillers, roles))
    bundle = sign_to_bit1(records.sum(dim=0, keepdim=True))
    query = bind(bundle, roles[2:3])
    sim = hamming_similarity(query, fillers).squeeze(0)
    assert int(sim.argmax().item()) == 2


def test_sign_to_bit1_zero_is_true():
    z = torch.tensor([[-3, 0, 2, -1]])
    assert torch.equal(to_pm1(sign_to_bit1(z)), torch.tensor([[-1.0, 1.0, 1.0, -1.0]]))


def test_position_codes_are_rolls():
    base = (torch.randint(0, 2, (64,)) * 2 - 1).float()
    codes = position_codes(base, 5)
    assert codes.shape == (5, 64)
    cp = to_pm1(codes)
    assert torch.equal(cp[0], base)
    for i in range(1, 5):
        assert torch.equal(cp[i], torch.roll(base, i))


def test_hierarchical_position_codes_factor_chunk_and_offset():
    D, n, chunk = 64, 10, 4
    g = torch.Generator().manual_seed(36)
    chunk_base = (torch.randint(0, 2, (D,), generator=g) * 2 - 1).float()
    offset_base = (torch.randint(0, 2, (D,), generator=g) * 2 - 1).float()
    codes = hierarchical_position_codes(chunk_base, offset_base, n, chunk=chunk)
    cp = to_pm1(codes)
    for i in range(n):
        b, o = divmod(i, chunk)
        ref = torch.roll(chunk_base, b) * torch.roll(offset_base, o)
        assert torch.equal(cp[i], ref)


def test_position_bind_unbind_is_exact():
    D = 512
    g = torch.Generator().manual_seed(34)
    token = random_hypervectors(1, D, generator=g)
    base = (torch.randint(0, 2, (D,), generator=g) * 2 - 1).float()
    pos = position_codes(base, 8)
    dressed = bind(token, pos[5:6])
    recovered = bind(dressed, pos[5:6])
    assert torch.equal(to_pm1(recovered), to_pm1(token))


def test_binary_equiangular_frame_shape_and_balance():
    frame = binary_equiangular_frame(
        12, 64, n_sweeps=3, generator=torch.Generator().manual_seed(37))
    assert frame.shape == (12, 64)
    assert set(frame.unique().tolist()) <= {-1.0, 1.0}
    sim = frame @ frame.t()
    assert torch.equal(torch.diag(sim), torch.full((12,), 64.0))
    off = sim[~torch.eye(12, dtype=torch.bool)]
    assert off.abs().max() < 64


def test_binary_equiangular_frame_large_inline_fallback():
    frame = binary_equiangular_frame(
        16, 32, generator=torch.Generator().manual_seed(38), max_optimized_codes=4)
    assert frame.shape == (16, 32)
    assert set(frame.unique().tolist()) <= {-1.0, 1.0}


def test_crosstalk_noise_shrinks_with_dimension():
    g = torch.Generator().manual_seed(35)
    small = random_hypervectors(128, 512, generator=g)
    large = random_hypervectors(128, 4096, generator=g)
    small_sim = hamming_similarity(small[:1], small[1:]).float().abs().mean() / 512
    large_sim = hamming_similarity(large[:1], large[1:]).float().abs().mean() / 4096
    assert large_sim < small_sim


def test_quasi_orthogonality():
    """Random hypervectors concentrate near orthogonal (similarity ≈ 0, std≈√D)."""
    D = 4096
    hv = random_hypervectors(64, D, generator=torch.Generator().manual_seed(7))
    sim = hamming_similarity(hv, hv).float()
    off = sim[~torch.eye(64, dtype=torch.bool)]
    assert abs(off.mean().item()) < 0.2 * (D ** 0.5)         # mean ≈ 0
    assert abs(off.std().item() - D ** 0.5) < 0.4 * (D ** 0.5)  # std ≈ √D


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

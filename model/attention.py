"""Native binary multi-head attention — token↔token, fully packed, BEP-trained.

This is the binary-native equivalent of a standard decoder self-attention block.
The persistent stream ``x_i ∈ 𝔹^D`` is **position-free**; position enters only as
an integer ALiBi bias on the score *register* (never bound into the decoded
concept).  Each head (``d_h = D / n_heads``) has its own 1-bit projections
``W_Q^h, W_K^h, W_V^h : 𝔹^D → 𝔹^{d_h}`` (the standard MHA factorisation), and::

    qᵢ, kᵢ, vᵢ = sign(W_{Q,K,V}^h·x)            # BooleanLinear, bit1, per head
    ℓ_ij       = ⟨qᵢ, kⱼ⟩ = d_h − 2·H(qᵢ,kⱼ)    # brute xnor-popcount → int32 register
    ℓ_ij      += b_h·(i−j)  (ALiBi)   and   −∞ for j>i (causal)
    oᵢ         = sign( Σ_{j≤i} n(ℓ_ij)·vⱼ )      # soft: int8 vote bundle  (NEW kernel)
               = v_{argmaxⱼ ℓ_ij}                # hardmax: packed gather
    a          = sign(W_O · concat_h oᵢ)         # BooleanLinear

Every persistent tensor stays packed ``brute.bit1`` (uint64).  The only integers
are *transient registers*: the int32 score ``ℓ``, the int8 attention weights
``n(ℓ)``, and the **int8** vote accumulator inside the value combine.  The
per-head value outputs are merged into the ``W_O`` input by concatenating the
underlying uint64 buffers (never ``cat``/``stack`` on the logical bits), so no
step ever unpacks.

Backward is **BEP** (Boolean error propagation):

  * **value path** (``backward``): the desired head output ``o*`` is routed to the
    value rows ``vⱼ`` weighted by the *same* attention weights (transposed) — the
    forward and backward of the value path are the identical signed-bundle reduce
    (fwd uses ``n_ij``, bwd uses ``n_ji``) — giving ``v*`` → ``W_V^h`` backward.
  * **score / address lane** (``margin_loss``): a per-head Hamming-margin
    objective pushes the helpful key's score above ``θ⁺`` and the best distractor
    below ``θ⁻``, emits binary desired ``q*, k*`` and accumulates ``ΔH`` into
    ``W_Q^h, W_K^h``.  The "helpful key" supervision is the self-supervised
    induction signal.

The soft value-combine ``oᵢ = sign(Σⱼ wᵢⱼ·pm1(vⱼ))`` is an *int8-weight × bit1 →
bit1* reduction.  ``brute`` ships bit1×bit1→int but not int×bit1, so a custom
register-bound kernel (:func:`brute.fast.signed_bundle`) supplies it.  When the
kernel is unavailable this module falls back to a correctness-equivalent
reference (Stage A) that materialises a *transient* int8 accumulator; the kernel
(Stage B) removes the last transient unpack.  ``attn_mode='hardmax'`` is exact
and fully packed in both stages.
"""

from __future__ import annotations

from typing import List, Optional

import torch

import brute
from brute.tensor import Tensor as _BT

import bep
from bep import BepParam, combine_desired, pm1_int
from layers import BooleanLinear
from vsa import sign_to_bit1, to_bit1


# Large negative sentinel for masked (j>i) score cells — never wins an argmax and
# drives the soft weight to 0 after the relu.  Stays well inside int32.
_NEG = -(1 << 24)

# int8 saturation rails for the vote accumulator.
_I8_MIN, _I8_MAX = -127, 127


def _kernel_available() -> bool:
    """True iff the compiled ``brute.signed_bundle`` op is actually registered.

    The Python wrapper ``brute.fast.signed_bundle`` may exist before the C++/Metal
    extension carrying the op has been rebuilt; this checks the *registered op* so
    Stage A transparently falls back to the reference until Stage B is built.
    """
    if not hasattr(brute.fast, "signed_bundle"):
        return False
    try:
        return hasattr(torch.ops.brute, "signed_bundle")
    except Exception:
        return False


# ── packed head split / merge (operate on the uint64 words, never unpack) ───────

def _split_head(x_bit: brute.Tensor, h: int, n_heads: int) -> brute.Tensor:
    """Return head ``h`` of a ``(..., D)`` bit1 tensor as ``(..., d_h)`` bit1.

    Reads the packed buffer directly: ``D`` and ``d_h`` are multiples of 64, so a
    head is a contiguous slice of packed words.  No bit gather, no unpack.
    """
    D = int(x_bit.shape[-1])
    d_h = D // n_heads
    w = d_h // 64                                     # packed words per head
    pb = x_bit._packed_buf                            # (..., D//64) uint64
    lead = list(x_bit.shape[:-1])
    pb_h = pb[..., h * w:(h + 1) * w].contiguous()    # (..., w)
    return _BT._make_bit1_from_packed(pb_h, lead + [d_h])


def _merge_heads(heads: List[brute.Tensor]) -> brute.Tensor:
    """Concatenate per-head ``(..., d_h)`` bit1 tensors into ``(..., D)`` bit1.

    Concatenates the packed uint64 buffers along the word axis (each head is a
    whole number of 64-bit words), then rewraps.  Equivalent to a last-axis
    ``cat`` of the logical bits but with zero unpack / repack.
    """
    H = len(heads)
    d_h = int(heads[0].shape[-1])
    lead = list(heads[0].shape[:-1])
    pb = torch.cat([h._packed_buf_contig() for h in heads], dim=-1)   # (..., H*w)
    return _BT._make_bit1_from_packed(pb, lead + [H * d_h])


# ── signed-bundle value combine (int8-weight × bit1 → bit1) ─────────────────────

def signed_bundle(W_int: torch.Tensor, V_bit: brute.Tensor, *,
                  prefer_kernel: bool = True) -> brute.Tensor:
    """``out[...,m,:] = sign( Σ_n W[...,m,n] · pm1(V[...,n,:]) )``.

    ``W`` is an integer weight tensor ``(B, M, N)`` (int8/-32); ``V`` is bit1
    ``(B, N, D)``.  Returns bit1 ``(B, M, D)``.

    Uses :func:`brute.fast.signed_bundle` (the register-bound kernel) when it is
    available — then ``V`` is read straight from its packed buffer and never
    materialised.  Otherwise falls back to the Stage-A reference: a *transient*
    ±1 unpack of ``V`` and a float reduce, with the materialised vote
    accumulator clamped to **int8** (so ``sign`` is exact — clamping preserves
    sign — and the persistent footprint matches the kernel's).
    """
    if prefer_kernel and _kernel_available():
        return brute.fast.signed_bundle(W_int, V_bit)
    B, M, N = int(W_int.shape[0]), int(W_int.shape[1]), int(W_int.shape[2])
    D = int(V_bit.shape[-1])
    v_pm1 = V_bit.unpack_pm1().reshape(B, N, D)                 # transient ±1 float
    acc = torch.bmm(W_int.to(torch.float32), v_pm1)            # transient register
    acc_i8 = acc.clamp_(_I8_MIN, _I8_MAX).to(torch.int8)        # int8 accumulator
    return sign_to_bit1(acc_i8).reshape(B, M, D)


# ── ALiBi integer slopes ────────────────────────────────────────────────────────

def alibi_slopes(n_heads: int, d_h: int, *, device=None) -> torch.Tensor:
    """Per-head **integer** ALiBi slopes ``b_h`` (locality strength).

    The score register is the *unnormalised* signed dot ``⟨q,k⟩ ∈ [−d_h, d_h]``,
    so an ALiBi bias ``−b_h·(i−j)`` only competes with content when ``b_h`` scales
    with ``d_h`` (a normalised-softmax ALiBi slope of ``O(1)`` would be swamped).
    The slopes therefore span a geometric range from ``1`` (content-dominated head:
    attends by ``q·k`` content, recency only breaks ties) up to ``> 2·d_h``
    (recency-dominated head: one position step outweighs any content swing, so it
    attends to the nearest causal key — the "previous-token" head of the induction
    circuit).  Integer-valued so the bias stays register-exact.
    """
    hi = 2 * d_h + 1                                   # recency dominates content
    if n_heads == 1:
        return torch.tensor([max(1, d_h // 4)], dtype=torch.int32, device=device)
    exps = torch.linspace(0.0, 1.0, n_heads)
    slopes = torch.round(torch.tensor(float(hi)) ** exps).clamp_(min=1, max=1 << 16)
    return slopes.to(torch.int32).to(device)


# ── Binary multi-head self-attention ────────────────────────────────────────────

class BinaryMultiHeadAttention:
    """Token↔token binary attention with BEP backward.  Packed end-to-end."""

    def __init__(self, D: int, n_heads: int, *, name: str,
                 attn_mode: str = "soft", alibi: bool = True, causal: bool = True,
                 causal_strict: bool = False, attn_band: int = 1, value_proj: bool = True,
                 alibi_slopes_override=None,
                 generator: Optional[torch.Generator] = None, device=None,
                 init_inertia: int = 1, update_clip: Optional[int] = None):
        if D % n_heads != 0:
            raise ValueError(f"D={D} not divisible by n_heads={n_heads}")
        d_h = D // n_heads
        if d_h % 64 != 0:
            raise ValueError(
                f"head dim d_h={d_h} must be a multiple of 64 to stay packed "
                f"(choose D / n_heads so D//n_heads % 64 == 0)")
        if attn_mode not in ("soft", "hardmax"):
            raise ValueError("attn_mode must be 'soft' or 'hardmax'")
        if attn_band < 1:
            raise ValueError("attn_band must be >= 1")
        self.D, self.H, self.d_h = D, n_heads, d_h
        self.attn_mode = attn_mode
        self.alibi = alibi
        self.causal = causal
        self.causal_strict = causal_strict     # exclude self (j<i) — recency picks i-1
        self.band = int(attn_band)
        self.value_proj = value_proj
        self.name = name

        def mk(nm, out):
            return BooleanLinear(D, out, name=f"{name}.{nm}", generator=generator,
                                 device=device, init_inertia=init_inertia,
                                 update_clip=update_clip)
        # Per-head projections D → d_h (standard MHA factorisation).
        self.Wq = [mk(f"Wq.h{h}", d_h) for h in range(n_heads)]
        self.Wk = [mk(f"Wk.h{h}", d_h) for h in range(n_heads)]
        if value_proj:
            # Faithful transformer: learned value projection + output projection.
            self.Wv = [mk(f"Wv.h{h}", d_h) for h in range(n_heads)]
            self.Wo = mk("Wo", D)
        else:
            # Raw-concept-copy attention (diagnostic retrieval mode): the value is the
            # source concept's head slice (no W_V), the output is the gathered
            # concatenation (no W_O).  The retrieved value is therefore a *clean*
            # codeword the LM head decodes directly — exact value transport for
            # copy / retrieval, with only the Q/K address lane learned.
            self.Wv = None
            self.Wo = None

        if alibi_slopes_override is not None:
            sl = torch.as_tensor(list(alibi_slopes_override), dtype=torch.int32, device=device)
            if sl.numel() != n_heads:
                raise ValueError(f"alibi_slopes_override must have {n_heads} entries")
            self.slopes = sl
        elif alibi:
            self.slopes = alibi_slopes(n_heads, d_h, device=device)
        else:
            self.slopes = torch.zeros(n_heads, dtype=torch.int32, device=device)
        self.device = device
        self._pos_cache: dict = {}
        self._cache: dict = {}

    def params(self) -> List[BepParam]:
        ps: List[BepParam] = []
        for h in range(self.H):
            ps += self.Wq[h].params() + self.Wk[h].params()
            if self.value_proj:
                ps += self.Wv[h].params()
        if self.value_proj:
            ps += self.Wo.params()
        return ps

    # ── score-register geometry (relative position + causal mask), cached per n ──
    def _geometry(self, n: int, device):
        c = self._pos_cache
        if c.get("n") != n or c.get("dev") != device:
            i = torch.arange(n, device=device).unsqueeze(1)
            j = torch.arange(n, device=device).unsqueeze(0)
            relpos = (i - j).to(torch.int32)                 # (n,n)  i−j  (≥0 below diag)
            if not self.causal:
                keep = torch.ones(n, n, dtype=torch.bool, device=device)
            else:
                keep = (j < i) if self.causal_strict else (j <= i)
            self._pos_cache = {"n": n, "dev": device, "relpos": relpos, "keep": keep}
        c = self._pos_cache
        return c["relpos"], c["keep"]

    def _scores_head(self, qh_b: brute.Tensor, kh_b: brute.Tensor,
                     relpos: torch.Tensor, keep: torch.Tensor, slope: int) -> torch.Tensor:
        """Int32 score register ``ℓ`` for one (head, batch): (n,n)."""
        ell = brute.fast.matmul(qh_b, kh_b)                  # (n,n) int32 = d_h − 2H
        if self.alibi and slope != 0:
            ell = ell - slope * relpos                       # ALiBi: −b·(i−j)
        if self.causal:
            ell = ell.masked_fill(~keep, _NEG)
        return ell

    # ── forward ────────────────────────────────────────────────────────────────
    def forward(self, x_bit: brute.Tensor) -> brute.Tensor:
        B, n, D = x_bit.shape
        H, d_h = self.H, self.d_h
        dev = x_bit._packed_buf.device
        xf = x_bit.reshape(B * n, D)
        relpos, keep = self._geometry(n, dev)

        o_heads: List[brute.Tensor] = []
        q_head_bits: List[brute.Tensor] = []
        k_head_bits: List[brute.Tensor] = []
        idx_heads: List[Optional[torch.Tensor]] = []      # hardmax: (B,n) argmax source
        w_heads: List[Optional[torch.Tensor]] = []        # soft: (B,n,n) int8 weights
        for h in range(H):
            qh, _ = self.Wq[h].forward(xf)                # (M, d_h)
            kh, _ = self.Wk[h].forward(xf)
            if self.value_proj:
                vh, _ = self.Wv[h].forward(xf)
                vh = vh.reshape(B, n, d_h)
            else:
                vh = _split_head(x_bit, h, H)             # raw source concept slice
            qh = qh.reshape(B, n, d_h)
            kh = kh.reshape(B, n, d_h)
            q_head_bits.append(qh)
            k_head_bits.append(kh)
            slope = int(self.slopes[h].item())
            if self.attn_mode == "hardmax":
                idx = torch.empty(B, n, dtype=torch.int64, device=dev)
                for b in range(B):
                    ell = self._scores_head(qh[b], kh[b], relpos, keep, slope)
                    idx[b] = ell.argmax(dim=1)
                vh_flat = vh.reshape(B * n, d_h)
                gidx = (torch.arange(B, device=dev).unsqueeze(1) * n + idx).reshape(-1)
                o_h = vh_flat.index_select(0, gidx).reshape(B, n, d_h)
                o_heads.append(o_h)
                idx_heads.append(idx)
                w_heads.append(None)
            else:
                w = torch.empty(B, n, n, dtype=torch.int8, device=dev)
                for b in range(B):
                    ell = self._scores_head(qh[b], kh[b], relpos, keep, slope)
                    mx = ell.max(dim=1, keepdim=True).values            # (n,1)
                    wb = (ell - mx + self.band).clamp_(min=0, max=_I8_MAX)
                    # masked cells (j>i, and fully-masked rows) carry zero weight.
                    wb = wb.masked_fill(ell <= (_NEG // 2), 0)
                    w[b] = wb.to(torch.int8)
                o_h = signed_bundle(w, vh)                              # (B,n,d_h) bit1
                o_heads.append(o_h)
                idx_heads.append(None)
                w_heads.append(w)

        o_cat = _merge_heads(o_heads)                                  # (B,n,D)
        if self.value_proj:
            a_bit, _ = self.Wo.forward(o_cat.reshape(B * n, D))
            a_bit = a_bit.reshape(B, n, D)
        else:
            a_bit = o_cat                                              # gathered concept

        self._cache = {
            "B": B, "n": n, "x_bit": x_bit,
            "q_head_bits": q_head_bits, "k_head_bits": k_head_bits,
            "idx_heads": idx_heads, "w_heads": w_heads,
        }
        return a_bit

    # ── value-path backward (BEP) — returns desired x* (B,n,D) ──────────────────
    def backward(self, a_star: brute.Tensor) -> brute.Tensor:
        B, n = self._cache["B"], self._cache["n"]
        H, d_h, D = self.H, self.d_h, self.D
        dev = a_star._packed_buf.device
        # W_O backward → desired head-concat output o* (or o* = a* when no W_O).
        if self.value_proj:
            o_star_cat = self.Wo.backward(a_star.reshape(B * n, D)).reshape(B, n, D)
        else:
            o_star_cat = a_star

        x_bit = self._cache["x_bit"]
        v_star_heads: List[brute.Tensor] = []
        x_star: Optional[brute.Tensor] = None
        for h in range(H):
            o_star_h = _split_head(o_star_cat, h, H)                  # (B,n,d_h)
            idx = self._cache["idx_heads"][h]
            w = self._cache["w_heads"][h]
            if w is not None:
                # soft: v* = sign(Σ_i w_ij · pm1(o*_i)) = signed_bundle(wᵀ, o*).
                w_t = w.transpose(1, 2).contiguous()                 # (B,n,n) w_ji
                v_star_h = signed_bundle(w_t, o_star_h)              # (B,n,d_h)
            else:
                # hardmax: scatter o*_i into its argmax source j; sign the vote.
                # A source can receive O(B*n) queries, so int8 would wrap here.
                o_pm1 = pm1_int(o_star_h, torch.int16).reshape(B, n, d_h)
                acc = torch.zeros(B, n, d_h, dtype=torch.int16, device=dev)
                src = idx.unsqueeze(-1).expand(B, n, d_h)
                acc.scatter_add_(1, src, o_pm1)
                v_star_h = sign_to_bit1(acc)
            if self.value_proj:
                x_h = self.Wv[h].backward(v_star_h.reshape(B * n, d_h)).reshape(B, n, D)
                x_star = x_h if x_star is None else combine_desired(x_star, x_h, x_bit)
            else:
                v_star_heads.append(v_star_h)
        if not self.value_proj:
            # value = raw source concept slice ⇒ v*_h *is* the desired x at head h.
            x_star = _merge_heads(v_star_heads)
        return x_star

    # ── score / address-lane backward (BEP Hamming-margin objective) ────────────
    def margin_loss(self, matched: torch.Tensor, *, theta_pos: float = 0.5,
                    theta_neg: float = 0.0, active_rows: Optional[torch.Tensor] = None) -> float:
        """Push ``⟨q_i, k_matched⟩`` above ``θ⁺·d_h`` and the best distractor below
        ``θ⁻·d_h``, per head; emit binary desired ``q*, k*`` and accumulate ΔH.

        ``matched`` is ``(B, n)`` long: the supervising key index for each query
        (the self-supervised induction match; ``-1`` = no supervision).  Mirrors
        Each head trains its own ``W_Q^h, W_K^h``.
        """
        B, n = self._cache["B"], self._cache["n"]
        H, d_h = self.H, self.d_h
        dev = self._cache["q_head_bits"][0]._packed_buf.device
        relpos, keep = self._geometry(n, dev)
        tp, tn = theta_pos * d_h, theta_neg * d_h
        matched = matched.to(dev).reshape(B, n).long()
        in_range = (matched >= 0) & (matched < n)
        safe = matched.clamp(0, max(n - 1, 0))
        keep_row = torch.gather(keep.unsqueeze(0).expand(B, n, n), 2, safe.unsqueeze(-1)).squeeze(-1)
        valid = in_range & keep_row
        if active_rows is not None:
            valid = valid & active_rows.to(dev, torch.bool).reshape(B, n)
        if int(valid.sum().item()) == 0:
            return 0.0

        total_loss = 0.0
        n_valid = max(int(valid.sum().item()), 1)
        prev_active = bep.active_mask()
        for h in range(H):
            qh = self._cache["q_head_bits"][h]                       # (B,n,d_h)
            kh = self._cache["k_head_bits"][h]
            slope = int(self.slopes[h].item())
            score = torch.empty(B, n, n, dtype=torch.int32, device=dev)
            for b in range(B):
                score[b] = self._scores_head(qh[b], kh[b], relpos, keep, slope)
            pos_score = torch.gather(score, 2, safe.unsqueeze(-1)).squeeze(-1)   # (B,n)
            distract = score.clone()
            distract.scatter_(2, safe.unsqueeze(-1), _NEG)
            neg_score, neg_idx = distract.max(dim=2)                            # (B,n)

            pos_viol = (pos_score < tp) & valid
            neg_viol = (neg_score > tn) & valid
            loss = (((tp - pos_score).clamp_min(0) + (neg_score - tn).clamp_min(0)) * valid).sum()
            total_loss += float(loss.item()) / n_valid

            # desired query: matched → equal k_match;  distractor → anti k_neg.
            qh_b = qh.bool(); kh_b = kh.bool()
            idx_m = safe.unsqueeze(-1).expand(B, n, d_h)
            idx_d = neg_idx.unsqueeze(-1).expand(B, n, d_h)
            k_match = torch.gather(kh_b, 1, idx_m)
            k_neg = torch.gather(kh_b, 1, idx_d)
            pv = pos_viol.unsqueeze(-1); nv = neg_viol.unsqueeze(-1)
            q_des = torch.where(nv, ~k_neg, qh_b)
            q_des = torch.where(pv, k_match, q_des)

            # desired key at the matched slot: agree with q_i (majority over queries).
            dk = torch.zeros(B, n, d_h, dtype=torch.int32, device=dev)
            q_pm1 = pm1_int(qh, torch.int32)
            dk.scatter_add_(1, idx_m, q_pm1 * pos_viol.unsqueeze(-1))
            k_des = torch.where(dk > 0, torch.ones_like(kh_b),
                                torch.where(dk < 0, torch.zeros_like(kh_b), kh_b))

            bep.set_active((pos_viol | neg_viol).reshape(-1))
            self.Wq[h].backward(to_bit1(q_des).reshape(B * n, d_h))
            self.Wk[h].backward(to_bit1(k_des).reshape(B * n, d_h))
        bep.set_active(prev_active)
        return total_loss / H

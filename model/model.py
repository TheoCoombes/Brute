"""HÆMMR v3 — a binary-first, concept-native autoregressive language model.

v3 tightens the v2 architecture around three non-negotiables: a single canonical
linear-time exact-memory path, a non-destructive vertical transport path for
depth, and a strict separation between compressed memory, exact memory, and
learned prior memory.

    token → E_lex(t)  ── input-bind ──▶  position-free concept c^0,  u^0 = ±1
        ┌──────────────────────────────────────────────────────────────────┐ ×L
        │  c^ℓ = sign(u^ℓ)                       # public binary concept       │
        │  r_bsr = delta-BSR(c^ℓ)                # compressed discourse        │
        │  r_epi = EpisodicRing(c^ℓ, POS)        # exact in-window recall      │
        │  r_hop = HopfieldPriors(c^ℓ)           # learned global priors       │
        │  r_ff  = channel-mix(c^ℓ)              # binary MLP                  │
        │  u^{ℓ+1} = clip_S( λ·u^ℓ + Σ_r α_r·r_r )   # clipped int highway     │
        └──────────────────────────────────────────────────────────────────┘
        c^L = sign(u^L)
        ── lex-proj ──▶ ℓ̂  ── min-Hamming decode vs FIXED E_lex ──▶ logits
                                (+ β·⟨c^L, E_sem⟩ rerank  + episodic shortlist)

The **clipped integer concept highway** ``u`` is the one concession to pure
1-bit activations (a small signed integer field, ``c = sign(u)``); every learned
matrix, codebook and bank stays 1-bit.  Each block's branches all read the same
``c^ℓ`` and emit a binary proposal; the highway accumulates them with fixed
power-of-two scales and a carry coefficient, binarising **once** per block.
There is no majority-vote residual merge.

Position never enters the decoded concept; it lives only in the episodic address
lane.  The lexical codebook is fixed after BEF initialisation; the concept→lexical
projection and the semantic rerank prototypes are learned.  Training is BOLD
bit-flips with optional boundary gating, flip-dropout, label smoothing, and a
local Hamming-margin objective on the episodic address lane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

import brute
from brute.tensor import Tensor as _BT

from bold import BoldParam
from layers import (
    BSR, BooleanLinear, DiagBind, EpisodicSlotMemory, HopfieldBank,
    TokenCodebook, _LazyPM1Cache,
)
from vsa import hierarchical_position_codes, sign_to_bit1, to_i8_pm1


IGNORE_INDEX = -1


@dataclass
class HaemmrConfig:
    vocab_size: int = 2048
    D: int = 1024                       # concept hypervector dimension
    n_layers: int = 2
    d_ff: int = 2048                    # channel-mix hidden width
    # episodic ring memory (exact local recall)
    epi_read_k: int = 1                 # top-k read width (1 ⇒ exact single-slot)
    epi_window: Optional[int] = None    # fixed ring width W (None ⇒ chunk-bounded)
    epi_chunk: int = 64                 # chunk size C for the chunked ring search
    # latent Hopfield priors (static learned)
    n_slots: int = 256                  # learned static slots (M)
    top_k: int = 15                     # WTA width (odd → no ties)
    # delta-BSR
    use_bsr: bool = True                # disable for packed-only hardware profiles
    decay_shifts: tuple = (1, 2, 3, 4, 0)   # power-of-two shifts; 0 = permanent
    bsr_erase: tuple = (1,)             # per-channel-group erase strengths
    bsr_write: tuple = (1,)             # per-channel-group write strengths
    bsr_state_clip: int = 31            # bounded BSR accumulator |A| ≤ clip
    # clipped integer concept highway
    highway_clip: int = 15              # |u| ≤ S (int5-style bounded score field)
    highway_carry: int = 1             # λ identity-carry coefficient (power of two)
    # Fixed power-of-two branch scales.  Exact recall must be reliable, so the
    # episodic branch is given a dominant scale: when it fires it can override the
    # carried concept (e.g. copy a literal); when it has no match it echoes the
    # concept and the carry is preserved.  All scales are tunable per deployment.
    alpha_bsr: int = 1
    alpha_epi: int = 4
    alpha_hop: int = 1
    alpha_ff: int = 1
    highway_nu: Optional[float] = None  # boundary gate: |u| ≤ ν·S propagates; None=off
    # multiscale memory horizons across depth
    multiscale: bool = True            # deeper layers ⇒ larger windows + slower decay
    # representation / positions
    use_position: bool = True          # positional addressing in the episodic lane
    pos_chunk: int = 256               # hierarchical position chunk size C
    # decode
    structured_codebook: bool = True   # fixed BEF lexical codebook (else random fixed)
    bef_alpha: float = 1.0
    bef_sweeps: int = 30
    sem_weight: float = 0.5            # semantic-rerank logit weight (0 disables)
    sem_flip_scale: float = 0.5        # E_sem flip-rate relative to transforms
    epi_bonus_weight: float = 0.0      # episodic shortlist seeding (decode-time only)
    # training mitigations and margin objective
    boundary_nu: Optional[float] = None  # BEP per-linear boundary gating |z|≤νD; None=off
    label_smoothing: float = 0.0
    flip_dropout: float = 0.0          # activation bit-flip rate in the training forward
    margin_theta_pos: float = 0.5      # matched-pair margin (fraction of D)
    margin_theta_neg: float = 0.0      # distractor margin
    margin_weight: float = 1.0
    # misc
    logit_temp: Optional[float] = None  # train-time logit scale; default sqrt(D)
    seed: int = 0

    def __post_init__(self):
        if self.logit_temp is None:
            self.logit_temp = float(self.D) ** 0.5
        if self.D <= 0:
            raise ValueError("D must be positive")
        if self.epi_read_k <= 0:
            raise ValueError("epi_read_k must be positive")
        if self.epi_window is not None and self.epi_window <= 0:
            raise ValueError("epi_window must be positive when set")
        if self.epi_chunk <= 0:
            raise ValueError("epi_chunk must be positive")
        if self.pos_chunk <= 0:
            raise ValueError("pos_chunk must be positive")
        if self.highway_clip <= 0:
            raise ValueError("highway_clip must be positive")
        if self.highway_clip > 127:
            raise ValueError("highway_clip must fit in int8 (<= 127)")
        if self.highway_carry < 0:
            raise ValueError("highway_carry must be non-negative")
        for name in ("highway_carry", "alpha_bsr", "alpha_epi", "alpha_hop", "alpha_ff"):
            if int(getattr(self, name)) != getattr(self, name):
                raise ValueError(f"{name} must be an integer scale")
        if self.sem_weight < 0:
            raise ValueError("sem_weight must be non-negative")
        if self.epi_bonus_weight < 0:
            raise ValueError("epi_bonus_weight must be non-negative")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not 0.0 <= self.flip_dropout < 1.0:
            raise ValueError("flip_dropout must be in [0, 1)")
        if self.boundary_nu is not None and self.boundary_nu < 0:
            raise ValueError("boundary_nu must be non-negative")
        if self.highway_nu is not None and self.highway_nu < 0:
            raise ValueError("highway_nu must be non-negative")


# ── multiscale depth schedules ─────────────────────────────────────────────────

def _layer_window(base: Optional[int], ell: int, multiscale: bool) -> Optional[int]:
    """Episodic ring width for layer ``ell`` — deeper layers get larger windows."""
    if base is None or not multiscale:
        return base
    return base * (ell + 1)


def _layer_decay_shifts(base_shifts, ell: int, multiscale: bool) -> tuple:
    """BSR decay palette for layer ``ell`` — deeper layers decay more slowly.

    Each non-permanent shift increases with depth (slower decay, longer horizon);
    permanent groups (``s == 0``) stay permanent.
    """
    if not multiscale:
        return tuple(base_shifts)
    return tuple((0 if s == 0 else s + ell) for s in base_shifts)


# ── channel-mixing MLP (two Boolean linears) ───────────────────────────────────

class ChannelMix:
    def __init__(self, D: int, d_ff: int, *, name: str, generator=None, device=None,
                 boundary_nu=None):
        self.D, self.d_ff = D, d_ff
        self.lin1 = BooleanLinear(D, d_ff, name=f"{name}.lin1", generator=generator,
                                  device=device, boundary_nu=boundary_nu)
        self.lin2 = BooleanLinear(d_ff, D, name=f"{name}.lin2", generator=generator,
                                  device=device, boundary_nu=boundary_nu)

    def params(self) -> List[BoldParam]:
        return self.lin1.params() + self.lin2.params()

    def forward(self, x_bit: brute.Tensor) -> brute.Tensor:
        h_bit, _ = self.lin1.forward(x_bit)
        m_bit, _ = self.lin2.forward(h_bit)
        return m_bit

    def backward(self, S: torch.Tensor) -> torch.Tensor:
        g = self.lin2.backward(S)
        g = self.lin1.backward(g)
        return g


# ── one Boolean block over the clipped integer concept highway ──────────────────

class Block:
    """A v3 block: every branch reads ``c^ℓ = sign(u^ℓ)``; the clipped integer
    highway accumulates their fixed-scale proposals and binarises once."""

    def __init__(self, cfg: HaemmrConfig, *, idx: int, generator=None, device=None):
        D = cfg.D
        nm = f"blk{idx}"
        nu = cfg.boundary_nu
        self.use_bsr = cfg.use_bsr
        shifts = _layer_decay_shifts(cfg.decay_shifts, idx, cfg.multiscale)
        window = _layer_window(cfg.epi_window, idx, cfg.multiscale)
        if self.use_bsr:
            self.bsr = BSR(D, name=f"{nm}.bsr", decay_shifts=shifts,
                           erase=cfg.bsr_erase, write=cfg.bsr_write,
                           state_clip=cfg.bsr_state_clip,
                           generator=generator, device=device, boundary_nu=nu)
        else:
            self.bsr = None
        self.epi = EpisodicSlotMemory(D, name=f"{nm}.epi", read_k=cfg.epi_read_k,
                                      n_slots=window, epi_chunk=cfg.epi_chunk,
                                      generator=generator, device=device, boundary_nu=nu)
        self.hop = HopfieldBank(D, cfg.n_slots, cfg.top_k, name=f"{nm}.hop",
                                generator=generator, device=device)
        self.mix = ChannelMix(D, cfg.d_ff, name=f"{nm}.mix", generator=generator,
                              device=device, boundary_nu=nu)
        # fixed power-of-two branch scales + identity carry + clip
        self.a_bsr = int(cfg.alpha_bsr)
        self.a_epi = int(cfg.alpha_epi)
        self.a_hop = int(cfg.alpha_hop)
        self.a_ff = int(cfg.alpha_ff)
        self.carry = int(cfg.highway_carry)
        self.clip = int(cfg.highway_clip)
        self.highway_nu = cfg.highway_nu
        self._epi_read: Optional[brute.Tensor] = None
        self._cache: dict = {}

    def params(self) -> List[BoldParam]:
        ps = []
        if self.use_bsr:
            ps += self.bsr.params()
        return ps + self.epi.params() + self.hop.params() + self.mix.params()

    def forward(self, u: torch.Tensor, pos_bit: brute.Tensor, flip_fn) -> torch.Tensor:
        """``u`` (B,n,D) int8 highway → new highway ``u'`` (B,n,D)."""
        B, n, D = u.shape
        c_bit = sign_to_bit1(u.reshape(B * n, D)).reshape(B, n, D)   # public concept
        c_in, flip_mask = flip_fn(c_bit)                            # branch input (noised)

        u_pre = self.carry * u.to(torch.int16)
        if self.use_bsr:
            r_bsr = self.bsr.forward(c_in)
            u_pre = u_pre + self.a_bsr * to_i8_pm1(r_bsr).to(torch.int16)
        r_epi = self.epi.forward(c_in, pos_bit)
        self._epi_read = r_epi
        u_pre = u_pre + self.a_epi * to_i8_pm1(r_epi).to(torch.int16)
        r_hop = self.hop.forward(c_in.reshape(B * n, D)).reshape(B, n, D)
        u_pre = u_pre + self.a_hop * to_i8_pm1(r_hop).to(torch.int16)
        r_ff = self.mix.forward(c_in.reshape(B * n, D)).reshape(B, n, D)
        u_pre = u_pre + self.a_ff * to_i8_pm1(r_ff).to(torch.int16)

        u_new = u_pre.clamp(-self.clip, self.clip).to(torch.int8)
        self._cache = {"u": u, "u_pre": u_pre, "flip_mask": flip_mask, "shape": (B, n, D)}
        return u_new

    def backward(self, S_u_new: torch.Tensor) -> torch.Tensor:
        if not S_u_new.is_floating_point():
            S_u_new = S_u_new.to(torch.float32)
        B, n, D = self._cache["shape"]
        u = self._cache["u"]
        u_pre = self._cache["u_pre"]
        flip_mask = self._cache["flip_mask"]

        sat = (u_pre.abs() < self.clip).to(S_u_new.dtype)          # clip Jacobian
        S_pre = S_u_new * sat
        S_u = self.carry * S_pre                                   # identity carry path

        S_c = torch.zeros(B, n, D, dtype=S_u_new.dtype, device=S_u_new.device)
        if self.use_bsr:
            S_c = S_c + self.bsr.backward(self.a_bsr * S_pre)
        S_c = S_c + self.epi.backward(self.a_epi * S_pre)
        S_c = S_c + self.hop.backward(
            (self.a_hop * S_pre).reshape(B * n, D)).reshape(B, n, D)
        S_c = S_c + self.mix.backward(
            (self.a_ff * S_pre).reshape(B * n, D)).reshape(B, n, D)

        if flip_mask is not None:                                  # un-flip dropout noise
            S_c = S_c * flip_mask
        if self.highway_nu is not None:                           # boundary gate on sign(u)
            elig = (u.abs() <= self.highway_nu * self.clip).to(S_c.dtype)
            S_c = S_c * elig
        return S_u + S_c                                          # c = sign(u): pass-through


# ── the model ────────────────────────────────────────────────────────────────

class HaemmrLM:
    def __init__(self, cfg: HaemmrConfig, *, device=None):
        self.cfg = cfg
        self.training = True
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        nu = cfg.boundary_nu

        self.codebook = TokenCodebook(cfg.vocab_size, cfg.D, name="E",
                                      structured=cfg.structured_codebook,
                                      bef_alpha=cfg.bef_alpha, bef_sweeps=cfg.bef_sweeps,
                                      sem=cfg.sem_weight > 0, sem_flip_scale=cfg.sem_flip_scale,
                                      generator=gen, device=self.device)
        self.input_bind = DiagBind(cfg.D, name="in", generator=gen, device=self.device)
        self.blocks = [Block(cfg, idx=i, generator=gen, device=self.device)
                       for i in range(cfg.n_layers)]
        # concept → lexical frame projection (learned binary projection)
        self.lex_proj = BooleanLinear(cfg.D, cfg.D, name="lex", generator=gen,
                                      device=self.device, boundary_nu=nu)

        # two orthogonal position bases for hierarchical codes (address lane only)
        self.pos_chunk_base = (torch.randint(0, 2, (cfg.D,), generator=gen).float() * 2 - 1).to(self.device)
        self.pos_offset_base = (torch.randint(0, 2, (cfg.D,), generator=gen).float() * 2 - 1).to(self.device)
        self._pos_cache: dict = {}
        self._neutral_pos_cache: dict = {}
        self._fwd_cache: dict = {}

    # ── parameters / checkpoint ──────────────────────────────────────────────
    def parameters(self) -> List[BoldParam]:
        ps = self.codebook.params() + self.input_bind.params()
        for b in self.blocks:
            ps += b.params()
        ps += self.lex_proj.params()
        return ps

    def num_bit_parameters(self) -> int:
        return sum(int(torch.tensor(p.shape).prod().item()) for p in self.parameters())

    def state_dict(self) -> dict:
        return {
            "cfg": self.cfg.__dict__,
            "pos_chunk_base": self.pos_chunk_base.detach().cpu().clone(),
            "pos_offset_base": self.pos_offset_base.detach().cpu().clone(),
            "E_lex_packed": self.codebook.E_lex._packed_buf.detach().cpu().clone(),
            "params": {p.name: p.state_dict() for p in self.parameters()},
        }

    def load_state_dict(self, sd: dict) -> None:
        self.pos_chunk_base = sd["pos_chunk_base"].to(self.device)
        self.pos_offset_base = sd["pos_offset_base"].to(self.device)
        if "E_lex_packed" in sd:
            packed = sd["E_lex_packed"].to(self.device)
            self.codebook.E_lex = _BT._make_bit1_from_packed(
                packed, [self.cfg.vocab_size, self.cfg.D])
            self.codebook._E_lex_pm1 = None
        byname = {p.name: p for p in self.parameters()}
        for name, psd in sd["params"].items():
            if name in byname:
                byname[name].load_state_dict(psd)
        self._pos_cache.clear()
        self._neutral_pos_cache.clear()

    # ── positions (address lane only; never bound into the decoded concept) ────
    def _positions(self, n: int):
        if self._pos_cache.get("n", -1) < n:
            codes = hierarchical_position_codes(self.pos_chunk_base, self.pos_offset_base,
                                                n, chunk=self.cfg.pos_chunk)
            self._pos_cache = {"n": n, "bit": codes}
        return self._pos_cache["bit"][:n]

    def _position_lane(self, n: int):
        """Address-lane position codes.

        ``use_position=False`` turns the positional part of episodic addressing
        into a neutral all-ones role rather than disabling the slot path.
        """
        if self.cfg.use_position:
            return self._positions(n)
        if self._neutral_pos_cache.get("n", -1) < n:
            self._neutral_pos_cache = {
                "n": n,
                "bit": brute.ones(n, self.cfg.D, dtype=brute.bit1, device=self.device),
            }
        return self._neutral_pos_cache["bit"][:n]

    # ── flip-dropout (train against the deployment bit-flip noise model) ──────
    def _flip_noise(self, c_bit: brute.Tensor):
        rate = self.cfg.flip_dropout
        if not self.training or rate <= 0:
            return c_bit, None
        flip = torch.rand(c_bit.shape, device=self.device) < rate         # bool mask
        flip_bit = brute.as_tensor(flip, dtype=brute.bit1).to(self.device)
        noised = brute.fast.bitwise_xor(c_bit, flip_bit)                  # XOR flips bits
        mask_pm1 = torch.where(flip, -1.0, 1.0)                            # un-flip factor
        return noised, mask_pm1

    # ── episodic shortlist seeding (decode-time only) ─────────────────────────
    def _episodic_bonus(self, B: int, n: int) -> torch.Tensor:
        """Top episodic lexical retrieval → additive logit bonus (seeds shortlist).

        Uses the final block's episodic read concept, projected to the lexical
        frame and decoded against the fixed codebook.  Computed at decode time;
        it seeds the shortlist without overriding the concept-native head.
        """
        e = self.blocks[-1]._epi_read
        if e is None:
            return torch.zeros(B * n, self.cfg.vocab_size, device=self.device)
        ell_e = brute.fast.matmul_sign(e.reshape(B * n, self.cfg.D),
                                       self.lex_proj.W.bit, self.cfg.D)
        return self.codebook.decode(ell_e).float()

    # ── forward ────────────────────────────────────────────────────────────
    def forward(self, ids: torch.Tensor, *, score_extras: bool = False) -> torch.Tensor:
        """``ids`` (B, n) long → logits (B*n, V) signed similarities.

        ``score_extras`` adds the decode-time episodic shortlist bonus (used by
        sampling / streaming inference, not by the training loss).
        """
        ids = ids.to(self.device)
        B, n = ids.shape
        D = self.cfg.D
        pos_bit = self._position_lane(n)

        emb = self.codebook.embed(ids)                          # fixed lexical code
        c0 = self.input_bind.forward(emb)                       # learned concept dressing
        u = to_i8_pm1(c0)                                       # int8 concept highway u^0 (±1)
        for blk in self.blocks:
            u = blk.forward(u, pos_bit, self._flip_noise)
        cL_bit = sign_to_bit1(u.reshape(B * n, D)).reshape(B, n, D)   # final concept c^L

        cL_flat = cL_bit.reshape(B * n, D)
        ell_bit, _ = self.lex_proj.forward(cL_flat)             # ℓ̂ lexical frame
        lex_logits = self.codebook.decode(ell_bit)             # ⟨ℓ̂, E_lex⟩

        if self.cfg.sem_weight > 0:
            sem_logits = self.codebook.decode_sem(cL_flat)     # ⟨c^L, E_sem⟩
            logits = lex_logits.float() + self.cfg.sem_weight * sem_logits.float()
            sem_used = True
        else:
            logits = lex_logits.float()
            sem_used = False

        cache_data = {"B": B, "n": n, "ell_bit": ell_bit, "cL_bit": cL_bit,
                      "sem_used": sem_used}
        pm1_sources = {"ell_pm1": ("ell_bit", (B * n, D))}
        if sem_used:
            pm1_sources["cL_pm1"] = ("cL_bit", (B * n, D))
        self._fwd_cache = _LazyPM1Cache(cache_data, pm1_sources=pm1_sources)

        if score_extras and self.cfg.epi_bonus_weight > 0:
            logits = logits + self.cfg.epi_bonus_weight * self._episodic_bonus(B, n)
        return logits

    # ── loss + BOLD backward ─────────────────────────────────────────────────
    def loss_and_backward(self, logits: torch.Tensor, targets: torch.Tensor,
                          matched: Optional[torch.Tensor] = None) -> dict:
        """CE loss + BOLD backward.  ``matched`` (B,n) optionally supplies episodic
        margin supervision (slot id each query should retrieve, -1 = none)."""
        cfg = self.cfg
        B, n = self._fwd_cache["B"], self._fwd_cache["n"]
        D = cfg.D
        V = cfg.vocab_size
        inv = 1.0 / float(cfg.logit_temp)
        tgt = targets.to(self.device).reshape(-1).long()        # (B*n,)
        valid = tgt != IGNORE_INDEX
        n_valid = int(valid.sum().item())

        scaled = logits.float() * inv
        logp = torch.log_softmax(scaled, dim=1)
        safe_tgt = tgt.clamp_min(0)
        nll = -logp.gather(1, safe_tgt.unsqueeze(1)).squeeze(1)[valid]
        loss = float(nll.mean().item()) if n_valid else 0.0
        pred = scaled.argmax(dim=1)
        acc = float((pred[valid] == tgt[valid]).float().mean().item()) if n_valid else 0.0

        # Native Boolean backward signal: for each mistake, decrease the
        # winning wrong logit and increase the target logit. CE remains the
        # reported metric; BOLD receives ternary variation, not a dense FP grad.
        S_logits = torch.zeros_like(logits, dtype=torch.int8)
        wrong = valid & (pred != tgt)
        if bool(wrong.any()):
            rows = wrong.nonzero(as_tuple=False).squeeze(1)
            S_logits[rows, pred[rows]] = 1
            S_logits[rows, safe_tgt[rows]] = -1

        # decode backward (lexical + optional semantic), both into the concept c^L
        g_ell = self.codebook.decode_backward(S_logits, self._fwd_cache["ell_bit"])
        g_cL = self.lex_proj.backward(g_ell)                    # (B*n, D)
        if self._fwd_cache["sem_used"]:
            g_cL = g_cL + self.codebook.decode_sem_backward(S_logits, self._fwd_cache["cL_bit"])
        S_u = g_cL.reshape(B, n, D)                             # c^L = sign(u^L): pass-through

        # margin objective on every block's episodic address lane (uses fwd cache)
        margin_val = 0.0
        if matched is not None and cfg.margin_weight > 0:
            for blk in self.blocks:
                margin_val += blk.epi.margin_loss(
                    matched.to(self.device), theta_pos=cfg.margin_theta_pos,
                    theta_neg=cfg.margin_theta_neg, weight=cfg.margin_weight)

        for blk in reversed(self.blocks):
            S_u = blk.backward(S_u)

        g_c0 = S_u                                              # u^0 = pm1(c0): pass-through
        self.input_bind.backward(g_c0)                         # trains the input mask;
        #                                                        E_lex is fixed (no embed grad)

        return {"loss": loss, "acc": acc, "n_valid": n_valid, "margin": margin_val,
                "ppl": float(torch.exp(torch.tensor(loss)).item()) if n_valid else float("inf")}

    @torch.no_grad()
    def metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """Cross-entropy / accuracy / perplexity with no backward pass (eval)."""
        inv = 1.0 / float(self.cfg.logit_temp)
        tgt = targets.to(self.device).reshape(-1).long()
        valid = tgt != IGNORE_INDEX
        n_valid = int(valid.sum().item())
        scaled = logits.float() * inv
        logp = torch.log_softmax(scaled, dim=1)
        safe = tgt.clamp_min(0)
        nll = -logp.gather(1, safe.unsqueeze(1)).squeeze(1)[valid]
        loss = float(nll.mean().item()) if n_valid else 0.0
        acc = float((scaled.argmax(1)[valid] == tgt[valid]).float().mean().item()) if n_valid else 0.0
        return {"loss": loss, "acc": acc, "n_valid": n_valid,
                "ppl": float(torch.exp(torch.tensor(loss)).item()) if n_valid else float("inf")}

    # ── sampling / inference ─────────────────────────────────────────────────
    @torch.no_grad()
    def logits_last(self, ids: torch.Tensor) -> torch.Tensor:
        """Return the (B, V) decode similarities for the last position only."""
        B, n = ids.shape
        was_training = self.training
        self.training = False                                   # disable flip-dropout
        logits = self.forward(ids, score_extras=True).reshape(B, n, self.cfg.vocab_size)
        self.training = was_training
        return logits[:, -1, :].float()

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, n_new: int, *, temperature: float = 1.0,
                 top_k: int = 0, ban_ids: Optional[List[int]] = None,
                 repetition_window: int = 0, max_ctx: int = 256,
                 stop_ids: Optional[List[int]] = None, min_new: int = 0) -> torch.Tensor:
        """Autoregressive min-Hamming / Boltzmann decoding.

        Recomputes the (linear-time) forward pass on the growing context each
        step; the streaming BSR/episodic ring state caches exist and are
        parity-tested but are not wired into this convenience path.
        """
        ids = ids.to(self.device)
        inv = 1.0 / float(self.cfg.logit_temp)
        stop = set(int(i) for i in (stop_ids or []))
        for step in range(n_new):
            ctx = ids[:, -max_ctx:]
            sim = self.logits_last(ctx)                         # (B, V) similarities
            sim = sim * inv / max(temperature, 1e-4)
            if ban_ids:
                sim[:, ban_ids] = float("-inf")
            if repetition_window > 0:
                for b in range(ids.shape[0]):
                    recent = ids[b, -repetition_window:].tolist()
                    sim[b, recent] = float("-inf")
            if top_k and top_k > 0:
                kk = min(top_k, sim.shape[1])
                topv, topi = torch.topk(sim, kk, dim=1)
                probs = torch.softmax(topv, dim=1)
                pick = torch.multinomial(probs, 1)
                nxt = topi.gather(1, pick)
            else:
                probs = torch.softmax(sim, dim=1)
                nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)
            if stop and step + 1 >= min_new and all(int(i) in stop for i in nxt[:, 0].tolist()):
                break
        return ids

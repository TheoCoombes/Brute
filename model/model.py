"""HÆMMR — a binary-first, concept-native autoregressive language model (BEP).

Architecture (unchanged from v2 — separation of concerns by job/role)::

    token → E_lex(t)  ── input-bind ──▶  position-free concept c_i
        ┌───────────────────────────────────────────────────────────────┐ ×L
        │  r = delta-BSR(c)              # compressed discourse (§B2)      │
        │  e = EpisodicSlots(c, POS)     # exact in-window recall (§B1)    │
        │  h = HopfieldBank(c)           # learned global priors (§B3)     │
        │  m = channel-mix(c)            # binary MLP                       │
        └───────────────────────────────────────────────────────────────┘
        ── out-bind ──▶ next-token concept ĉ  (position-free)
        ── lex-proj ──▶ ℓ̂ ── decode against fixed prototype codebook ──▶ logits

Training is **BEP** (Boolean error propagation): the only large buffer per
parameter is the integer hidden weight ``H`` (Int16); the visible weight is
``W = sign(H)``.  The backward pass threads **binary desired activations** ``a*``
(bit1) — never a float signal — and the head is **margin-triggered against the
fixed prototypes** (BEP Eqs. 1-2): an update fires for a position only when
``logit[target] − max_other < r·D``, and the desired lexical activation is the
target's prototype ``E[target]``.  No softmax/CE on the backward path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

import brute

import bep
from bep import BepParam, combine_desired, mux
from layers import (
    BSR, BooleanLinear, DiagBind, EpisodicSlotMemory, HopfieldBank,
    ResidualMerge, TokenCodebook,
)
from vsa import hierarchical_position_codes, to_bit1


IGNORE_INDEX = -1


@dataclass
class HaemmrConfig:
    vocab_size: int = 2048
    D: int = 1024                       # concept hypervector dimension
    n_layers: int = 2
    d_ff: int = 2048                    # channel-mix hidden width
    # episodic slot memory (§B1) — the exact-recall fix
    epi_read_k: int = 1
    epi_slots: Optional[int] = None
    # latent Hopfield priors (§B3)
    n_slots: int = 256
    top_k: int = 15
    # delta-BSR decay palette (§B2)
    decay_shifts: tuple = (1, 2, 3, 4, 0)
    # representation / positions (§A)
    use_position: bool = True
    pos_chunk: int = 256
    # decode (§C1)
    structured_codebook: bool = True
    bef_alpha: float = 1.0
    bef_sweeps: int = 30
    sem_weight: float = 0.5
    hopfield_cleanup: bool = False
    cleanup_slots: int = 256
    cleanup_top_k: int = 7
    # BEP training (margin trigger + fixed prototypes, eligibility gate, reinforcement)
    r: float = 0.1                      # margin trigger: logit[tgt] − max_other < r·D
    boundary_nu: Optional[float] = None  # eligibility gate |z| ≤ ν·in_dim (stage-2; off)
    p_r: float = 0.0                    # CP+R reinforcement probability (BEP §3.3)
    bits: int = 15                      # integer hidden-weight H bit-width
    flip_dropout: float = 0.0
    # §F episodic address-margin objective
    margin_theta_pos: float = 0.5
    margin_theta_neg: float = 0.0
    margin_weight: float = 1.0
    # misc
    gate_open: float = 0.05
    seed: int = 0

    def __post_init__(self):
        if self.D <= 0:
            raise ValueError("D must be positive")
        if self.epi_read_k <= 0:
            raise ValueError("epi_read_k must be positive")
        if self.epi_slots is not None and self.epi_slots <= 0:
            raise ValueError("epi_slots must be positive when set")
        if self.pos_chunk <= 0:
            raise ValueError("pos_chunk must be positive")
        if self.sem_weight < 0:
            raise ValueError("sem_weight must be non-negative")
        if not 0.0 <= self.flip_dropout < 1.0:
            raise ValueError("flip_dropout must be in [0, 1)")
        if self.boundary_nu is not None and self.boundary_nu < 0:
            raise ValueError("boundary_nu must be non-negative")


# ── channel-mixing MLP (two Boolean linears) ───────────────────────────────────

class ChannelMix:
    def __init__(self, D: int, d_ff: int, *, name: str, generator=None, device=None,
                 boundary_nu=None):
        self.D, self.d_ff = D, d_ff
        self.lin1 = BooleanLinear(D, d_ff, name=f"{name}.lin1", generator=generator,
                                  device=device, boundary_nu=boundary_nu)
        self.lin2 = BooleanLinear(d_ff, D, name=f"{name}.lin2", generator=generator,
                                  device=device, boundary_nu=boundary_nu)

    def params(self) -> List[BepParam]:
        return self.lin1.params() + self.lin2.params()

    def forward(self, x_bit: brute.Tensor) -> brute.Tensor:
        h_bit, _ = self.lin1.forward(x_bit)
        m_bit, _ = self.lin2.forward(h_bit)
        return m_bit

    def backward(self, a_star: brute.Tensor) -> brute.Tensor:
        a_h = self.lin2.backward(a_star)
        return self.lin1.backward(a_h)


# ── one Boolean block ──────────────────────────────────────────────────────────

class Block:
    def __init__(self, cfg: HaemmrConfig, *, idx: int, generator=None, device=None):
        D = cfg.D
        nm = f"blk{idx}"
        nu = cfg.boundary_nu
        self.bsr = BSR(D, name=f"{nm}.bsr", decay_shifts=cfg.decay_shifts,
                       generator=generator, device=device, boundary_nu=nu)
        self.merge_bsr = ResidualMerge(D, name=f"{nm}.merge_bsr", p_open=cfg.gate_open,
                                       generator=generator, device=device)
        self.epi = EpisodicSlotMemory(D, name=f"{nm}.epi", read_k=cfg.epi_read_k,
                                      n_slots=cfg.epi_slots, generator=generator,
                                      device=device, boundary_nu=nu)
        self.merge_epi = ResidualMerge(D, name=f"{nm}.merge_epi", p_open=cfg.gate_open,
                                       value_path=True, generator=generator, device=device)
        self.hop = HopfieldBank(D, cfg.n_slots, cfg.top_k, name=f"{nm}.hop",
                                generator=generator, device=device)
        self.merge_hop = ResidualMerge(D, name=f"{nm}.merge_hop", p_open=cfg.gate_open,
                                       generator=generator, device=device)
        self.mix = ChannelMix(D, cfg.d_ff, name=f"{nm}.mix", generator=generator,
                              device=device, boundary_nu=nu)
        self.merge_mix = ResidualMerge(D, name=f"{nm}.merge_mix", p_open=cfg.gate_open,
                                       generator=generator, device=device)
        self._shape = None
        self._cache: dict = {}

    def params(self) -> List[BepParam]:
        return (self.bsr.params() + self.merge_bsr.params()
                + self.epi.params() + self.merge_epi.params()
                + self.hop.params() + self.merge_hop.params()
                + self.mix.params() + self.merge_mix.params())

    def forward(self, c_bit: brute.Tensor, pos_bit: brute.Tensor) -> brute.Tensor:
        B, n, D = c_bit.shape
        self._shape = (B, n, D)
        r = self.bsr.forward(c_bit)
        c1 = self.merge_bsr.forward(c_bit, r)
        e = self.epi.forward(c1, pos_bit)
        c2 = self.merge_epi.forward(c1, e)
        h = self.hop.forward(c2.reshape(B * n, D)).reshape(B, n, D)
        c3 = self.merge_hop.forward(c2, h)
        m = self.mix.forward(c3.reshape(B * n, D)).reshape(B, n, D)
        c4 = self.merge_mix.forward(c3, m)
        self._cache = {"c_bit": c_bit, "c1": c1, "c2": c2, "c3": c3}
        return c4

    def backward(self, a_c4: brute.Tensor) -> brute.Tensor:
        B, n, D = self._shape
        c_bit, c1, c2, c3 = (self._cache["c_bit"], self._cache["c1"],
                             self._cache["c2"], self._cache["c3"])
        a_c3_s, a_m = self.merge_mix.backward(a_c4)
        a_c3_t = self.mix.backward(a_m.reshape(B * n, D)).reshape(B, n, D)
        a_c3 = combine_desired(a_c3_s, a_c3_t, c3)
        a_c2_s, a_h = self.merge_hop.backward(a_c3)
        self.hop.backward(a_h.reshape(B * n, D))           # no upstream signal
        a_c2 = a_c2_s
        a_c1_s, a_e = self.merge_epi.backward(a_c2)
        a_c1_t = self.epi.backward(a_e)
        a_c1 = combine_desired(a_c1_s, a_c1_t, c1)
        a_c_s, a_r = self.merge_bsr.backward(a_c1)
        a_c_t = self.bsr.backward(a_r)
        return combine_desired(a_c_s, a_c_t, c_bit)


# ── the model ────────────────────────────────────────────────────────────────

class HaemmrLM:
    def __init__(self, cfg: HaemmrConfig, *, device=None,
                 codebook_init: Optional[torch.Tensor] = None):
        self.cfg = cfg
        self.training = True
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.inv = float(cfg.D) ** -0.5                    # logit scale (logging only)
        gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        nu = cfg.boundary_nu

        # ``codebook_init`` (V, D) ±1 floats overrides BEF/random init — e.g. an
        # offline GPT-2 SimHash codebook (see codebook.py).
        self.codebook = TokenCodebook(cfg.vocab_size, cfg.D, name="E",
                                      structured=cfg.structured_codebook,
                                      bef_alpha=cfg.bef_alpha, bef_sweeps=cfg.bef_sweeps,
                                      init_pm1=codebook_init,
                                      generator=gen, device=self.device)
        self.input_bind = DiagBind(cfg.D, name="in", generator=gen, device=self.device)
        self.blocks = [Block(cfg, idx=i, generator=gen, device=self.device)
                       for i in range(cfg.n_layers)]
        self.out_bind = DiagBind(cfg.D, name="out", generator=gen, device=self.device)
        self.lex_proj = BooleanLinear(cfg.D, cfg.D, name="lex", generator=gen,
                                      device=self.device, boundary_nu=nu)
        self.sem_bind = DiagBind(cfg.D, name="sem", generator=gen, device=self.device)
        if cfg.hopfield_cleanup:
            self.cleanup_hop = HopfieldBank(cfg.D, cfg.cleanup_slots, cfg.cleanup_top_k,
                                            name="clean", generator=gen, device=self.device)
            self.cleanup_merge = ResidualMerge(cfg.D, name="clean.merge", p_open=cfg.gate_open,
                                               generator=gen, device=self.device)

        self.pos_chunk_base = (torch.randint(0, 2, (cfg.D,), generator=gen).float() * 2 - 1).to(self.device)
        self.pos_offset_base = (torch.randint(0, 2, (cfg.D,), generator=gen).float() * 2 - 1).to(self.device)
        self._pos_cache: dict = {}
        self._neutral_pos_cache: dict = {}
        self._fwd_cache: dict = {}

    # ── parameters / checkpoint ──────────────────────────────────────────────
    def parameters(self) -> List[BepParam]:
        ps = self.codebook.params() + self.input_bind.params()
        for b in self.blocks:
            ps += b.params()
        ps += self.out_bind.params() + self.lex_proj.params() + self.sem_bind.params()
        if self.cfg.hopfield_cleanup:
            ps += self.cleanup_hop.params() + self.cleanup_merge.params()
        return ps

    def num_bit_parameters(self) -> int:
        return sum(int(torch.tensor(p.shape).prod().item()) for p in self.parameters())

    def param_bytes(self) -> int:
        """Total bytes held by trainable parameters (the integer ``H`` buffers)."""
        return sum(p.param_bytes() for p in self.parameters())

    def state_dict(self) -> dict:
        return {
            "cfg": self.cfg.__dict__,
            "pos_chunk_base": self.pos_chunk_base.detach().cpu().clone(),
            "pos_offset_base": self.pos_offset_base.detach().cpu().clone(),
            "params": {p.name: p.state_dict() for p in self.parameters()},
        }

    def load_state_dict(self, sd: dict) -> None:
        self.pos_chunk_base = sd["pos_chunk_base"].to(self.device)
        self.pos_offset_base = sd["pos_offset_base"].to(self.device)
        byname = {p.name: p for p in self.parameters()}
        for name, psd in sd["params"].items():
            if name in byname:
                byname[name].load_state_dict(psd)
        self._pos_cache.clear()
        self._neutral_pos_cache.clear()

    # ── positions (address lane only — never bound into the decoded concept) ───
    def _positions(self, n: int):
        if self._pos_cache.get("n", -1) < n:
            codes = hierarchical_position_codes(self.pos_chunk_base, self.pos_offset_base,
                                                n, chunk=self.cfg.pos_chunk)
            self._pos_cache = {"n": n, "bit": codes}
        return self._pos_cache["bit"][:n]

    def _position_lane(self, n: int):
        if self.cfg.use_position:
            return self._positions(n)
        if self._neutral_pos_cache.get("n", -1) < n:
            ones = torch.ones(n, self.cfg.D, dtype=torch.bool, device=self.device)
            self._neutral_pos_cache = {
                "n": n,
                "bit": brute.as_tensor(ones, dtype=brute.bit1).to(self.device),
            }
        return self._neutral_pos_cache["bit"][:n]

    # ── flip-dropout (train against the deployment bit-flip noise model, §D) ──
    def _flip_noise(self, c_bit: brute.Tensor):
        rate = self.cfg.flip_dropout
        if not self.training or rate <= 0:
            return c_bit, None
        flip = torch.rand(c_bit.shape, device=self.device) < rate
        flip_bit = brute.as_tensor(flip, dtype=brute.bit1).to(self.device)
        noised = brute.fast.bitwise_xor(c_bit, flip_bit)
        return noised, flip_bit

    # ── forward ────────────────────────────────────────────────────────────
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.to(self.device)
        B, n = ids.shape
        D = self.cfg.D
        pos_bit = self._position_lane(n)

        emb = self.codebook.embed(ids)
        c = self.input_bind.forward(emb)
        flip_masks = []
        for blk in self.blocks:
            c, fbit = self._flip_noise(c)
            flip_masks.append(fbit)
            c = blk.forward(c, pos_bit)
        chat = self.out_bind.forward(c)

        cleanup_cache = None
        if self.cfg.hopfield_cleanup:
            hc = self.cleanup_hop.forward(chat.reshape(B * n, D)).reshape(B, n, D)
            chat = self.cleanup_merge.forward(chat, hc)
            cleanup_cache = True

        chat_flat = chat.reshape(B * n, D)
        ell_bit, _ = self.lex_proj.forward(chat_flat)
        lex_logits = self.codebook.decode(ell_bit)

        sem_bit = None
        if self.cfg.sem_weight > 0:
            sem_bit = self.sem_bind.forward(chat_flat)
            sem_logits = self.codebook.decode(sem_bit)
            logits = lex_logits.float() + self.cfg.sem_weight * sem_logits.float()
        else:
            logits = lex_logits.float()

        self._fwd_cache = {
            "B": B, "n": n, "ell_bit": ell_bit, "sem_bit": sem_bit,
            "chat_flat": chat_flat, "flip_masks": flip_masks, "cleanup": cleanup_cache,
        }
        return logits

    # ── loss + BEP backward (margin trigger + fixed prototypes) ───────────────
    def loss_and_backward(self, logits: torch.Tensor, targets: torch.Tensor,
                          matched: Optional[torch.Tensor] = None) -> dict:
        cfg = self.cfg
        B, n = self._fwd_cache["B"], self._fwd_cache["n"]
        D, V = cfg.D, cfg.vocab_size
        M = B * n
        tgt = targets.to(self.device).reshape(-1).long()
        valid = tgt != IGNORE_INDEX
        n_valid = int(valid.sum().item())
        safe_tgt = tgt.clamp_min(0)

        # logging only (no role in the backward path)
        scaled = logits.float() * self.inv
        logp = torch.log_softmax(scaled, dim=1)
        nll = -logp.gather(1, safe_tgt.unsqueeze(1)).squeeze(1)[valid]
        loss = float(nll.mean().item()) if n_valid else 0.0
        pred = scaled.argmax(dim=1)
        acc = float((pred[valid] == tgt[valid]).float().mean().item()) if n_valid else 0.0

        # BEP margin trigger (Eqs. 1-2): fire where logit[tgt] − max_other < r·D
        logit_tgt = logits.gather(1, safe_tgt.unsqueeze(1)).squeeze(1)
        other = logits.clone()
        other.scatter_(1, safe_tgt.unsqueeze(1), float("-inf"))
        max_other = other.max(dim=1).values
        trigger = valid & ((logit_tgt - max_other) < cfg.r * D)
        trig_rate = float(trigger.float().mean().item()) if M else 0.0

        # desired lexical activation = target prototype where triggered, else current
        proto = self.codebook.prototype(safe_tgt)          # (M, D) bit1
        sel = to_bit1(trigger.unsqueeze(1).expand(M, D))
        ell_bit = self._fwd_cache["ell_bit"]
        ell_des = mux(sel, proto, ell_bit)
        chat_flat = self._fwd_cache["chat_flat"]
        # only triggered positions inject a backward signal (BEP); the rest carry
        # no opinion (desired = current activation, no weight update).
        bep.set_active(trigger)
        g_chat = self.lex_proj.backward(ell_des)

        sem_bit = self._fwd_cache["sem_bit"]
        if sem_bit is not None:
            sem_des = mux(sel, proto, sem_bit)
            g_chat_sem = self.sem_bind.backward(sem_des)
            g_chat = combine_desired(g_chat, g_chat_sem, chat_flat)
        g_chat = g_chat.reshape(B, n, D)

        if self._fwd_cache["cleanup"]:
            g_chat_skip, g_hc = self.cleanup_merge.backward(g_chat)
            self.cleanup_hop.backward(g_hc.reshape(B * n, D))
            g_chat = g_chat_skip

        g_c = self.out_bind.backward(g_chat)

        margin_val = 0.0
        if matched is not None and cfg.margin_weight > 0:
            for blk in self.blocks:
                margin_val += blk.epi.margin_loss(
                    matched.to(self.device), theta_pos=cfg.margin_theta_pos,
                    theta_neg=cfg.margin_theta_neg, weight=cfg.margin_weight)

        flip_masks = self._fwd_cache["flip_masks"]
        for blk, fbit in zip(reversed(self.blocks), reversed(flip_masks)):
            g_c = blk.backward(g_c)
            if fbit is not None:
                g_c = brute.fast.bitwise_xor(g_c, fbit)    # un-flip the dropout noise
        g_emb = self.input_bind.backward(g_c)
        self.codebook.backward_embed(g_emb)
        bep.set_active(None)

        return {"loss": loss, "acc": acc, "n_valid": n_valid, "margin": margin_val,
                "trigger_rate": trig_rate,
                "ppl": float(torch.exp(torch.tensor(loss)).item()) if n_valid else float("inf")}

    @torch.no_grad()
    def metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        tgt = targets.to(self.device).reshape(-1).long()
        valid = tgt != IGNORE_INDEX
        n_valid = int(valid.sum().item())
        scaled = logits.float() * self.inv
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
        B, n = ids.shape
        was_training = self.training
        self.training = False
        logits = self.forward(ids).reshape(B, n, self.cfg.vocab_size)
        self.training = was_training
        return logits[:, -1, :].float()

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, n_new: int, *, temperature: float = 1.0,
                 top_k: int = 0, ban_ids: Optional[List[int]] = None,
                 repetition_window: int = 0, max_ctx: int = 256) -> torch.Tensor:
        ids = ids.to(self.device)
        for _ in range(n_new):
            ctx = ids[:, -max_ctx:]
            sim = self.logits_last(ctx) * self.inv / max(temperature, 1e-4)
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
                nxt = topi.gather(1, torch.multinomial(probs, 1))
            else:
                probs = torch.softmax(sim, dim=1)
                nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids

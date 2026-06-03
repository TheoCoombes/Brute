"""HÆMMR — a binary-first, concept-native autoregressive language model.

The architecture separates memory mechanisms by job and representation
subspaces by role:

    token → E_lex(t)  ── input-bind ──▶  position-free concept c_i
        ┌───────────────────────────────────────────────────────────────┐ ×L
        │  r = delta-BSR(c)              # compressed discourse             │
        │  e = EpisodicSlots(c, POS)     # exact in-window recall           │
        │  h = HopfieldBank(c)           # learned global priors            │
        │  m = channel-mix(c)            # binary MLP                       │
        └───────────────────────────────────────────────────────────────┘
        ── out-bind ──▶ next-token concept ĉ  (position-free)
        ── lex-proj ──▶ ℓ̂ ── decode against BEF codebook ──▶ logits
                              (+ optional semantic rerank)

Position never enters the decoded concept. It lives only in the address lane
(BSR/episodic keys), so decoding does not need positional unbinding. Episodic
slots provide exact in-window lookup, and the head emits a concept that is
projected to a lexical frame before Hamming decode against the token codebook.

Forward is bitwise (packed ``brute.bit1``); the only integers are signed matmul
pre-activations and the BSR accumulator. Training is BOLD bit-flips with
boundary gating and discrete-noise regularisation options.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

import brute

from bold import BoldParam
from layers import (
    BSR, BooleanLinear, DiagBind, EpisodicSlotMemory, HopfieldBank,
    ResidualMerge, TokenCodebook, _LazyPM1Cache,
)
from vsa import hierarchical_position_codes, to_pm1


IGNORE_INDEX = -1


@dataclass
class HaemmrConfig:
    vocab_size: int = 2048
    D: int = 1024                       # concept hypervector dimension
    n_layers: int = 2
    d_ff: int = 2048                    # channel-mix hidden width
    # episodic slot memory
    epi_read_k: int = 1                 # top-k read width (1 ⇒ exact single-slot)
    epi_slots: Optional[int] = None     # streaming Tier-1 window cap; None ⇒ 2*epi_chunk
    epi_chunk: int = 64                 # chunk size C for the two-tier chunked forward
    epi_registers: int = 64             # Tier-2 register slots S (0 = disabled)
    # latent Hopfield priors
    n_slots: int = 256                  # learned static slots (M)
    top_k: int = 15                     # WTA width (odd → no ties)
    # delta-BSR decay palette
    use_bsr: bool = True                # disable for packed-only hardware profiles
    decay_shifts: tuple = (1, 2, 3, 4, 0)   # power-of-two shifts; 0 = permanent
    # representation / positions
    use_position: bool = True           # positional addressing in the episodic lane
    pos_chunk: int = 256                # hierarchical position chunk size C
    # decode
    structured_codebook: bool = True    # BEF codebook (else random)
    bef_alpha: float = 1.0
    bef_sweeps: int = 30
    sem_weight: float = 0.5             # semantic-rerank logit weight (0 disables)
    hopfield_cleanup: bool = False      # one-step concept cleanup before decode
    cleanup_slots: int = 256
    cleanup_top_k: int = 7
    # training mitigations and margin objective
    boundary_nu: Optional[float] = None  # BEP boundary gating |z|≤νD (e.g. 0.5); None=off
    label_smoothing: float = 0.0
    flip_dropout: float = 0.0           # activation bit-flip rate in the training forward
    margin_theta_pos: float = 0.5       # matched-pair margin (fraction of D)
    margin_theta_neg: float = 0.0       # distractor margin
    margin_weight: float = 1.0
    # misc
    logit_temp: Optional[float] = None  # train-time logit scale; default sqrt(D)
    gate_open: float = 0.05             # residual-gate init openness
    codebook_flip_scale: float = 0.3    # codebook flip-rate vs transforms
    seed: int = 0

    def __post_init__(self):
        if self.logit_temp is None:
            self.logit_temp = float(self.D) ** 0.5
        if self.D <= 0:
            raise ValueError("D must be positive")
        if self.epi_read_k <= 0:
            raise ValueError("epi_read_k must be positive")
        if self.epi_slots is not None and self.epi_slots <= 0:
            raise ValueError("epi_slots must be positive when set")
        if self.epi_chunk <= 0:
            raise ValueError("epi_chunk must be positive")
        if self.epi_registers < 0:
            raise ValueError("epi_registers must be non-negative")
        if self.pos_chunk <= 0:
            raise ValueError("pos_chunk must be positive")
        if self.sem_weight < 0:
            raise ValueError("sem_weight must be non-negative")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
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


# ── one Boolean block ──────────────────────────────────────────────────────────

class Block:
    def __init__(self, cfg: HaemmrConfig, *, idx: int, generator=None, device=None):
        D = cfg.D
        nm = f"blk{idx}"
        nu = cfg.boundary_nu
        self.use_bsr = cfg.use_bsr
        if self.use_bsr:
            self.bsr = BSR(D, name=f"{nm}.bsr", decay_shifts=cfg.decay_shifts,
                           generator=generator, device=device, boundary_nu=nu)
            self.merge_bsr = ResidualMerge(D, name=f"{nm}.merge_bsr", p_open=cfg.gate_open,
                                           generator=generator, device=device)
        else:
            self.bsr = None
            self.merge_bsr = None
        self.epi = EpisodicSlotMemory(D, name=f"{nm}.epi", read_k=cfg.epi_read_k,
                                      n_slots=cfg.epi_slots,
                                      epi_chunk=cfg.epi_chunk,
                                      epi_registers=cfg.epi_registers,
                                      generator=generator,
                                      device=device, boundary_nu=nu)
        self.merge_epi = ResidualMerge(D, name=f"{nm}.merge_epi", p_open=cfg.gate_open,
                                       generator=generator, device=device)
        self.hop = HopfieldBank(D, cfg.n_slots, cfg.top_k, name=f"{nm}.hop",
                                generator=generator, device=device)
        self.merge_hop = ResidualMerge(D, name=f"{nm}.merge_hop", p_open=cfg.gate_open,
                                       generator=generator, device=device)
        self.mix = ChannelMix(D, cfg.d_ff, name=f"{nm}.mix", generator=generator,
                              device=device, boundary_nu=nu)
        self.merge_mix = ResidualMerge(D, name=f"{nm}.merge_mix", p_open=cfg.gate_open,
                                       generator=generator, device=device)
        self._shape = None

    def params(self) -> List[BoldParam]:
        ps = []
        if self.use_bsr:
            ps += self.bsr.params() + self.merge_bsr.params()
        return (ps + self.epi.params() + self.merge_epi.params()
                + self.hop.params() + self.merge_hop.params()
                + self.mix.params() + self.merge_mix.params())

    def forward(self, c_bit: brute.Tensor, pos_bit: brute.Tensor) -> brute.Tensor:
        B, n, D = c_bit.shape
        self._shape = (B, n, D)
        if self.use_bsr:
            r = self.bsr.forward(c_bit)                        # discourse (B,n,D)
            c1 = self.merge_bsr.forward(c_bit, r)
        else:
            c1 = c_bit
        e = self.epi.forward(c1, pos_bit)                     # exact recall (B,n,D)
        c2 = self.merge_epi.forward(c1, e)
        h = self.hop.forward(c2.reshape(B * n, D)).reshape(B, n, D)
        c3 = self.merge_hop.forward(c2, h)
        m = self.mix.forward(c3.reshape(B * n, D)).reshape(B, n, D)
        c4 = self.merge_mix.forward(c3, m)
        return c4

    def backward(self, S_c4: torch.Tensor) -> torch.Tensor:
        B, n, D = self._shape
        S_c3_a, S_m = self.merge_mix.backward(S_c4)
        S_c3_b = self.mix.backward(S_m.reshape(B * n, D)).reshape(B, n, D)
        S_c3 = S_c3_a + S_c3_b
        S_c2_a, S_h = self.merge_hop.backward(S_c3)
        S_c2_b = self.hop.backward(S_h.reshape(B * n, D)).reshape(B, n, D)
        S_c2 = S_c2_a + S_c2_b
        S_c1_a, S_e = self.merge_epi.backward(S_c2)
        S_c1_b = self.epi.backward(S_e)
        S_c1 = S_c1_a + S_c1_b
        if self.use_bsr:
            S_c_a, S_r = self.merge_bsr.backward(S_c1)
            S_c_b = self.bsr.backward(S_r)
            return S_c_a + S_c_b
        return S_c1


# ── the model ────────────────────────────────────────────────────────────────

class HaemmrLM:
    def __init__(self, cfg: HaemmrConfig, *, device=None):
        self.cfg = cfg
        self.training = True
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        nu = cfg.boundary_nu

        self.codebook = TokenCodebook(cfg.vocab_size, cfg.D, name="E",
                                      flip_scale=cfg.codebook_flip_scale,
                                      structured=cfg.structured_codebook,
                                      bef_alpha=cfg.bef_alpha, bef_sweeps=cfg.bef_sweeps,
                                      generator=gen, device=self.device)
        self.input_bind = DiagBind(cfg.D, name="in", generator=gen, device=self.device)
        self.blocks = [Block(cfg, idx=i, generator=gen, device=self.device)
                       for i in range(cfg.n_layers)]
        self.out_bind = DiagBind(cfg.D, name="out", generator=gen, device=self.device)
        # concept -> lexical frame projection
        self.lex_proj = BooleanLinear(cfg.D, cfg.D, name="lex", generator=gen,
                                      device=self.device, boundary_nu=nu)
        # semantic view of the concept for rerank
        self.sem_bind = DiagBind(cfg.D, name="sem", generator=gen, device=self.device)
        # optional concept cleanup bank
        if cfg.hopfield_cleanup:
            self.cleanup_hop = HopfieldBank(cfg.D, cfg.cleanup_slots, cfg.cleanup_top_k,
                                            name="clean", generator=gen, device=self.device)
            self.cleanup_merge = ResidualMerge(cfg.D, name="clean.merge", p_open=cfg.gate_open,
                                               generator=gen, device=self.device)

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
        ps += self.out_bind.params() + self.lex_proj.params() + self.sem_bind.params()
        if self.cfg.hopfield_cleanup:
            ps += self.cleanup_hop.params() + self.cleanup_merge.params()
        return ps

    def num_bit_parameters(self) -> int:
        return sum(int(torch.tensor(p.shape).prod().item()) for p in self.parameters())

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
        into a neutral all-ones role rather than disabling the slot path.  The
        memory remains content-addressed and receives a valid packed tensor.
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
        # XOR a bit means *flip* it; bind with ¬flip would xnor — use xor on packed
        noised = brute.fast.bitwise_xor(c_bit, flip_bit)
        mask_pm1 = torch.where(flip, -1.0, 1.0)                            # un-flip factor
        return noised, mask_pm1

    # ── forward ────────────────────────────────────────────────────────────
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """``ids`` (B, n) long → logits (B*n, V) signed similarities."""
        ids = ids.to(self.device)
        B, n = ids.shape
        D = self.cfg.D
        pos_bit = self._position_lane(n)

        emb = self.codebook.embed(ids)                          # (B,n,D) position-free
        c = self.input_bind.forward(emb)                        # learned concept dressing
        flip_masks = []
        for blk in self.blocks:
            c, mpm1 = self._flip_noise(c)
            flip_masks.append(mpm1)
            c = blk.forward(c, pos_bit)
        chat = self.out_bind.forward(c)                         # next-token concept ĉ

        cleanup_cache = None
        if self.cfg.hopfield_cleanup:
            hc = self.cleanup_hop.forward(chat.reshape(B * n, D)).reshape(B, n, D)
            chat_clean = self.cleanup_merge.forward(chat, hc)
            cleanup_cache = True
            chat = chat_clean

        chat_flat = chat.reshape(B * n, D)
        ell_bit, _ = self.lex_proj.forward(chat_flat)           # ℓ̂ lexical frame
        lex_logits = self.codebook.decode(ell_bit)              # ⟨ℓ̂, E⟩

        sem_bit = None
        if self.cfg.sem_weight > 0:
            sem_bit = self.sem_bind.forward(chat_flat)          # ĉ ⊗ W_sem
            sem_logits = self.codebook.decode(sem_bit)
            logits = lex_logits.float() + self.cfg.sem_weight * sem_logits.float()
        else:
            logits = lex_logits.float()

        cache_data = {
            "B": B, "n": n, "ell_bit": ell_bit, "sem_bit": sem_bit,
            "flip_masks": flip_masks, "cleanup": cleanup_cache,
        }
        pm1_sources = {"ell_pm1": ("ell_bit", tuple(ell_bit.shape))}
        if sem_bit is None:
            cache_data["sem_pm1"] = None
        else:
            pm1_sources["sem_pm1"] = ("sem_bit", tuple(sem_bit.shape))
        self._fwd_cache = _LazyPM1Cache(cache_data, pm1_sources=pm1_sources)
        return logits

    # ── loss + BOLD backward ─────────────────────────────────────────────────
    def loss_and_backward(self, logits: torch.Tensor, targets: torch.Tensor,
                          matched: Optional[torch.Tensor] = None) -> dict:
        """CE loss + BOLD backward.  ``matched`` (B,n) optionally supplies
        episodic margin supervision (slot id each query should retrieve, -1 = none)."""
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
        p = logp.exp()
        safe_tgt = tgt.clamp_min(0)
        nll = -logp.gather(1, safe_tgt.unsqueeze(1)).squeeze(1)[valid]
        loss = float(nll.mean().item()) if n_valid else 0.0
        pred = scaled.argmax(dim=1)
        acc = float((pred[valid] == tgt[valid]).float().mean().item()) if n_valid else 0.0

        # target distribution with optional label smoothing
        eps = cfg.label_smoothing
        onehot = torch.zeros_like(p)
        onehot.scatter_(1, safe_tgt.unsqueeze(1), 1.0)
        if eps > 0:
            target = (1 - eps) * onehot + eps / V
        else:
            target = onehot
        S_logits = (p - target) * (inv / max(n_valid, 1))
        S_logits[~valid] = 0.0

        # decode backward (lexical + optional semantic), both into the concept ĉ
        ell_pm1 = self._fwd_cache["ell_pm1"]
        g_ell = self.codebook.decode_backward(S_logits, ell_pm1)
        g_chat = self.lex_proj.backward(g_ell)                  # (B*n, D)
        sem_pm1 = self._fwd_cache["sem_pm1"]
        if sem_pm1 is not None:
            g_sem = self.codebook.decode_backward(S_logits * cfg.sem_weight, sem_pm1)
            g_chat = g_chat + self.sem_bind.backward(g_sem)
        g_chat = g_chat.reshape(B, n, D)

        if self._fwd_cache["cleanup"]:
            g_chat_skip, g_hc = self.cleanup_merge.backward(g_chat)
            self.cleanup_hop.backward(g_hc.reshape(B * n, D))   # priors: zero query grad
            g_chat = g_chat_skip

        g_c = self.out_bind.backward(g_chat)                    # (B,n,D)

        # margin objective on every block's episodic address lane (uses fwd cache)
        margin_val = 0.0
        if matched is not None and cfg.margin_weight > 0:
            for blk in self.blocks:
                margin_val += blk.epi.margin_loss(
                    matched.to(self.device), theta_pos=cfg.margin_theta_pos,
                    theta_neg=cfg.margin_theta_neg, weight=cfg.margin_weight)

        flip_masks = self._fwd_cache["flip_masks"]
        for blk, mpm1 in zip(reversed(self.blocks), reversed(flip_masks)):
            g_c = blk.backward(g_c)
            if mpm1 is not None:                                # un-flip the dropout noise
                g_c = g_c * mpm1
        g_emb = self.input_bind.backward(g_c)                   # (B,n,D)
        self.codebook.backward_embed(g_emb)

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
        logits = self.forward(ids).reshape(B, n, self.cfg.vocab_size)
        self.training = was_training
        return logits[:, -1, :].float()

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, n_new: int, *, temperature: float = 1.0,
                 top_k: int = 0, ban_ids: Optional[List[int]] = None,
                 repetition_window: int = 0, max_ctx: int = 256,
                 stop_ids: Optional[List[int]] = None, min_new: int = 0) -> torch.Tensor:
        """Autoregressive min-Hamming / Boltzmann decoding.

        Recomputes the (linear-time) forward pass on the growing context each
        step; the streaming BSR/episodic state caches exist and are parity-tested
        but are not wired into this convenience path.
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

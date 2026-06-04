"""A native **binary decoder transformer** — token↔token attention, BEP-trained.

This is a 1:1 binary-native equivalent of a standard decoder transformer, built
entirely on packed ``brute.bit1`` (uint64) tensors::

    token → E(t)  ── input-bind ──▶  position-free concept  x_i ∈ 𝔹^D
        ┌──────────────────────────────────────────────────────────────┐ ×L
        │  a = BinaryMultiHeadAttention(x)     # token↔token (ALiBi, causal) │
        │  x = maj3(x, a, c_A)                 # majority residual          │
        │  f = BinaryGLU(x)                    # exact XNOR-gated FFN        │
        │  x = maj3(x, f, c_F)                 # majority residual          │
        └──────────────────────────────────────────────────────────────┘
        ── out-bind ──▶ ĉ ── lex-proj ──▶ ℓ̂ ── tied-codebook Hamming ──▶ logits

Sequence mixing is one native :class:`~attention.BinaryMultiHeadAttention` (no
recurrence, no associative bank, no slot memory): queries, keys and values are
1-bit projections per head, scores are XNOR-popcount signed dot products on an
integer register, position is a relative integer ALiBi bias, and the value
combine is a packed gather (hardmax) or an int8 vote bundle (soft).

Training is **BEP** (Boolean error propagation): the only large buffer per
parameter is the integer hidden weight ``H`` (Int16); the visible weight is
``W = sign(H)``.  The backward pass threads **binary desired activations** ``a*``
(bit1) — never a float signal — and the head is **contrastive margin-triggered**
(an update fires for a position only when ``logit[target] − max_other < r·D``).
The attention score lane (``W_Q, W_K``) is trained by a per-head Hamming-margin
objective supervised by the self-supervised induction signal.  No softmax/CE on
the backward path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional

import torch

import brute

import bep
from bep import BepParam, combine_desired, mux
from attention import BinaryMultiHeadAttention
from layers import BinaryGLU, BooleanLinear, DiagBind, MajorityResidual, TokenCodebook


IGNORE_INDEX = -1


@dataclass
class TransformerConfig:
    vocab_size: int = 50257              # GPT-2 tokenizer size; training overrides from data
    D: int = 512                        # concept hypervector / model dimension
    n_layers: int = 2
    n_heads: int = 4                    # D // n_heads must be a multiple of 64
    d_ff: int = 1024                    # GLU hidden width
    # attention
    attn_mode: str = "soft"             # 'soft' (int8 vote bundle) | 'hardmax' (gather)
    attn_band: int = 1                  # soft attention integer margin band (≥1)
    alibi: bool = True                  # relative integer ALiBi score bias
    causal: bool = True                 # causal mask (j ≤ i)
    causal_strict: bool = False         # exclude self (j < i): recency selects i-1
    value_proj: bool = True             # True: learned W_V/W_O; False: raw-concept copy
    alibi_slopes_override: Optional[tuple] = None  # explicit per-head integer slopes
    # readout / codebook
    structured_codebook: bool = True
    bef_alpha: float = 1.0
    bef_sweeps: int = 30
    sem_weight: float = 0.5             # semantic rerank lane weight (0 disables)
    # BEP training (margin trigger + fixed prototypes, reinforcement)
    r: float = 0.1                      # margin trigger: logit[tgt] − max_other < r·D
    train_output_codebook: bool = True
    margin_r_final: Optional[float] = None
    margin_anneal_steps: int = 0
    readout_warmup_steps: int = 0
    max_trigger_rate: Optional[float] = None
    boundary_nu: Optional[float] = None
    p_r: float = 0.0
    bits: int = 15
    init_inertia: int = 1
    update_clip: Optional[int] = None
    block_init_inertia: Optional[int] = None
    block_update_clip: Optional[int] = None
    flip_dropout: float = 0.0
    # attention score-lane margin objective
    margin_theta_pos: float = 0.5
    margin_theta_neg: float = 0.0
    margin_weight: float = 1.0
    self_supervised_induction: bool = True
    # binary residual: 'mux' (branch replaces skip — copy/retrieval) | 'majority'
    # (maj3(skip, branch, c) — BOLD-faithful, aggregation)
    residual_mode: str = "mux"
    gate_open: float = 0.05             # residual gate initial openness (→ identity init)
    residual_c_p_true: float = 0.5
    seed: int = 0

    def __post_init__(self):
        if self.D <= 0:
            raise ValueError("D must be positive")
        if self.n_heads <= 0 or self.D % self.n_heads != 0:
            raise ValueError("n_heads must divide D")
        if (self.D // self.n_heads) % 64 != 0:
            raise ValueError("D // n_heads must be a multiple of 64 (packed head dim)")
        if self.attn_mode not in ("soft", "hardmax"):
            raise ValueError("attn_mode must be 'soft' or 'hardmax'")
        if self.residual_mode not in ("mux", "majority"):
            raise ValueError("residual_mode must be 'mux' or 'majority'")
        if self.attn_band < 1:
            raise ValueError("attn_band must be >= 1")
        if not 0.0 <= self.gate_open <= 1.0:
            raise ValueError("gate_open must be in [0, 1]")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if self.sem_weight < 0:
            raise ValueError("sem_weight must be non-negative")
        if not 0.0 <= self.flip_dropout < 1.0:
            raise ValueError("flip_dropout must be in [0, 1)")
        if self.boundary_nu is not None and self.boundary_nu < 0:
            raise ValueError("boundary_nu must be non-negative")
        if self.init_inertia <= 0:
            raise ValueError("init_inertia must be positive")
        if self.block_init_inertia is not None and self.block_init_inertia <= 0:
            raise ValueError("block_init_inertia must be positive when set")
        if self.update_clip is not None and self.update_clip <= 0:
            raise ValueError("update_clip must be positive when set")
        if self.block_update_clip is not None and self.block_update_clip <= 0:
            raise ValueError("block_update_clip must be positive when set")
        if self.margin_r_final is not None and self.margin_r_final < 0:
            raise ValueError("margin_r_final must be non-negative when set")
        if self.margin_anneal_steps < 0:
            raise ValueError("margin_anneal_steps must be non-negative")
        if self.readout_warmup_steps < 0:
            raise ValueError("readout_warmup_steps must be non-negative")
        if self.max_trigger_rate is not None and not (0 < self.max_trigger_rate <= 1):
            raise ValueError("max_trigger_rate must be in (0, 1] when set")


# ── one binary transformer block: MHA → maj-residual → GLU → maj-residual ───────

class Block:
    def __init__(self, cfg: TransformerConfig, *, idx: int, generator=None, device=None):
        D = cfg.D
        self.cfg = cfg
        nm = f"blk{idx}"
        nu = cfg.boundary_nu
        init_inertia = cfg.block_init_inertia or cfg.init_inertia
        update_clip = cfg.block_update_clip if cfg.block_update_clip is not None else cfg.update_clip
        self.mha = BinaryMultiHeadAttention(
            D, cfg.n_heads, name=f"{nm}.mha", attn_mode=cfg.attn_mode, alibi=cfg.alibi,
            causal=cfg.causal, causal_strict=cfg.causal_strict, attn_band=cfg.attn_band,
            value_proj=cfg.value_proj, alibi_slopes_override=cfg.alibi_slopes_override,
            generator=generator, device=device,
            boundary_nu=nu, init_inertia=init_inertia, update_clip=update_clip)
        self.res_a = MajorityResidual(D, name=f"{nm}.res_a", mode=cfg.residual_mode,
                                      p_open=cfg.gate_open, c_p_true=cfg.residual_c_p_true,
                                      value_path=True, generator=generator, device=device,
                                      init_inertia=init_inertia, update_clip=update_clip)
        self.glu = BinaryGLU(D, cfg.d_ff, name=f"{nm}.glu", generator=generator, device=device,
                             boundary_nu=nu, init_inertia=init_inertia, update_clip=update_clip)
        self.res_f = MajorityResidual(D, name=f"{nm}.res_f", mode=cfg.residual_mode,
                                      p_open=cfg.gate_open, c_p_true=cfg.residual_c_p_true,
                                      value_path=False, generator=generator, device=device,
                                      init_inertia=init_inertia, update_clip=update_clip)
        self._cache: dict = {}

    def params(self) -> List[BepParam]:
        return (self.mha.params() + self.res_a.params()
                + self.glu.params() + self.res_f.params())

    def forward(self, x_bit: brute.Tensor) -> brute.Tensor:
        B, n, D = x_bit.shape
        a = self.mha.forward(x_bit)                       # attention sublayer
        x2 = self.res_a.forward(x_bit, a)                 # majority residual
        f = self.glu.forward(x2.reshape(B * n, D)).reshape(B, n, D)   # FFN sublayer
        x4 = self.res_f.forward(x2, f)                    # majority residual
        self._cache = {"shape": (B, n, D), "x_bit": x_bit, "x2": x2}
        return x4

    def backward(self, x4_star: brute.Tensor) -> brute.Tensor:
        B, n, D = self._cache["shape"]
        x_bit, x2 = self._cache["x_bit"], self._cache["x2"]
        x2_skip, f_star = self.res_f.backward(x4_star)
        x2_trans = self.glu.backward(f_star.reshape(B * n, D)).reshape(B, n, D)
        x2_star = combine_desired(x2_skip, x2_trans, x2)
        x_skip, a_star = self.res_a.backward(x2_star)
        x_trans = self.mha.backward(a_star)
        return combine_desired(x_skip, x_trans, x_bit)

    def margin_loss(self, matched: torch.Tensor, *, theta_pos: float, theta_neg: float,
                    active_rows: Optional[torch.Tensor]) -> float:
        return self.mha.margin_loss(matched, theta_pos=theta_pos, theta_neg=theta_neg,
                                    active_rows=active_rows)


# ── the model ────────────────────────────────────────────────────────────────

class BinaryTransformerLM:
    def __init__(self, cfg: TransformerConfig, *, device=None):
        self.cfg = cfg
        self.training = True
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.inv = float(cfg.D) ** -0.5                    # logit scale (logging only)
        gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        nu = cfg.boundary_nu

        self.codebook = TokenCodebook(cfg.vocab_size, cfg.D, name="E",
                                      structured=cfg.structured_codebook,
                                      bef_alpha=cfg.bef_alpha, bef_sweeps=cfg.bef_sweeps,
                                      generator=gen, device=self.device, seed=cfg.seed)
        self.input_bind = DiagBind(cfg.D, name="in", generator=gen, device=self.device,
                                   init_inertia=cfg.init_inertia, update_clip=cfg.update_clip)
        self.blocks = [Block(cfg, idx=i, generator=gen, device=self.device)
                       for i in range(cfg.n_layers)]
        self.out_bind = DiagBind(cfg.D, name="out", generator=gen, device=self.device,
                                 init_inertia=cfg.init_inertia, update_clip=cfg.update_clip)
        self.lex_proj = BooleanLinear(cfg.D, cfg.D, name="lex", generator=gen,
                                      device=self.device, boundary_nu=nu,
                                      init_inertia=cfg.init_inertia, update_clip=cfg.update_clip)
        self.sem_bind = DiagBind(cfg.D, name="sem", generator=gen, device=self.device,
                                 init_inertia=cfg.init_inertia, update_clip=cfg.update_clip)
        self._fwd_cache: dict = {}
        self._train_step = 0

    # ── parameters / checkpoint ──────────────────────────────────────────────
    def parameters(self) -> List[BepParam]:
        ps = self.codebook.params() + self.input_bind.params()
        for b in self.blocks:
            ps += b.params()
        ps += self.out_bind.params() + self.lex_proj.params() + self.sem_bind.params()
        return ps

    def num_bit_parameters(self) -> int:
        return sum(int(torch.tensor(p.shape).prod().item()) for p in self.parameters())

    def param_bytes(self) -> int:
        """Total bytes held by trainable parameters (the integer ``H`` buffers)."""
        return sum(p.param_bytes() for p in self.parameters())

    def state_dict(self) -> dict:
        return {
            "cfg": self.cfg.__dict__,
            "train_step": self._train_step,
            "params": {p.name: p.state_dict() for p in self.parameters()},
        }

    def load_state_dict(self, sd: dict) -> None:
        self._train_step = int(sd.get("train_step", 0))
        byname = {p.name: p for p in self.parameters()}
        for name, psd in sd["params"].items():
            if name in byname:
                byname[name].load_state_dict(psd)

    # ── flip-dropout (train against the deployment bit-flip noise model) ──────
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

        emb = self.codebook.embed(ids)
        c = self.input_bind.forward(emb)
        flip_masks = []
        for blk in self.blocks:
            c, fbit = self._flip_noise(c)
            flip_masks.append(fbit)
            c = blk.forward(c)
        chat = self.out_bind.forward(c)

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
            "chat_flat": chat_flat, "flip_masks": flip_masks, "ids": ids,
        }
        return logits

    def _synth_induction_matched(self, targets: torch.Tensor) -> torch.Tensor:
        """Most recent previous position whose follower equals this target.

        The self-supervised induction signal: for query ``i`` with target token
        ``t``, find the latest ``m < i`` whose own target equals ``t`` — i.e. the
        position attention should copy from.  Returns ``(B, n)`` long (``-1`` =
        none).
        """
        B, n = self._fwd_cache["B"], self._fwd_cache["n"]
        tgt = targets.to(self.device).reshape(B, n).long()
        valid_t = tgt != IGNORE_INDEX
        same_follower = tgt.unsqueeze(2) == tgt.unsqueeze(1)      # (B,t,m)
        t = torch.arange(n, device=self.device).unsqueeze(1)
        m = torch.arange(n, device=self.device).unsqueeze(0)
        causal = m < t
        valid = same_follower & valid_t.unsqueeze(2) & valid_t.unsqueeze(1) & causal.unsqueeze(0)
        idx = torch.arange(n, device=self.device).view(1, 1, n).expand(B, n, n)
        return torch.where(valid, idx, torch.full_like(idx, -1)).max(dim=2).values

    def _effective_margin_r(self) -> float:
        cfg = self.cfg
        if cfg.margin_r_final is None or cfg.margin_anneal_steps <= 0:
            return float(cfg.r)
        post_warmup = max(0, self._train_step - cfg.readout_warmup_steps)
        t = min(1.0, post_warmup / max(1, cfg.margin_anneal_steps))
        return float(cfg.r + (cfg.margin_r_final - cfg.r) * t)

    def _cap_hidden_trigger(self, trigger: torch.Tensor, deficit: torch.Tensor,
                            valid: torch.Tensor) -> torch.Tensor:
        cap_rate = self.cfg.max_trigger_rate
        if cap_rate is None:
            return trigger
        n_valid = int(valid.sum().item())
        n_trigger = int(trigger.sum().item())
        cap = max(1, int(math.ceil(n_valid * float(cap_rate)))) if n_valid else 0
        if n_trigger <= cap:
            return trigger
        priority = torch.where(trigger, deficit, torch.full_like(deficit, float("-inf")))
        keep = torch.topk(priority, k=cap, dim=0).indices
        capped = torch.zeros_like(trigger)
        capped[keep] = True
        return capped

    # ── loss + BEP backward (contrastive margin trigger) ─────────────────────
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

        # BEP margin trigger: fire where logit[tgt] - max_other < r_eff*D.
        effective_r = self._effective_margin_r()
        logit_tgt = logits.gather(1, safe_tgt.unsqueeze(1)).squeeze(1)
        other = logits.clone()
        other.scatter_(1, safe_tgt.unsqueeze(1), float("-inf"))
        max_other, wrong = other.max(dim=1)
        margin_deficit = effective_r * D - (logit_tgt - max_other)
        trigger = valid & (margin_deficit > 0)
        readout_warmup = (
            cfg.train_output_codebook
            and cfg.readout_warmup_steps > 0
            and self._train_step < cfg.readout_warmup_steps
        )
        if readout_warmup:
            hidden_trigger = torch.zeros_like(trigger)
        else:
            hidden_trigger = self._cap_hidden_trigger(trigger, margin_deficit, valid)
        trig_rate = float(hidden_trigger.float().mean().item()) if M else 0.0
        margin_trig_rate = float(trigger.float().mean().item()) if M else 0.0

        # Desired activation is discriminative: agreement coords carry no opinion.
        proto = self.codebook.prototype(safe_tgt)          # (M, D) bit1
        wrong_proto = self.codebook.prototype(wrong)
        agree = brute.fast.eq(proto, wrong_proto)
        sel = brute.as_tensor(hidden_trigger.unsqueeze(1).expand(M, D).contiguous(),
                              dtype=brute.bit1)
        ell_bit = self._fwd_cache["ell_bit"]
        ell_margin_des = mux(agree, ell_bit, proto)
        ell_des = mux(sel, ell_margin_des, ell_bit)
        chat_flat = self._fwd_cache["chat_flat"]
        if cfg.train_output_codebook:
            decode_update = trigger & (pred != tgt)
            self.codebook.backward_decode(ell_bit, safe_tgt, wrong, decode_update)
        # only triggered positions inject a backward signal (BEP); the rest carry
        # no opinion (desired = current activation, no weight update).
        bep.set_active(hidden_trigger)
        g_chat = self.lex_proj.backward(ell_des)

        sem_bit = self._fwd_cache["sem_bit"]
        if sem_bit is not None:
            sem_margin_des = mux(agree, sem_bit, proto)
            sem_des = mux(sel, sem_margin_des, sem_bit)
            g_chat_sem = self.sem_bind.backward(sem_des)
            g_chat = combine_desired(g_chat, g_chat_sem, chat_flat)
        g_chat = g_chat.reshape(B, n, D)

        g_c = self.out_bind.backward(g_chat)

        margin_val = 0.0
        if (matched is None and cfg.self_supervised_induction and cfg.margin_weight > 0):
            matched = self._synth_induction_matched(targets)
        if matched is not None and cfg.margin_weight > 0:
            active_rows = hidden_trigger.reshape(B, n)
            for blk in self.blocks:
                margin_val += blk.margin_loss(
                    matched.to(self.device), theta_pos=cfg.margin_theta_pos,
                    theta_neg=cfg.margin_theta_neg, active_rows=active_rows)

        flip_masks = self._fwd_cache["flip_masks"]
        for blk, fbit in zip(reversed(self.blocks), reversed(flip_masks)):
            g_c = blk.backward(g_c)
            if fbit is not None:
                g_c = brute.fast.bitwise_xor(g_c, fbit)    # un-flip the dropout noise
        g_emb = self.input_bind.backward(g_c)
        self.codebook.backward_embed(g_emb)
        bep.set_active(None)
        self._train_step += 1

        return {"loss": loss, "acc": acc, "n_valid": n_valid, "margin": margin_val,
                "trigger_rate": trig_rate,
                "margin_trigger_rate": margin_trig_rate,
                "effective_r": effective_r,
                "readout_warmup": readout_warmup,
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

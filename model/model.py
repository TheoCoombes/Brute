"""HÆMMR — a binary-first, concept-native autoregressive language model.

Assembles the Boolean block stack from :mod:`layers`:

    token → E(t) ⊗ ρ^i(POS) ── input-bind ──▶ [ BSR ⊕ Hopfield ⊕ channel-mix ]×L
        ──▶ out-bind ──▶ concept ĉ ──▶ min-Hamming decode against E ──▶ logits

Forward is bitwise (packed ``brute.bit1``); the only integers are the signed
matmul pre-activations and the BSR accumulator, as the paper prescribes.
Training is BOLD: a cross-entropy loss over the decode similarities produces a
real signal that the layers' ``backward`` passes turn into per-weight flip
signals (see :mod:`bold`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

import brute

from bold import BoldParam
from layers import (
    BSR, BooleanLinear, DiagBind, HopfieldBank, ResidualMerge, TokenCodebook,
)
from vsa import bind, position_codes, to_bit1, to_pm1


IGNORE_INDEX = -1


@dataclass
class HaemmrConfig:
    vocab_size: int = 2048
    D: int = 1024                 # concept hypervector dimension
    n_layers: int = 2
    d_ff: int = 2048              # channel-mix hidden width
    n_slots: int = 256            # Hopfield bank slots (M)
    top_k: int = 15               # Hopfield winner-take-all width (odd → no ties)
    gamma_min: float = 0.90       # BSR decay spread (fixed, multi-timescale)
    gamma_max: float = 0.999
    logit_temp: Optional[float] = None  # train-time logit scale; default 1/sqrt(D)
    use_position: bool = True     # bind ρ^i(POS) into the concept (else order via BSR decay)
    position_decode: Optional[str] = None  # None -> next_unbind if positioned, else none
    gate_open: float = 0.05       # residual-gate init openness (more ⇒ context flows sooner)
    codebook_flip_scale: float = 0.3   # codebook flip-rate vs transforms (lower = more stable)
    seed: int = 0

    def __post_init__(self):
        if self.logit_temp is None:
            self.logit_temp = float(self.D) ** 0.5
        if self.position_decode is None:
            self.position_decode = "next_unbind" if self.use_position else "none"
        valid = {"none", "next_unbind"}
        if self.position_decode not in valid:
            raise ValueError(f"position_decode must be one of {sorted(valid)}, got {self.position_decode!r}")
        if not self.use_position and self.position_decode != "none":
            raise ValueError("position_decode requires use_position=True")


# ── channel-mixing MLP (two Boolean linears) ───────────────────────────────────

class ChannelMix:
    def __init__(self, D: int, d_ff: int, *, name: str, generator=None, device=None):
        self.D, self.d_ff = D, d_ff
        self.lin1 = BooleanLinear(D, d_ff, name=f"{name}.lin1", generator=generator, device=device)
        self.lin2 = BooleanLinear(d_ff, D, name=f"{name}.lin2", generator=generator, device=device)

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
        self.bsr = BSR(D, name=f"{nm}.bsr", gamma_min=cfg.gamma_min,
                       gamma_max=cfg.gamma_max, generator=generator, device=device)
        self.merge_bsr = ResidualMerge(D, name=f"{nm}.merge_bsr", p_open=cfg.gate_open,
                                       generator=generator, device=device)
        self.hop = HopfieldBank(D, cfg.n_slots, cfg.top_k, name=f"{nm}.hop",
                                generator=generator, device=device)
        self.merge_hop = ResidualMerge(D, name=f"{nm}.merge_hop", p_open=cfg.gate_open,
                                       generator=generator, device=device)
        self.mix = ChannelMix(D, cfg.d_ff, name=f"{nm}.mix", generator=generator, device=device)
        self.merge_mix = ResidualMerge(D, name=f"{nm}.merge_mix", p_open=cfg.gate_open,
                                       generator=generator, device=device)
        self._shape = None

    def params(self) -> List[BoldParam]:
        return (self.bsr.params() + self.merge_bsr.params() + self.hop.params()
                + self.merge_hop.params() + self.mix.params() + self.merge_mix.params())

    def forward(self, c_bit: brute.Tensor) -> brute.Tensor:
        B, n, D = c_bit.shape
        self._shape = (B, n, D)
        r = self.bsr.forward(c_bit)                            # context (B,n,D)
        c1 = self.merge_bsr.forward(c_bit, r)
        h = self.hop.forward(c1.reshape(B * n, D)).reshape(B, n, D)
        c2 = self.merge_hop.forward(c1, h)
        m = self.mix.forward(c2.reshape(B * n, D)).reshape(B, n, D)
        c3 = self.merge_mix.forward(c2, m)
        return c3

    def backward(self, S_c3: torch.Tensor) -> torch.Tensor:
        B, n, D = self._shape
        S_c2_a, S_m = self.merge_mix.backward(S_c3)
        S_c2_b = self.mix.backward(S_m.reshape(B * n, D)).reshape(B, n, D)
        S_c2 = S_c2_a + S_c2_b
        S_c1_a, S_h = self.merge_hop.backward(S_c2)
        S_c1_b = self.hop.backward(S_h.reshape(B * n, D)).reshape(B, n, D)
        S_c1 = S_c1_a + S_c1_b
        S_c_a, S_r = self.merge_bsr.backward(S_c1)
        S_c_b = self.bsr.backward(S_r)
        return S_c_a + S_c_b


# ── the model ────────────────────────────────────────────────────────────────

class HaemmrLM:
    def __init__(self, cfg: HaemmrConfig, *, device=None):
        self.cfg = cfg
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))

        self.codebook = TokenCodebook(cfg.vocab_size, cfg.D, name="E",
                                      flip_scale=cfg.codebook_flip_scale,
                                      generator=gen, device=self.device)
        self.input_bind = DiagBind(cfg.D, name="in", generator=gen, device=self.device)
        self.blocks = [Block(cfg, idx=i, generator=gen, device=self.device)
                       for i in range(cfg.n_layers)]
        self.out_bind = DiagBind(cfg.D, name="out", generator=gen, device=self.device)

        # Fixed position base hypervector (cyclic-shift permutation encoding).
        pos_bits = torch.randint(0, 2, (cfg.D,), generator=gen).bool().to(self.device)
        self.pos_base_pm1 = (pos_bits.float() * 2 - 1)
        self._pos_cache: dict = {}
        self._fwd_cache: dict = {}

    # ── parameters / checkpoint ──────────────────────────────────────────────
    def parameters(self) -> List[BoldParam]:
        ps = self.codebook.params() + self.input_bind.params()
        for b in self.blocks:
            ps += b.params()
        ps += self.out_bind.params()
        return ps

    def num_bit_parameters(self) -> int:
        return sum(int(torch.tensor(p.shape).prod().item()) for p in self.parameters())

    def state_dict(self) -> dict:
        return {
            "cfg": self.cfg.__dict__,
            "pos_base_pm1": self.pos_base_pm1.detach().cpu().clone(),
            "params": {p.name: p.state_dict() for p in self.parameters()},
        }

    def load_state_dict(self, sd: dict) -> None:
        self.pos_base_pm1 = sd["pos_base_pm1"].to(self.device)
        byname = {p.name: p for p in self.parameters()}
        for name, psd in sd["params"].items():
            if name in byname:
                byname[name].load_state_dict(psd)
        self._pos_cache.clear()

    # ── positions ────────────────────────────────────────────────────────────
    def _positions(self, n: int):
        if self._pos_cache.get("n", -1) < n:
            codes = position_codes(self.pos_base_pm1, n)        # (n, D) bit1
            self._pos_cache = {"n": n, "bit": codes, "pm1": to_pm1(codes)}
        return self._pos_cache["bit"], self._pos_cache["pm1"]

    # ── forward ────────────────────────────────────────────────────────────
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """``ids`` (B, n) long → logits (B*n, V) int32 (signed similarities)."""
        ids = ids.to(self.device)
        B, n = ids.shape
        D = self.cfg.D
        emb = self.codebook.embed(ids)                          # (B,n,D)
        decode_pos_pm1 = None
        if self.cfg.use_position:
            need_pos = n + 1 if self.cfg.position_decode == "next_unbind" else n
            pos_bit, pos_pm1_all = self._positions(need_pos)
            c = bind(emb, pos_bit[:n])                          # E(t) ⊗ ρ^i(POS)
            pos_pm1 = pos_pm1_all[:n]
        else:
            c = emb                                            # order comes from BSR decay
            pos_pm1 = None
        c = self.input_bind.forward(c)                          # learned concept dressing
        for blk in self.blocks:
            c = blk.forward(c)
        chat = self.out_bind.forward(c)                        # next-token concept ĉ
        decode_chat = chat
        if self.cfg.position_decode == "next_unbind":
            decode_pos_bit = pos_bit[1:n + 1]
            decode_pos_pm1 = pos_pm1_all[1:n + 1]
            decode_chat = bind(chat, decode_pos_bit)            # unrole next-position concept
        logits = self.codebook.decode(decode_chat.reshape(B * n, D))   # (B*n, V)
        self._fwd_cache = {
            "B": B, "n": n, "pos_pm1": pos_pm1, "decode_pos_pm1": decode_pos_pm1,
        }
        return logits

    # ── loss + BOLD backward ─────────────────────────────────────────────────
    def loss_and_backward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        B, n = self._fwd_cache["B"], self._fwd_cache["n"]
        D = self.cfg.D
        inv = 1.0 / float(self.cfg.logit_temp)
        tgt = targets.to(self.device).reshape(-1).long()        # (B*n,)
        valid = tgt != IGNORE_INDEX
        n_valid = int(valid.sum().item())

        scaled = logits.float() * inv
        logp = torch.log_softmax(scaled, dim=1)
        p = logp.exp()

        safe_tgt = tgt.clamp_min(0)
        nll = -logp.gather(1, safe_tgt.unsqueeze(1)).squeeze(1)
        nll = nll[valid]
        loss = float(nll.mean().item()) if n_valid else 0.0
        pred = scaled.argmax(dim=1)
        acc = float((pred[valid] == tgt[valid]).float().mean().item()) if n_valid else 0.0

        # ∂(mean CE)/∂logit = (p − onehot)·inv / N_valid, zeroed on ignored rows.
        onehot = torch.zeros_like(p)
        onehot.scatter_(1, safe_tgt.unsqueeze(1), 1.0)
        S_logits = (p - onehot) * (inv / max(n_valid, 1))
        S_logits[~valid] = 0.0

        # backward through the stack
        g_chat = self.codebook.backward_decode(S_logits).reshape(B, n, D)
        decode_pos_pm1 = self._fwd_cache["decode_pos_pm1"]      # (n,D) or None
        if decode_pos_pm1 is not None:
            g_chat = g_chat * decode_pos_pm1
        g_c = self.out_bind.backward(g_chat.reshape(B, n, D))
        for blk in reversed(self.blocks):
            g_c = blk.backward(g_c)
        g_posbound = self.input_bind.backward(g_c)              # (B,n,D)
        pos_pm1 = self._fwd_cache["pos_pm1"]                    # (n,D) or None
        g_emb = g_posbound * pos_pm1 if pos_pm1 is not None else g_posbound
        self.codebook.backward_embed(g_emb)

        return {"loss": loss, "acc": acc, "n_valid": n_valid,
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
        logits = self.forward(ids).reshape(B, n, self.cfg.vocab_size)
        return logits[:, -1, :].float()

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, n_new: int, *, temperature: float = 1.0,
                 top_k: int = 0, ban_ids: Optional[List[int]] = None,
                 repetition_window: int = 0, max_ctx: int = 256) -> torch.Tensor:
        """Autoregressive min-Hamming / Boltzmann decoding (HÆMMR §6).

        Recomputes the (linear-time) forward pass on the growing context each
        step — simple and correct; a streaming BSR-state cache would make this
        O(1)/token but is omitted for clarity.
        """
        ids = ids.to(self.device)
        inv = 1.0 / float(self.cfg.logit_temp)
        for _ in range(n_new):
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
        return ids

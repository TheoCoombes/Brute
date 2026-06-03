"""Binary Error Propagation (BEP) — fully binary RNN *language model*.

This is the language-domain analogue of the BEP MLP demo in ``examples/bep``.
Where the MLP version trains a depth-``L`` binary feed-forward network, here we
train a single binary recurrent cell unrolled through time, predicting the
next token at every position.

Reference
---------
Colombo, Pittorino, Zambon, Baldassi, Roveri, Alippi.
"BEP: A Binary Error Propagation Algorithm for Binary Neural Networks
Training." ICLR 2026 (arXiv:2512.04189). The paper validates BEP on both
binary MLPs (Sec. 4.2) and binary RNNs (Sec. 4.3); this file targets the
RNN setting.

Architecture at a glance
------------------------
* A single BEF codebook ``P ∈ {±1}^{V × K_h}`` plays two roles:
    - as **input embedding**: token ``v`` is represented as ``e_v = P[v]``;
    - as **output classifier**: logits at time ``t`` are ``ŷ_t = P a_t``.
  Re-using ``P`` is what keeps the model fully binary without learning a
  separate embedding table — the prototypes already form a maximally
  separated ±1 codebook, exactly the property a token embedding wants.

* Two integer hidden weight matrices, both in the BEP integer
  metaplasticity range ``[-2^{B-1}, 2^{B-1}-1]`` (default B=16):
    - ``H_xh ∈ ℤ^{K_h × K_h}``  — input-to-hidden  ``(W_xh = sign(H_xh))``
    - ``H_hh ∈ ℤ^{K_h × K_h}``  — recurrent        ``(W_hh = sign(H_hh))``
  Both ``W`` matrices are stored as packed ``brute.bit1`` for the
  XNOR/popcount fast path.

* Forward (per time step ``t = 0..T-1``)::
      a_{-1} = +1 vector                                                # init state
      e_t    = P[x_t]                                                   # ±1 embedding
      z_t    = W_xh e_t + W_hh a_{t-1}                                  # int32 ±1 dot
      a_t    = sign(z_t)                                                # ±1
      ŷ_t    = P a_t                                                    # int32 (B, V)

* Update trigger (Eq. 1, per timestep): margin = ŷ_t[target_t] − max_other.
  Time ``t`` is triggered iff ``margin < r · K_h``.

* Backward (Eq. 7) — fully binary BPTT::
      Walk t = T-1 .. 0.
      a*_t^{direct} = ρ^{target_t}                  if margin_t triggers
      a*_t^{recur}  = sign(W_hh^T (g_{t+1} ⊙ a*_{t+1}))   if a*_{t+1} exists
                       with g_{t+1,i} = 1 iff |z_{t+1,i}| ≤ ν · 2K_h
      a*_t          = sign(direct + recur)            # sign(0) = +1
  A position is "active" if at least one of the two contributions exists.

* Update (Eq. 9, winner-takes-update inside neuron groups, broadcast over
  every active (batch, timestep) position). At each active position the
  *joint* stability of neuron ``j`` is the full pre-activation contribution
  from *both* matrices::
      stability[μ, j] = a*_t[μ, j] · z_t[μ, j]
                      = a*_t[μ, j] · (W_xh e_t + W_hh a_{t-1})[μ, j]
      mask M^{μ}      = per-group argmin of |stability| over neg. neurons
      H_xh ← H_xh + 2 · Σ_{μ,t}  (M^{μ,t} ⊙ a*_t^{μ})^T  e_t^{μ}
      H_hh ← H_hh + 2 · Σ_{μ,t}  (M^{μ,t} ⊙ a*_t^{μ})^T  a_{t-1}^{μ}
  Using the *joint* z keeps the rule consistent with the Baldassi 2009
  CP+R baseline: the neuron is "misclassified" iff the full integrated
  signal has the wrong sign, in which case BOTH incoming matrices receive
  the same nudge toward the desired output.

* Reinforcement (Sec. 3.3) drifts non-zero weights further from zero with
  probability ``p_r √(2/(π K_in))``; ``K_in = K_h`` for both matrices.

Every dot product is a ``brute.bit1`` XNOR/popcount when its inputs are
strictly ±1; gated products (where one operand has zero entries) use a
small int32 matmul on the unpacked ±1 weights, **only on the small set of
active positions**. The only float arithmetic is the integer-weight
accumulator (int32 headroom, clipped to int16 range).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt, pi
from typing import Optional

import torch

import brute


# Sentinel target id meaning "ignore this position" in step / accuracy.
IGNORE_INDEX: int = -1


# ── Tiny utilities (bit1 ↔ ±1) ────────────────────────────────────────────────

def _pm1_to_bit1(t: torch.Tensor) -> brute.Tensor:
    """Encode a ±1 tensor (any non-bool dtype is fine) as a packed bit1 tensor.

    Convention: ``+1 ↦ True (1)``, ``-1 ↦ False (0)``.
    """
    if t.dtype == torch.bool:
        return brute.as_tensor(t, dtype=brute.bit1)
    return brute.as_tensor(t > 0, dtype=brute.bit1)


def _bit1_matmul_pm1(a_bit: brute.Tensor, w_bit: brute.Tensor) -> torch.Tensor:
    """``A (B, K) @ W (N, K)`` where both are bit1; returns int32 ``(B, N)``
    in the ±1 dot-product domain (the brute kernel contracts over the last
    dim of both operands)."""
    return a_bit @ w_bit


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class BEPConfig:
    """Hyperparameters for BEP RNN-LM training (paper Sec. 4)."""

    r: float = 0.5
    """Trigger margin in Eq. 1 — per-timestep update if
    ``correct_logit − max_other < r · K_h``."""

    nu: float = 0.05
    """Backward gating threshold (Eq. 5). Neuron ``i`` of step ``t+1`` is
    included in the backward signal iff ``|z_{t+1,i}| ≤ ν · 2K_h`` (the
    pre-activation aggregates ``K_h + K_emb = 2K_h`` ±1 inputs)."""

    group_size_init: int = 4
    """Initial winner-takes-update group size γ_0; must divide ``K_h``."""

    p_reinforce: float = 0.5
    """Reinforcement probability (CP+R). Rescaled per matrix by
    ``√(2/(π K_in))``."""

    weight_clip: int = 2048
    """Range for ``|H_{·}|``. Paper uses B=16 ⇒ ±32 767; smaller is faster
    and still trains stably."""

    h_init_std: float = 4.0
    """Initial standard deviation for the integer hidden weights."""

    use_reinforcement: bool = True
    """If True, apply the Sec. 3.3 reinforcement step every BEP update."""


# ── Forward-pass state ───────────────────────────────────────────────────────

@dataclass
class BEPLMState:
    """Per-timestep intermediates cached during the forward pass.

    Shapes (using ``B`` = batch, ``T`` = sequence length, ``K`` = hidden,
    ``V`` = vocab):

    * ``a_prev``  : int8 ``(B, T, K)`` ±1 — recurrent state *before* each
                    step. ``a_prev[:, 0]`` is the init state (all +1) and
                    ``a_prev[:, t]`` for ``t > 0`` is the state produced
                    by step ``t-1``.
    * ``a_curr``  : int8 ``(B, T, K)`` ±1 — recurrent state *after* each
                    step (i.e. ``a_curr[:, t] = sign(z_t)``).
    * ``a_curr_bit`` : list of length ``T`` of packed bit1 ``(B, K)`` views
                    of ``a_curr[:, t]`` (the input to the per-step output
                    projection and to the next step's W_hh).
    * ``e_pm1``   : int8 ``(B, T, K)`` ±1 — input embeddings ``e_t``.
    * ``z``       : int32 ``(B, T, K)`` — pre-activations.
    * ``logits``  : int32 ``(B, T, V)`` — per-step logits ``ŷ_t = P a_t``.
    """

    a_prev: torch.Tensor = None       # (B, T, K) int8
    a_curr: torch.Tensor = None       # (B, T, K) int8
    a_curr_bit: list[brute.Tensor] = field(default_factory=list)
    e_pm1: torch.Tensor = None        # (B, T, K) int8
    z: torch.Tensor = None            # (B, T, K) int32
    logits: torch.Tensor = None       # (B, T, V) int32


# ── BEP RNN language model ───────────────────────────────────────────────────

class BEPLanguageModel:
    """Fully binary Elman RNN language model trained with BEP.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size ``V`` exposed to the model.
    hidden_size : int
        Width ``K_h`` of the recurrent state. Must be divisible by
        ``config.group_size_init``.
    classifier : torch.Tensor
        ±1 prototype matrix ``P`` of shape ``(V, K_h)``. Generated by
        :func:`bef.generate_bef`. Used both as input embedding **and** as
        the fixed output classifier.
    config : BEPConfig
    device : torch.device, optional
    seed : int, optional
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        classifier: torch.Tensor,
        *,
        config: Optional[BEPConfig] = None,
        device: Optional[torch.device] = None,
        seed: Optional[int] = 0,
    ):
        self.config = config or BEPConfig()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.vocab_size = int(vocab_size)
        self.K_h = int(hidden_size)

        assert classifier.dim() == 2, "classifier P must be 2-D"
        assert classifier.shape[0] == self.vocab_size, (
            f"classifier rows {classifier.shape[0]} != vocab_size {self.vocab_size}"
        )
        assert classifier.shape[1] == self.K_h, (
            f"classifier dim {classifier.shape[1]} != hidden_size {self.K_h}"
        )
        if self.K_h % self.config.group_size_init != 0:
            raise ValueError(
                f"group_size_init={self.config.group_size_init} must divide "
                f"hidden_size={self.K_h}"
            )

        self._P_pm1 = classifier.to(self.device).to(torch.int8)
        self._P_bit = _pm1_to_bit1(self._P_pm1)

        # Integer hidden weights.
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))
        self.H_xh = (torch.randn(self.K_h, self.K_h, generator=gen)
                     * self.config.h_init_std).round().to(torch.int32).to(self.device)
        self.H_hh = (torch.randn(self.K_h, self.K_h, generator=gen)
                     * self.config.h_init_std).round().to(torch.int32).to(self.device)

        # Packed visible weights, refreshed lazily.
        self._W_xh_bit: Optional[brute.Tensor] = None
        self._W_hh_bit: Optional[brute.Tensor] = None
        self._xh_ver = -1
        self._hh_ver = -1

    # ── Visible-weight cache ────────────────────────────────────────────────

    def _refresh_xh(self) -> None:
        ver = self.H_xh._version
        if self._xh_ver == ver:
            return
        sign_w = self.H_xh >= 0
        self._W_xh_bit = brute.as_tensor(sign_w, dtype=brute.bit1)
        self._xh_ver = ver

    def _refresh_hh(self) -> None:
        ver = self.H_hh._version
        if self._hh_ver == ver:
            return
        sign_w = self.H_hh >= 0
        self._W_hh_bit = brute.as_tensor(sign_w, dtype=brute.bit1)
        self._hh_ver = ver

    def visible_W_xh(self) -> brute.Tensor:
        self._refresh_xh()
        return self._W_xh_bit

    def visible_W_hh(self) -> brute.Tensor:
        self._refresh_hh()
        return self._W_hh_bit

    # ── Forward pass (recurrent unroll) ─────────────────────────────────────

    def forward(self, tokens: torch.Tensor) -> BEPLMState:
        """Run the binary RNN forward on ``(B, T)`` input token ids.

        Parameters
        ----------
        tokens : torch.LongTensor
            Input token ids of shape ``(B, T)``; values in ``[0, V)``.

        Returns
        -------
        BEPLMState
        """
        tokens = tokens.to(self.device).long()
        B, T = tokens.shape
        K = self.K_h
        V = self.vocab_size

        W_xh_bit = self.visible_W_xh()
        W_hh_bit = self.visible_W_hh()

        # Gather embeddings once: e_pm1[b, t, :] = P[tokens[b, t]] ∈ {±1}^K.
        emb_pm1 = self._P_pm1[tokens]                        # (B, T, K) int8 ±1

        # Pre-allocate the (B, T, ·) buffers.
        a_prev_buf = torch.empty(B, T, K, dtype=torch.int8, device=self.device)
        a_curr_buf = torch.empty(B, T, K, dtype=torch.int8, device=self.device)
        z_buf      = torch.empty(B, T, K, dtype=torch.int32, device=self.device)
        logit_buf  = torch.empty(B, T, V, dtype=torch.int32, device=self.device)
        a_curr_bit_list: list[brute.Tensor] = []

        # Init state a_{-1} = +1 (replicated over the batch).
        a_prev_pm1 = torch.ones((B, K), dtype=torch.int8, device=self.device)
        a_prev_bit = _pm1_to_bit1(a_prev_pm1)

        for t in range(T):
            e_t_pm1 = emb_pm1[:, t].contiguous()             # (B, K) int8 ±1
            e_t_bit = _pm1_to_bit1(e_t_pm1)

            # z_t = W_xh e_t + W_hh a_{t-1}.
            z_in  = _bit1_matmul_pm1(e_t_bit,    W_xh_bit)   # (B, K) int32
            z_rec = _bit1_matmul_pm1(a_prev_bit, W_hh_bit)   # (B, K) int32
            z_t   = z_in + z_rec

            a_t_bit = _pm1_to_bit1(z_t > 0)
            a_t_pm1 = a_t_bit.unpack_pm1().to(torch.int8)

            a_prev_buf[:, t] = a_prev_pm1
            a_curr_buf[:, t] = a_t_pm1
            z_buf[:, t] = z_t
            logit_buf[:, t] = _bit1_matmul_pm1(a_t_bit, self._P_bit)
            a_curr_bit_list.append(a_t_bit)

            a_prev_pm1 = a_t_pm1
            a_prev_bit = a_t_bit

        return BEPLMState(
            a_prev=a_prev_buf,
            a_curr=a_curr_buf,
            a_curr_bit=a_curr_bit_list,
            e_pm1=emb_pm1,
            z=z_buf,
            logits=logit_buf,
        )

    # ── Backward + update (BPTT BEP) ────────────────────────────────────────

    def step(self, state: BEPLMState, targets: torch.Tensor) -> dict:
        """One BPTT BEP weight-update step (Eq. 7 + Eq. 9, unrolled in time).

        Parameters
        ----------
        state : BEPLMState
            Returned by :meth:`forward`.
        targets : torch.LongTensor
            Per-timestep target token ids of shape ``(B, T)``. Use
            :data:`IGNORE_INDEX` (-1) to disable a position.

        Returns
        -------
        dict
            Diagnostics: ``n_triggered`` positions, ``n_correct`` positions
            (argmax), ``n_seen`` (valid positions), ``acc``.
        """
        targets = targets.to(self.device).long()
        logits = state.logits                              # (B, T, V) int32
        B, T, V = logits.shape
        K = self.K_h

        valid = targets != IGNORE_INDEX                    # (B, T) bool
        tgt_clamped = targets.clamp(min=0)                 # safe for gather

        # ── (1) Triggering rule (Eq. 1, per timestep) ───────────────────────
        correct_logit = logits.gather(2, tgt_clamped.unsqueeze(2)).squeeze(2).to(torch.float32)
        masked = logits.to(torch.float32).clone()
        masked.scatter_(2, tgt_clamped.unsqueeze(2), float("-inf"))
        max_other = masked.max(dim=2).values                 # (B, T)
        margin = correct_logit - max_other                    # (B, T)
        trigger = (margin < self.config.r * K) & valid        # (B, T) bool

        pred = logits.argmax(dim=2)                           # (B, T)
        n_correct = int(((pred == targets) & valid).sum().item())
        n_trig = int(trigger.sum().item())
        n_seen = int(valid.sum().item())

        if n_trig == 0:
            return {
                "n_triggered": 0,
                "n_correct": n_correct,
                "n_seen": n_seen,
                "acc": n_correct / max(n_seen, 1),
            }

        # Gating threshold ν · (K_h + K_emb) = ν · 2K_h.
        gate_thresh = int(round(self.config.nu * 2 * K))

        # Unpacked ±1 W_hh — used for the gated backward matmul.
        W_hh_pm1 = (self.H_hh >= 0).to(torch.int8) * 2 - 1     # (K, K)

        # ── (2) BPTT desired activations (Eq. 7) ────────────────────────────
        a_star_buf = torch.zeros((B, T, K), dtype=torch.int8, device=self.device)
        active_buf = torch.zeros((B, T), dtype=torch.bool, device=self.device)

        a_star_next: Optional[torch.Tensor] = None
        active_next: Optional[torch.Tensor] = None

        for t in range(T - 1, -1, -1):
            direct_sig = trigger[:, t]                       # (B,)
            # Direct desired hidden state at step t is the prototype of the
            # target token; rows of P (int8 ±1).
            direct = self._P_pm1[tgt_clamped[:, t]]          # (B, K) int8 ±1
            direct_contrib = direct * direct_sig.to(torch.int8).unsqueeze(1)

            if a_star_next is not None:
                # Gate on next step's pre-activation.
                z_next = state.z[:, t + 1]                    # (B, K) int32
                gate = (z_next.abs() <= gate_thresh).to(torch.int8)
                gated = a_star_next * gate                    # (B, K) ∈ {-1,0,1}
                # W_hh^T applied to (gate ⊙ a*_{t+1}) — int32 matmul because
                # gated has zero entries. Active mask is applied via
                # ``active_next`` row-zeroing.
                pre = gated.to(torch.int32) @ W_hh_pm1.to(torch.int32)   # (B, K)
                recur = torch.where(pre > 0,
                                    torch.ones_like(pre, dtype=torch.int8),
                                    -torch.ones_like(pre, dtype=torch.int8))
                recur_contrib = recur * active_next.to(torch.int8).unsqueeze(1)
            else:
                recur_contrib = torch.zeros((B, K), dtype=torch.int8, device=self.device)

            combined = direct_contrib + recur_contrib       # (B, K) ∈ {-2..+2}
            # sign(x): +1 for x ≥ 0, -1 otherwise (matches W = sign(H) convention).
            a_star_t = torch.where(combined >= 0,
                                   torch.ones_like(combined, dtype=torch.int8),
                                   -torch.ones_like(combined, dtype=torch.int8))

            if a_star_next is not None:
                active_t = direct_sig | active_next
            else:
                active_t = direct_sig

            # Zero out a*_t on inactive positions so it can't accidentally
            # contribute to the aggregate update.
            a_star_t = a_star_t * active_t.to(torch.int8).unsqueeze(1)

            a_star_buf[:, t] = a_star_t
            active_buf[:, t] = active_t

            a_star_next = a_star_t
            active_next = active_t

        # ── (3) Aggregate weight update across active positions ─────────────
        # Flatten (B, T) → (BT,) and sub-select active positions.
        a_star_flat = a_star_buf.reshape(B * T, K).to(torch.int32)         # (BT, K)
        active_flat = active_buf.reshape(B * T)                            # (BT,)
        if not active_flat.any():
            return {
                "n_triggered": n_trig,
                "n_correct": n_correct,
                "n_seen": n_seen,
                "acc": n_correct / max(n_seen, 1),
            }
        idx_active = active_flat.nonzero(as_tuple=False).squeeze(1)        # (M,)

        a_star_act = a_star_flat[idx_active]                                # (M, K) int32
        e_act      = state.e_pm1.reshape(B * T, K)[idx_active].to(torch.int32)
        a_prev_act = state.a_prev.reshape(B * T, K)[idx_active].to(torch.int32)
        z_act      = state.z.reshape(B * T, K)[idx_active].to(torch.int32)

        # ── (3a) Joint winner-takes-update mask ─────────────────────────────
        # Stability uses the *full* pre-activation z_t (sum of both matrices'
        # contributions): a neuron is "wrong" iff its joint dot product
        # disagrees with the desired output. Same mask for H_xh and H_hh.
        stability = a_star_act * z_act                                     # (M, K) int32
        group_size = self.config.group_size_init
        n_groups = K // group_size
        stab_r = stability.view(-1, n_groups, group_size)
        negatives = stab_r < 0
        score = torch.where(negatives, stab_r.abs(),
                            torch.full_like(stab_r, 2**30))
        local_argmin = score.argmin(dim=2)                                 # (M, n_groups)
        any_neg = negatives.any(dim=2)                                      # (M, n_groups)

        mask = torch.zeros(stability.shape, dtype=torch.int32, device=self.device)
        m_idx = torch.arange(stability.shape[0], device=self.device).unsqueeze(1).expand(-1, n_groups)
        g_offset = torch.arange(n_groups, device=self.device) * group_size
        flat_neuron = g_offset.unsqueeze(0) + local_argmin
        mask[m_idx[any_neg], flat_neuron[any_neg]] = 1

        # ── (3b) Aggregate updates: ΔH = 2 · (mask ⊙ a*)^T @ prev ───────────
        masked_a_star = a_star_act * mask                                   # (M, K)
        # H_xh ← H_xh + 2 · (mask ⊙ a*)^T @ e
        delta_xh = masked_a_star.t() @ e_act                                # (K, K) int32
        self.H_xh += 2 * delta_xh
        # H_hh ← H_hh + 2 · (mask ⊙ a*)^T @ a_{t-1}
        delta_hh = masked_a_star.t() @ a_prev_act                           # (K, K) int32
        self.H_hh += 2 * delta_hh

        # ── (3c) Reinforcement (CP+R, Sec. 3.3) ───────────────────────────────
        if self.config.use_reinforcement and self.config.p_reinforce > 0:
            p = self.config.p_reinforce * sqrt(2.0 / (pi * max(K, 1)))
            p = min(max(p, 0.0), 1.0)
            if p > 0:
                for H in (self.H_xh, self.H_hh):
                    rein_mask = (torch.rand(H.shape, device=self.device) < p)
                    sign_h = torch.where(
                        H > 0,  torch.tensor(1,  device=self.device, dtype=torch.int32),
                        torch.where(
                            H < 0, torch.tensor(-1, device=self.device, dtype=torch.int32),
                            torch.tensor(0, device=self.device, dtype=torch.int32),
                        ),
                    )
                    H += 2 * sign_h * rein_mask.to(torch.int32)

        # ── (3d) Clip ───────────────────────────────────────────────────────
        clip = int(self.config.weight_clip)
        self.H_xh.clamp_(-clip, clip)
        self.H_hh.clamp_(-clip, clip)

        return {
            "n_triggered": n_trig,
            "n_correct": n_correct,
            "n_seen": n_seen,
            "acc": n_correct / max(n_seen, 1),
        }

    # ── Inference helpers ───────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, tokens: torch.Tensor) -> torch.Tensor:
        """Per-timestep argmax predictions of shape ``(B, T)``."""
        state = self.forward(tokens)
        return state.logits.argmax(dim=2)

    @torch.no_grad()
    def accuracy(self, tokens: torch.Tensor, targets: torch.Tensor,
                 *, batch_size: int = 64) -> float:
        """Next-token top-1 accuracy over a corpus of ``(B, T)`` chunks."""
        n_correct = 0
        n_seen = 0
        for i in range(0, tokens.shape[0], batch_size):
            t = tokens[i : i + batch_size].to(self.device)
            y = targets[i : i + batch_size].to(self.device)
            valid = y != IGNORE_INDEX
            pred = self.predict(t)
            n_correct += int(((pred == y) & valid).sum().item())
            n_seen += int(valid.sum().item())
        return n_correct / max(n_seen, 1)

    # ── Diagnostics ─────────────────────────────────────────────────────────

    def num_parameters(self) -> int:
        return int(self.H_xh.numel()) + int(self.H_hh.numel())

    def __repr__(self) -> str:
        return (f"BEPLanguageModel(vocab={self.vocab_size}, K_h={self.K_h}, "
                f"params={self.num_parameters():,})")

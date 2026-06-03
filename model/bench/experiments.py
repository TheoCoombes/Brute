"""Experiment catalog for HÆMMR synthetic architecture validation."""

from __future__ import annotations

from typing import Dict, List, Optional

from .runner import ExperimentSpec


GROUP_ORDER = ["ablation", "bsr", "episodic", "hopfield", "bep", "codebook", "arch"]

BASELINE_CFG = {
    "--D": "128",
    "--layers": "1",
    "--d-ff": "256",
    "--slots": "32",
    "--top-k": "3",
    "--r": "0.1",
    "--gate-open": "0.05",
    "--seed": "0",
    "--batch-size": "64",
    "--codebook-mode": "structured",
    "--bef-sweeps": "10",
    "--sem-weight": "0.5",
}


def spec(
    name: str,
    group: str,
    task: str,
    seq_len: int,
    *,
    steps: int = 300,
    hypothesis: str = "",
    overrides: Optional[Dict[str, str]] = None,
) -> ExperimentSpec:
    cfg = dict(BASELINE_CFG)
    if overrides:
        cfg.update(overrides)
    return ExperimentSpec(
        name=name,
        group=group,
        task=task,
        seq_len=seq_len,
        steps=steps,
        hypothesis=hypothesis,
        cfg_overrides=cfg,
    )


ABLATION_EXPERIMENTS: List[ExperimentSpec] = [
    spec("abl_baseline_marker", "ablation", "marker", 16,
         hypothesis="Full model should reach >=90% on marker"),
    spec("abl_baseline_copy", "ablation", "copy", 16,
         hypothesis="Full model should reach >=80% on copy"),
    spec("abl_no_bsr_marker", "ablation", "marker", 16,
         overrides={"--no-bsr": ""},
         hypothesis="Without BSR, marker should still work via episodic recall"),
    spec("abl_no_episodic_marker", "ablation", "marker", 16,
         overrides={"--no-episodic": ""},
         hypothesis="Without episodic recall, marker should degrade significantly"),
    spec("abl_no_hopfield_marker", "ablation", "marker", 16,
         overrides={"--no-hopfield": ""},
         hypothesis="Without Hopfield, marker should still work"),
    spec("abl_no_bsr_copy", "ablation", "copy", 16,
         overrides={"--no-bsr": ""},
         hypothesis="Without BSR, copy may still work via episodic recall"),
    spec("abl_no_episodic_copy", "ablation", "copy", 16,
         overrides={"--no-episodic": ""},
         hypothesis="Without episodic recall, copy should degrade"),
    spec("abl_no_hopfield_copy", "ablation", "copy", 16,
         overrides={"--no-hopfield": ""},
         hypothesis="Without Hopfield, copy should be unaffected"),
]

BSR_EXPERIMENTS: List[ExperimentSpec] = [
    spec("bsr_long_gap8", "bsr", "bsr_long", 16,
         overrides={"--D": "128", "--layers": "1"},
         hypothesis="Short gap: BSR should carry easily"),
    spec("bsr_long_gap16", "bsr", "bsr_long", 24,
         overrides={"--D": "128", "--layers": "1"},
         hypothesis="Medium gap: BSR decay matters"),
    spec("bsr_long_gap32", "bsr", "bsr_long", 40,
         overrides={"--D": "128", "--layers": "1"},
         hypothesis="Long gap: multi-scale decay needed"),
    spec("bsr_decay_fixed1", "bsr", "bsr_long", 24,
         overrides={"--decay-shifts": "1,1,1,1,1"},
         hypothesis="Single-timescale decay should be worse at long gaps"),
    spec("bsr_decay_fixed4", "bsr", "bsr_long", 24,
         overrides={"--decay-shifts": "4,4,4,4,4"},
         hypothesis="Fast decay should lose early context"),
    spec("bsr_decay_permanent", "bsr", "bsr_long", 24,
         overrides={"--decay-shifts": "0,0,0,0,0"},
         hypothesis="Permanent accumulation may overflow with noise"),
    spec("bsr_decay_multiscale", "bsr", "bsr_long", 24,
         overrides={"--decay-shifts": "1,2,3,4,0"},
         hypothesis="Default multi-scale palette should be best"),
    spec("bsr_nolayers_copy_distracted", "bsr", "copy_distracted", 32,
         overrides={"--no-bsr": ""},
         hypothesis="Without BSR, episodic alone handles distracted copy"),
    spec("bsr_with_episodic_copy_distracted", "bsr", "copy_distracted", 32,
         hypothesis="With BSR and episodic, cooperation should improve copy"),
]

EPISODIC_EXPERIMENTS: List[ExperimentSpec] = [
    spec("epi_read_k1_marker16", "episodic", "marker", 16,
         overrides={"--epi-read-k": "1"},
         hypothesis="k=1 exact read should work"),
    spec("epi_read_k2_marker16", "episodic", "marker", 16,
         overrides={"--epi-read-k": "2"},
         hypothesis="k=2 should be no worse or slightly better"),
    spec("epi_read_k4_marker16", "episodic", "marker", 16,
         overrides={"--epi-read-k": "4"},
         hypothesis="k=4 averaging may hurt exact recall"),
    spec("epi_read_k1_induction64", "episodic", "induction", 64,
         overrides={"--epi-read-k": "1"},
         hypothesis="Long-range induction with exact read"),
    spec("epi_read_k4_induction64", "episodic", "induction", 64,
         overrides={"--epi-read-k": "4"},
         hypothesis="Bundled read for induction may help or add noise"),
    spec("epi_window_small_marker32", "episodic", "marker", 32,
         overrides={"--epi-slots": "4"},
         hypothesis="Small window cannot see the antecedent"),
    spec("epi_window_full_marker32", "episodic", "marker", 32,
         hypothesis="Full window keeps antecedent reachable"),
    spec("epi_no_position_distracted", "episodic", "epi_distracted", 32,
         overrides={"--no-position": ""},
         hypothesis="Content-only scoring should work if content is unique"),
    spec("epi_with_position_distracted", "episodic", "epi_distracted", 32,
         hypothesis="Content and position should improve disambiguation"),
    spec("epi_no_margin_sup_marker", "episodic", "marker", 16,
         overrides={"--no-margin-supervision": ""},
         hypothesis="Without address-margin supervision, convergence should slow"),
]

HOPFIELD_EXPERIMENTS: List[ExperimentSpec] = [
    spec("hop_slots16_marker", "hopfield", "marker", 16,
         overrides={"--slots": "16"}, hypothesis="Small bank should be adequate"),
    spec("hop_slots64_marker", "hopfield", "marker", 16,
         overrides={"--slots": "64"}, hypothesis="Medium bank should match baseline"),
    spec("hop_slots256_marker", "hopfield", "marker", 16,
         overrides={"--slots": "256"}, hypothesis="Large bank is overparameterized"),
    spec("hop_topk1_marker", "hopfield", "marker", 16,
         overrides={"--top-k": "1"}, hypothesis="Hard top-1 is exact but brittle"),
    spec("hop_topk3_marker", "hopfield", "marker", 16,
         overrides={"--top-k": "3"}, hypothesis="Default top-k"),
    spec("hop_topk15_marker", "hopfield", "marker", 16,
         overrides={"--top-k": "15"}, hypothesis="Wide WTA may add superposition noise"),
    spec("hop_prior_task_small", "hopfield", "hopfield_prior", 16,
         overrides={"--slots": "16", "--top-k": "3"},
         hypothesis="Small Hopfield bank should learn four prior tokens"),
    spec("hop_prior_task_large", "hopfield", "hopfield_prior", 16,
         overrides={"--slots": "256", "--top-k": "15"},
         hypothesis="Large Hopfield bank should learn prior faster"),
    spec("hop_no_hopfield_prior_task", "hopfield", "hopfield_prior", 16,
         overrides={"--no-hopfield": ""},
         hypothesis="Without Hopfield, prior task should fail"),
]

BEP_EXPERIMENTS: List[ExperimentSpec] = [
    spec("bep_r005_marker", "bep", "marker", 16,
         overrides={"--r": "0.05"}, hypothesis="Tight margin: slow but precise"),
    spec("bep_r010_marker", "bep", "marker", 16,
         overrides={"--r": "0.1"}, hypothesis="Default margin"),
    spec("bep_r020_marker", "bep", "marker", 16,
         overrides={"--r": "0.2"}, hypothesis="Loose margin may add updates"),
    spec("bep_r040_marker", "bep", "marker", 16,
         overrides={"--r": "0.4"}, hypothesis="Very loose margin may destabilize"),
    spec("bep_gate000_marker", "bep", "marker", 16,
         overrides={"--gate-open": "0.0"}, hypothesis="All gates closed: cold start"),
    spec("bep_gate005_marker", "bep", "marker", 16,
         overrides={"--gate-open": "0.05"}, hypothesis="Default small-open gates"),
    spec("bep_gate050_marker", "bep", "marker", 16,
         overrides={"--gate-open": "0.5"}, hypothesis="Half-open gates weaken identity residual"),
    spec("bep_bits8_marker", "bep", "marker", 16,
         overrides={"--bits": "8"}, hypothesis="8-bit H clips more aggressively"),
    spec("bep_bits12_marker", "bep", "marker", 16,
         overrides={"--bits": "12"}, hypothesis="12-bit H range"),
    spec("bep_bits15_marker", "bep", "marker", 16,
         overrides={"--bits": "15"}, hypothesis="Default 15-bit H range"),
    spec("bep_flipdrop01_copy", "bep", "copy", 16,
         overrides={"--flip-dropout": "0.1"}, hypothesis="10% flip noise may regularize or harm"),
    spec("bep_flipdrop02_copy", "bep", "copy", 16,
         overrides={"--flip-dropout": "0.2"}, hypothesis="Heavy flip noise should be harder"),
]

CODEBOOK_EXPERIMENTS: List[ExperimentSpec] = [
    spec("cb_random_marker", "codebook", "marker", 16,
         overrides={"--codebook-mode": "random"},
         hypothesis="Random codebook should hurt decode geometry"),
    spec("cb_bef10_marker", "codebook", "marker", 16,
         overrides={"--bef-sweeps": "10"}, hypothesis="BEF 10 sweeps is fast"),
    spec("cb_bef30_marker", "codebook", "marker", 16,
         overrides={"--bef-sweeps": "30"}, hypothesis="BEF 30 sweeps has stronger separation"),
    spec("cb_semw00_marker", "codebook", "marker", 16,
         overrides={"--sem-weight": "0.0"}, hypothesis="No semantic rerank: lexical only"),
    spec("cb_semw05_marker", "codebook", "marker", 16,
         overrides={"--sem-weight": "0.5"}, hypothesis="Default semantic rerank"),
    spec("cb_semw10_marker", "codebook", "marker", 16,
         overrides={"--sem-weight": "1.0"}, hypothesis="Equal lexical and semantic weight"),
    spec("cb_positionprobe_no_pos", "codebook", "position_probe", 16,
         overrides={"--no-position": ""},
         hypothesis="Position disabled should fail position-dependent task"),
]

ARCH_EXPERIMENTS: List[ExperimentSpec] = [
    spec("arch_D64_marker", "arch", "marker", 16,
         overrides={"--D": "64", "--d-ff": "128"}, hypothesis="D=64: minimum useful size"),
    spec("arch_D128_marker", "arch", "marker", 16,
         overrides={"--D": "128", "--d-ff": "256"}, hypothesis="D=128: default"),
    spec("arch_D256_marker", "arch", "marker", 16,
         overrides={"--D": "256", "--d-ff": "512"}, hypothesis="D=256: richer geometry"),
    spec("arch_D512_marker", "arch", "marker", 16,
         overrides={"--D": "512", "--d-ff": "1024"}, hypothesis="D=512: larger but slower"),
    spec("arch_L1_induction64", "arch", "induction", 64,
         overrides={"--layers": "1"}, hypothesis="One layer at long range"),
    spec("arch_L2_induction64", "arch", "induction", 64,
         overrides={"--layers": "2"}, hypothesis="Two layers: test depth benefit"),
    spec("arch_L3_induction64", "arch", "induction", 64,
         overrides={"--layers": "3"}, hypothesis="Three layers: diminishing returns"),
    spec("arch_L1_copy16", "arch", "copy", 16,
         overrides={"--layers": "1"}, hypothesis="One-layer copy task"),
    spec("arch_L2_copy16", "arch", "copy", 16,
         overrides={"--layers": "2"}, hypothesis="Two-layer copy task"),
]

ALL_EXPERIMENTS = (
    ABLATION_EXPERIMENTS
    + BSR_EXPERIMENTS
    + EPISODIC_EXPERIMENTS
    + HOPFIELD_EXPERIMENTS
    + BEP_EXPERIMENTS
    + CODEBOOK_EXPERIMENTS
    + ARCH_EXPERIMENTS
)

EXPERIMENTS_BY_GROUP = {g: [e for e in ALL_EXPERIMENTS if e.group == g] for g in GROUP_ORDER}

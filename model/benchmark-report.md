# Benchmark Report

> Auto-generated from `model/bench/results/`. Run `python model/bench/run_all.py` to refresh.

## Summary of Findings

<!-- BEGIN_AUTO:summary -->
### Component Necessity (from ablations)
- **BSR**: OPTIONAL - marker acc under ablation: 1.000
- **EpisodicSlotMemory**: REQUIRED - marker acc under ablation: 0.501
- **HopfieldBank**: OPTIONAL - marker acc under ablation: 1.000

### Minimum Useful Dimension
- D=128 is the best completed marker dimension (acc 1.000).

### Position Codes
- position_probe acc WITH positions: not run | WITHOUT: 1.000
- Positions optional for position-dependent tasks.

### Best BEP Config
- r=0.1, gate_open=0.05, bits=12 (acc 1.000).

### Best BSR Decay
- Decay palette 1,1,1,1,1 (acc 1.000).

### Recommended epi_read_k
- k=1 (acc 1.000).
<!-- END_AUTO:summary -->

---

## Group: Component Ablations

<!-- BEGIN_AUTO:ablation -->
| Experiment | Task | seq_len | Acc | Loss | Hypothesis | Verdict |
|---|---|---:|---:|---|---|---|
| abl_baseline_copy | copy | 16 | 1.000 | 0.454 | Full model should reach >=80% on copy | PASS |
| abl_baseline_marker | marker | 16 | 1.000 | 0.131 | Full model should reach >=90% on marker | PASS |
| abl_no_bsr_copy | copy | 16 | 1.000 | 0.474 | Without BSR, copy may still work via episodic recall | PASS |
| abl_no_bsr_marker | marker | 16 | 1.000 | 0.009 | Without BSR, marker should still work via episodic recall | PASS |
| abl_no_episodic_copy | copy | 16 | 0.139 | 2.123 | Without episodic recall, copy should degrade | FAIL |
| abl_no_episodic_marker | marker | 16 | 0.501 | 4.456 | Without episodic recall, marker should degrade significantly | FAIL |
| abl_no_hopfield_copy | copy | 16 | 1.000 | 0.403 | Without Hopfield, copy should be unaffected | PASS |
| abl_no_hopfield_marker | marker | 16 | 1.000 | 0.229 | Without Hopfield, marker should still work | PASS |
<!-- END_AUTO:ablation -->

---

## Group: BSR Sweep

<!-- BEGIN_AUTO:bsr -->
| Experiment | Task | seq_len | Acc | Loss | Hypothesis | Verdict |
|---|---|---:|---:|---|---|---|
| bsr_decay_fixed1 | bsr_long | 24 | 1.000 | 0.001 | Single-timescale decay should be worse at long gaps | PASS |
| bsr_decay_fixed4 | bsr_long | 24 | 1.000 | 0.001 | Fast decay should lose early context | PASS |
| bsr_decay_multiscale | bsr_long | 24 | 1.000 | 0.001 | Default multi-scale palette should be best | PASS |
| bsr_decay_permanent | bsr_long | 24 | 1.000 | 0.002 | Permanent accumulation may overflow with noise | PASS |
| bsr_long_gap16 | bsr_long | 24 | 1.000 | 0.001 | Medium gap: BSR decay matters | PASS |
| bsr_long_gap32 | bsr_long | 40 | 1.000 | 0.002 | Long gap: multi-scale decay needed | PASS |
| bsr_long_gap8 | bsr_long | 16 | 1.000 | 0.001 | Short gap: BSR should carry easily | PASS |
| bsr_nolayers_copy_distracted | copy_distracted | 32 | 1.000 | 0.771 | Without BSR, episodic alone handles distracted copy | PASS |
| bsr_with_episodic_copy_distracted | copy_distracted | 32 | 1.000 | 0.775 | With BSR and episodic, cooperation should improve copy | PASS |
<!-- END_AUTO:bsr -->

---

## Group: Episodic Sweep

<!-- BEGIN_AUTO:episodic -->
| Experiment | Task | seq_len | Acc | Loss | Hypothesis | Verdict |
|---|---|---:|---:|---|---|---|
| epi_no_margin_sup_marker | marker | 16 | 0.499 | 3.741 | Without address-margin supervision, convergence should slow | FAIL |
| epi_no_position_distracted | epi_distracted | 32 | 1.000 | 0.003 | Content-only scoring should work if content is unique | PASS |
| epi_read_k1_induction64 | induction | 64 | 1.000 | 0.123 | Long-range induction with exact read | PASS |
| epi_read_k1_marker16 | marker | 16 | 1.000 | 0.131 | k=1 exact read should work | PASS |
| epi_read_k2_marker16 | marker | 16 | 1.000 | 0.055 | k=2 should be no worse or slightly better | PASS |
| epi_read_k4_induction64 | induction | 64 | 1.000 | 0.011 | Bundled read for induction may help or add noise | PASS |
| epi_read_k4_marker16 | marker | 16 | 1.000 | 0.090 | k=4 averaging may hurt exact recall | PASS |
| epi_window_full_marker32 | marker | 32 | 1.000 | 0.008 | Full window keeps antecedent reachable | PASS |
| epi_window_small_marker32 | marker | 32 | 0.504 | 4.186 | Small window cannot see the antecedent | FAIL |
| epi_with_position_distracted | epi_distracted | 32 | 1.000 | 0.004 | Content and position should improve disambiguation | PASS |
<!-- END_AUTO:episodic -->

---

## Group: Hopfield Sweep

<!-- BEGIN_AUTO:hopfield -->
| Experiment | Task | seq_len | Acc | Loss | Hypothesis | Verdict |
|---|---|---:|---:|---|---|---|
| hop_no_hopfield_prior_task | hopfield_prior | 16 | 1.000 | 0.138 | Without Hopfield, prior task should fail | PASS |
| hop_prior_task_large | hopfield_prior | 16 | 1.000 | 0.192 | Large Hopfield bank should learn prior faster | PASS |
| hop_prior_task_small | hopfield_prior | 16 | 1.000 | 0.308 | Small Hopfield bank should learn four prior tokens | PASS |
| hop_slots16_marker | marker | 16 | 1.000 | 0.094 | Small bank should be adequate | PASS |
| hop_slots256_marker | marker | 16 | 1.000 | 0.006 | Large bank is overparameterized | PASS |
| hop_slots64_marker | marker | 16 | 1.000 | 0.004 | Medium bank should match baseline | PASS |
| hop_topk15_marker | marker | 16 | 1.000 | 0.187 | Wide WTA may add superposition noise | PASS |
| hop_topk1_marker | marker | 16 | 1.000 | 0.025 | Hard top-1 is exact but brittle | PASS |
| hop_topk3_marker | marker | 16 | 1.000 | 0.131 | Default top-k | PASS |
<!-- END_AUTO:hopfield -->

---

## Group: BEP Hyperparameters

<!-- BEGIN_AUTO:bep -->
| Experiment | Task | seq_len | Acc | Loss | Hypothesis | Verdict |
|---|---|---:|---:|---|---|---|
| bep_bits12_marker | marker | 16 | 1.000 | 0.131 | 12-bit H range | PASS |
| bep_bits15_marker | marker | 16 | 1.000 | 0.131 | Default 15-bit H range | PASS |
| bep_bits8_marker | marker | 16 | 1.000 | 0.057 | 8-bit H clips more aggressively | PASS |
| bep_flipdrop01_copy | copy | 16 | 0.989 | 0.250 | 10% flip noise may regularize or harm | PASS |
| bep_flipdrop02_copy | copy | 16 | 0.999 | 0.094 | Heavy flip noise should be harder | PASS |
| bep_gate000_marker | marker | 16 | 1.000 | 0.008 | All gates closed: cold start | PASS |
| bep_gate005_marker | marker | 16 | 1.000 | 0.131 | Default small-open gates | PASS |
| bep_gate050_marker | marker | 16 | 1.000 | 0.026 | Half-open gates weaken identity residual | PASS |
| bep_r005_marker | marker | 16 | 1.000 | 0.340 | Tight margin: slow but precise | PASS |
| bep_r010_marker | marker | 16 | 1.000 | 0.131 | Default margin | PASS |
| bep_r020_marker | marker | 16 | 1.000 | 0.131 | Loose margin may add updates | PASS |
| bep_r040_marker | marker | 16 | 1.000 | 0.005 | Very loose margin may destabilize | PASS |
<!-- END_AUTO:bep -->

---

## Group: Codebook

<!-- BEGIN_AUTO:codebook -->
| Experiment | Task | seq_len | Acc | Loss | Hypothesis | Verdict |
|---|---|---:|---:|---|---|---|
| cb_bef10_marker | marker | 16 | 1.000 | 0.131 | BEF 10 sweeps is fast | PASS |
| cb_bef30_marker | marker | 16 | 1.000 | 0.001 | BEF 30 sweeps has stronger separation | PASS |
| cb_positionprobe_no_pos | position_probe | 16 | 1.000 | 0.016 | Position disabled should fail position-dependent task | PASS |
| cb_random_marker | marker | 16 | 1.000 | 0.300 | Random codebook should hurt decode geometry | PASS |
| cb_semw00_marker | marker | 16 | 1.000 | 0.120 | No semantic rerank: lexical only | PASS |
| cb_semw05_marker | marker | 16 | 1.000 | 0.131 | Default semantic rerank | PASS |
| cb_semw10_marker | marker | 16 | 1.000 | 0.111 | Equal lexical and semantic weight | PASS |
<!-- END_AUTO:codebook -->

---

## Group: Architecture (D and Depth)

<!-- BEGIN_AUTO:arch -->
| Experiment | Task | seq_len | Acc | Loss | Hypothesis | Verdict |
|---|---|---:|---:|---|---|---|
| arch_D128_marker | marker | 16 | 1.000 | 0.131 | D=128: default | PASS |
| arch_D256_marker | marker | 16 | 1.000 | 0.032 | D=256: richer geometry | PASS |
| arch_D512_marker | marker | 16 | 1.000 | 0.000 | D=512: larger but slower | PASS |
| arch_D64_marker | marker | 16 | 1.000 | 0.232 | D=64: minimum useful size | PASS |
| arch_L1_copy16 | copy | 16 | 1.000 | 0.454 | One-layer copy task | PASS |
| arch_L1_induction64 | induction | 64 | 1.000 | 0.123 | One layer at long range | PASS |
| arch_L2_copy16 | copy | 16 | 1.000 | 0.238 | Two-layer copy task | PASS |
| arch_L2_induction64 | induction | 64 | 1.000 | 0.065 | Two layers: test depth benefit | PASS |
| arch_L3_induction64 | induction | 64 | 1.000 | 0.017 | Three layers: diminishing returns | PASS |
<!-- END_AUTO:arch -->

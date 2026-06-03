# Benchmark Report

> Auto-generated from `model/bench/results/`. Run `python model/bench/run_all.py` to refresh.

## Summary of Findings

<!-- BEGIN_AUTO:summary -->
### Component Necessity (from ablations)
- **BSR**: OPTIONAL - marker acc under ablation: 1.000
- **EpisodicSlotMemory**: REQUIRED - marker acc under ablation: 0.499
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
| abl_baseline_copy | copy | 16 | 1.000 | 0.124 | Full model should reach >=80% on copy | PASS |
| abl_baseline_marker | marker | 16 | 1.000 | 0.000 | Full model should reach >=90% on marker | PASS |
| abl_no_bsr_copy | copy | 16 | 1.000 | 0.094 | Without BSR, copy may still work via episodic recall | PASS |
| abl_no_bsr_marker | marker | 16 | 1.000 | 0.000 | Without BSR, marker should still work via episodic recall | PASS |
| abl_no_episodic_copy | copy | 16 | 0.126 | 3.287 | Without episodic recall, copy should degrade | FAIL |
| abl_no_episodic_marker | marker | 16 | 0.499 | 8.900 | Without episodic recall, marker should degrade significantly | FAIL |
| abl_no_hopfield_copy | copy | 16 | 1.000 | 0.099 | Without Hopfield, copy should be unaffected | PASS |
| abl_no_hopfield_marker | marker | 16 | 1.000 | 0.000 | Without Hopfield, marker should still work | PASS |
<!-- END_AUTO:ablation -->

---

## Group: BSR Sweep

<!-- BEGIN_AUTO:bsr -->
| Experiment | Task | seq_len | Acc | Loss | Hypothesis | Verdict |
|---|---|---:|---:|---|---|---|
| bsr_decay_fixed1 | bsr_long | 24 | 1.000 | 0.000 | Single-timescale decay should be worse at long gaps | PASS |
| bsr_decay_fixed4 | bsr_long | 24 | 1.000 | 0.000 | Fast decay should lose early context | PASS |
| bsr_decay_multiscale | bsr_long | 24 | 1.000 | 0.000 | Default multi-scale palette should be best | PASS |
| bsr_decay_permanent | bsr_long | 24 | 1.000 | 0.000 | Permanent accumulation may overflow with noise | PASS |
| bsr_long_gap16 | bsr_long | 24 | 1.000 | 0.000 | Medium gap: BSR decay matters | PASS |
| bsr_long_gap32 | bsr_long | 40 | 1.000 | 0.000 | Long gap: multi-scale decay needed | PASS |
| bsr_long_gap8 | bsr_long | 16 | 1.000 | 0.070 | Short gap: BSR should carry easily | PASS |
| bsr_nolayers_copy_distracted | copy_distracted | 32 | 0.184 | 2.408 | Without BSR, episodic alone handles distracted copy | FAIL |
| bsr_with_episodic_copy_distracted | copy_distracted | 32 | 0.118 | 3.093 | With BSR and episodic, cooperation should improve copy | FAIL |
<!-- END_AUTO:bsr -->

---

## Group: Episodic Sweep

<!-- BEGIN_AUTO:episodic -->
| Experiment | Task | seq_len | Acc | Loss | Hypothesis | Verdict |
|---|---|---:|---:|---|---|---|
| epi_no_margin_sup_marker | marker | 16 | 0.501 | 8.445 | Without address-margin supervision, convergence should slow | FAIL |
| epi_no_position_distracted | epi_distracted | 32 | 0.984 | 0.095 | Content-only scoring should work if content is unique | PASS |
| epi_read_k1_induction64 | induction | 64 | 1.000 | 0.001 | Long-range induction with exact read | PASS |
| epi_read_k1_marker16 | marker | 16 | 1.000 | 0.000 | k=1 exact read should work | PASS |
| epi_read_k2_marker16 | marker | 16 | 1.000 | 0.000 | k=2 should be no worse or slightly better | PASS |
| epi_read_k4_induction64 | induction | 64 | 1.000 | 0.002 | Bundled read for induction may help or add noise | PASS |
| epi_read_k4_marker16 | marker | 16 | 0.501 | 7.270 | k=4 averaging may hurt exact recall | FAIL |
| epi_window_full_marker32 | marker | 32 | 1.000 | 0.000 | Full window keeps antecedent reachable | PASS |
| epi_window_small_marker32 | marker | 32 | 0.504 | 8.421 | Small window cannot see the antecedent | FAIL |
| epi_with_position_distracted | epi_distracted | 32 | 1.000 | 0.000 | Content and position should improve disambiguation | PASS |
<!-- END_AUTO:episodic -->

---

## Group: Hopfield Sweep

<!-- BEGIN_AUTO:hopfield -->
| Experiment | Task | seq_len | Acc | Loss | Hypothesis | Verdict |
|---|---|---:|---:|---|---|---|
| hop_no_hopfield_prior_task | hopfield_prior | 16 | 1.000 | 0.049 | Without Hopfield, prior task should fail | PASS |
| hop_prior_task_large | hopfield_prior | 16 | 1.000 | 0.311 | Large Hopfield bank should learn prior faster | PASS |
| hop_prior_task_small | hopfield_prior | 16 | 1.000 | 0.200 | Small Hopfield bank should learn four prior tokens | PASS |
| hop_slots16_marker | marker | 16 | 1.000 | 0.000 | Small bank should be adequate | PASS |
| hop_slots256_marker | marker | 16 | 1.000 | 0.000 | Large bank is overparameterized | PASS |
| hop_slots64_marker | marker | 16 | 1.000 | 0.000 | Medium bank should match baseline | PASS |
| hop_topk15_marker | marker | 16 | 1.000 | 0.000 | Wide WTA may add superposition noise | PASS |
| hop_topk1_marker | marker | 16 | 1.000 | 0.000 | Hard top-1 is exact but brittle | PASS |
| hop_topk3_marker | marker | 16 | 1.000 | 0.000 | Default top-k | PASS |
<!-- END_AUTO:hopfield -->

---

## Group: BEP Hyperparameters

<!-- BEGIN_AUTO:bep -->
| Experiment | Task | seq_len | Acc | Loss | Hypothesis | Verdict |
|---|---|---:|---:|---|---|---|
| bep_bits12_marker | marker | 16 | 1.000 | 0.000 | 12-bit H range | PASS |
| bep_bits15_marker | marker | 16 | 1.000 | 0.000 | Default 15-bit H range | PASS |
| bep_bits8_marker | marker | 16 | 1.000 | 0.000 | 8-bit H clips more aggressively | PASS |
| bep_flipdrop01_copy | copy | 16 | 0.916 | 0.598 | 10% flip noise may regularize or harm | PASS |
| bep_flipdrop02_copy | copy | 16 | 0.691 | 1.134 | Heavy flip noise should be harder | FAIL |
| bep_gate000_marker | marker | 16 | 1.000 | 0.000 | All gates closed: cold start | PASS |
| bep_gate005_marker | marker | 16 | 1.000 | 0.000 | Default small-open gates | PASS |
| bep_gate050_marker | marker | 16 | 1.000 | 0.681 | Half-open gates weaken identity residual | PASS |
| bep_r005_marker | marker | 16 | 1.000 | 0.000 | Tight margin: slow but precise | PASS |
| bep_r010_marker | marker | 16 | 1.000 | 0.000 | Default margin | PASS |
| bep_r020_marker | marker | 16 | 1.000 | 0.000 | Loose margin may add updates | PASS |
| bep_r040_marker | marker | 16 | 1.000 | 0.000 | Very loose margin may destabilize | PASS |
<!-- END_AUTO:bep -->

---

## Group: Codebook

<!-- BEGIN_AUTO:codebook -->
| Experiment | Task | seq_len | Acc | Loss | Hypothesis | Verdict |
|---|---|---:|---:|---|---|---|
| cb_bef10_marker | marker | 16 | 1.000 | 0.000 | BEF 10 sweeps is fast | PASS |
| cb_bef30_marker | marker | 16 | 1.000 | 0.000 | BEF 30 sweeps has stronger separation | PASS |
| cb_positionprobe_no_pos | position_probe | 16 | 1.000 | 0.000 | Position disabled should fail position-dependent task | PASS |
| cb_random_marker | marker | 16 | 1.000 | 0.000 | Random codebook should hurt decode geometry | PASS |
| cb_semw00_marker | marker | 16 | 1.000 | 0.000 | No semantic rerank: lexical only | PASS |
| cb_semw05_marker | marker | 16 | 1.000 | 0.000 | Default semantic rerank | PASS |
| cb_semw10_marker | marker | 16 | 1.000 | 0.000 | Equal lexical and semantic weight | PASS |
<!-- END_AUTO:codebook -->

---

## Group: Architecture (D and Depth)

<!-- BEGIN_AUTO:arch -->
| Experiment | Task | seq_len | Acc | Loss | Hypothesis | Verdict |
|---|---|---:|---:|---|---|---|
| arch_D128_marker | marker | 16 | 1.000 | 0.000 | D=128: default | PASS |
| arch_D256_marker | marker | 16 | 1.000 | 0.000 | D=256: richer geometry | PASS |
| arch_D512_marker | marker | 16 | 0.000 | inf | ERROR: TimeoutExpired: Command '['/Users/theocoombes/Documents/Programming Projects/Binary LLM/Implementation/.venv/bin/python', '/Users/theocoombes/Documents/Programming Projects/Binary LLM/Implementation/model/probe_synthetic.py', '--tasks', 'marker', '--lengths', '16', '--steps', '300', '--json-out', '/Users/theocoombes/Documents/Programming Projects/Binary LLM/Implementation/model/bench/results/.arch_D512_marker_97202_20260603T221724.tmp.json', '--D', '512', '--layers', '1', '--d-ff', '1024', '--slots', '32', '--top-k', '3', '--r', '0.1', '--gate-open', '0.05', '--seed', '0', '--batch-size', '64', '--codebook-mode', 'structured', '--bef-sweeps', '10', '--sem-weight', '0.5']' timed out after 90 seconds | ERROR |
| arch_D64_marker | marker | 16 | 0.590 | 4.717 | D=64: minimum useful size | FAIL |
| arch_L1_copy16 | copy | 16 | 1.000 | 0.124 | One-layer copy task | PASS |
| arch_L1_induction64 | induction | 64 | 1.000 | 0.001 | One layer at long range | PASS |
| arch_L2_copy16 | copy | 16 | 1.000 | 0.337 | Two-layer copy task | PASS |
| arch_L2_induction64 | induction | 64 | 1.000 | 0.030 | Two layers: test depth benefit | PASS |
| arch_L3_induction64 | induction | 64 | 0.000 | inf | ERROR: TimeoutExpired: Command '['/Users/theocoombes/Documents/Programming Projects/Binary LLM/Implementation/.venv/bin/python', '/Users/theocoombes/Documents/Programming Projects/Binary LLM/Implementation/model/probe_synthetic.py', '--tasks', 'induction', '--lengths', '64', '--steps', '300', '--json-out', '/Users/theocoombes/Documents/Programming Projects/Binary LLM/Implementation/model/bench/results/.arch_L3_induction64_97199_20260603T221806.tmp.json', '--D', '128', '--layers', '3', '--d-ff', '256', '--slots', '32', '--top-k', '3', '--r', '0.1', '--gate-open', '0.05', '--seed', '0', '--batch-size', '64', '--codebook-mode', 'structured', '--bef-sweeps', '10', '--sem-weight', '0.5']' timed out after 90 seconds | ERROR |
<!-- END_AUTO:arch -->

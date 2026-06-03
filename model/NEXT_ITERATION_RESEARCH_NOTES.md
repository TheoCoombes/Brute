# Next Iteration Research Notes

This note is for the next Haemmr research pass. The codebase is cleaned around
the current architecture; the remaining work is about scaling, kernel fusion,
and stronger training evidence.

## Priority Improvements

1. Replace quadratic batched episodic training.

   `EpisodicSlotMemory.forward` still forms dense `(B, n, n)` content and
   position score tensors. The streaming path is windowed and packed, but
   training needs a blockwise, recurrent, or sampled-address objective that
   avoids all-pairs sequence scoring.

2. Fuse episodic top-1 and top-k lookup.

   The current top-1 read gathers packed payload rows, but scoring is still
   composed from separate packed matmuls and dense masks. A fused causal
   Hamming-search kernel should accept packed queries, keys, positions, and
   payloads, then return packed reads plus compact training metadata.

3. Finish packed vote kernels.

   Multi-slot Hopfield and episodic reads still materialise selected payloads as
   real/integer tallies before thresholding. Add packed majority or bit-sliced
   vote kernels for top-k readout, including deterministic tie handling.

4. Make BSR backward match the packed forward more closely.

   BSR forward now uses an integer shift-decay scan. Backward still uses a
   real-valued reverse-scan surrogate with the float decay palette. Decide
   whether that surrogate is acceptable, or implement an integer-aware local
   surrogate that better matches the forward dynamics.

5. Add CUDA validation.

   CPU and MPS paths are locally tested. CUDA implementations for `pack_sign`
   and `bsr_scan` are present but need compile/runtime validation on a CUDA
   host, including tail dimensions and nontrivial decay palettes.

6. Improve codebook scaling.

   The inline BEF initializer is intentionally capped. For serious vocabularies,
   precompute/load codebooks, add nearest-neighbor or shortlist decode, and
   benchmark full-vocab decode versus shortlist rerank.

7. Establish training benchmarks that are not historical comparisons.

   Keep benchmarks focused on current Haemmr capabilities: synthetic recall,
   induction, WikiText compact-vocab loss, throughput, memory, and ablations of
   current mechanisms. Store generated outputs outside the source tree.

8. Add streaming generation over model state.

   `generate` still recomputes the full context each token. Wire block-level
   streaming BSR and episodic state into an incremental inference path, then
   verify parity against full-context forward on fixed prompts.

9. Tighten BOLD optimiser diagnostics.

   Track per-parameter flip rates, accumulator saturation, boundary-hit rates,
   and codebook drift. Use those signals to tune `threshold`, `eta`, boundary
   gating, and codebook flip scale instead of relying only on loss curves.

10. Broaden device-aware performance tests.

   Add benchmark cases for `pack_sign`, `bsr_scan`, episodic streaming reads,
   and decode. Report CPU/MPS/CUDA separately and include packed-buffer memory
   footprint, not just tokens per second.

## Current Verification Baseline

The cleanup pass validated:

- `tests/unit/test_packed_kernels.py` on local CPU/MPS availability;
- full `model/tests`;
- full `tests/unit`;
- an explicit MPS Haemmr forward smoke test.

Future work should keep these passing and add targeted tests before changing
recurrence semantics or memory layout.

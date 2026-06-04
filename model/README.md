# HÆMMR Model

This directory contains the current packed-bit reference implementation of HÆMMR.
The model is not a standard float transformer. The hot path stays on `brute.bit1`
tensors, the recurrent state is explicit, and training uses BEP/BOLD-style
integer hidden weights (`bep.py`) instead of float gradients.

## Core Data Flow

```text
token ids
  -> TokenCodebook.embed()
  -> input DiagBind
  -> [ BSR
       EpisodicSlotMemory
       HopfieldBank
       ChannelMix ] x L
  -> out DiagBind
  -> lex_proj
  -> optional semantic DiagBind + semantic decode
  -> fixed codebook decode
  -> logits
```

Positions are only used in the episodic address lane. The decoded concept stream
stays position-free. If `use_position=False`, the episodic lane receives a
neutral all-ones code instead of hierarchical position codes.

## Main Components

| Component | File | Role |
|---|---|---|
| `TokenCodebook` | [`layers.py`](./layers.py) | Fixed token prototypes plus min-Hamming decode. Supports inline BEF or offline GPT-2 SimHash initialisation. |
| `DiagBind` | [`layers.py`](./layers.py) | Learned diagonal bitwise binding masks for input, output, and semantic decode lanes. |
| `BooleanLinear` | [`layers.py`](./layers.py) | Packed XNOR/popcount linear projection with BEP backward wiring. |
| `ResidualMerge` | [`layers.py`](./layers.py) | Binary gated skip/transform merge. |
| `BSR` | [`layers.py`](./layers.py) | Delta-corrected recurrent bundle with a small multi-timescale decay palette. |
| `EpisodicSlotMemory` | [`layers.py`](./layers.py) | Exact in-window causal recall using content + position scoring and optional local margin supervision. |
| `HopfieldBank` | [`layers.py`](./layers.py) | Static learned key/value prior bank with top-k winner-take-all readout. |
| `ChannelMix` | [`model.py`](./model.py) | Two-stage binary MLP used as the per-block channel mixer. |
| `HaemmrLM` | [`model.py`](./model.py) | Full stack assembly, loss/backward path, generation, and checkpointing. |
| `BepParam` / `BepOptimizer` | [`bep.py`](./bep.py) | Integer hidden weights, visible `sign(H)` weights, and bit-flip stepping. |

The model also has an optional cleanup Hopfield pass in `HaemmrConfig`
(`hopfield_cleanup=True`) for experiments, but the standard CLI leaves it off.

## Training

`train.py` is the main training entry point.

The backward pass is BEP:

* each parameter stores an integer hidden weight buffer `H` (`int16`);
* the visible weight is `sign(H)`;
* the logging loss is a normal NLL/perplexity calculation only;
* the raw decode margin is tracked as
  (`logit[target] - max_other < r_eff * D`);
* output-codebook/readout updates can run on all raw violations, while hidden
  BEP updates can be warmed up, annealed, and capped to avoid whole-batch
  rewrites after early readout adaptation;
* triggered positions emit binary desired activations, not float gradients;
* episodic address projections can also receive local matched-slot supervision.

Useful flags on the current CLI:

* `--D`, `--layers`, `--d-ff`
* `--slots`, `--top-k`
* `--epi-slots`, `--epi-read-k`
* `--no-position`
* `--codebook-mode offline|structured|random`
* `--bef-sweeps`
* `--sem-weight`
* `--r`, `--p-r`, `--bits`, `--flip-dropout`
* `--readout-warmup-steps`, `--margin-r-final`, `--margin-anneal-steps`
* `--max-trigger-rate`
* `--gate-open`, `--boundary-nu`

`--codebook-mode structured` is the default for WikiText training and probing.
`offline` builds a GPT-2 SimHash codebook from pretrained embeddings and caches
it under `model/.data/codebook`.

`--codebook-mode structured` keeps the inline BEF initializer.
`--codebook-mode random` disables the structured initializer and uses a random
codebook instead.

The synthetic probe (`probe_synthetic.py`) uses `structured` by default and
supports `random` for ablations.

Example:

```bash
./.venv/bin/python model/train.py \
  --steps 1500 \
  --D 1024 \
  --layers 2 \
  --seq-len 64 \
  --r 0.1 \
  --bits 15
```

## Data And Codebooks

[`data.py`](./data.py) loads WikiText with the full GPT-2 tokenizer and returns
raw GPT-2 ids plus the `compact_to_gpt2` mapping used for decode and
checkpoint compatibility.

[`codebook.py`](./codebook.py) can build an offline GPT-2 SimHash codebook from
pretrained token embeddings. That is the default codebook mode for WikiText
training and is cached under `model/.data/codebook` so the expensive download
only happens on the first build.

## Entry Points

* [`train.py`](./train.py): main WikiText training CLI.
* [`sample.py`](./sample.py): sample from a checkpoint.
* [`probe_synthetic.py`](./probe_synthetic.py): small synthetic induction/copy/retrieval probes.
* [`probe_wikitext.py`](./probe_wikitext.py): compare content-only and positioned WikiText probes.
* [`benchmark_matrix.py`](./benchmark_matrix.py): geometry, component, synthetic, and efficiency matrix.
* [`bench/run_all.py`](./bench/run_all.py): run benchmark groups and refresh `benchmark-report.md`.

## Reports

`benchmark-report.md` is the current auto-generated report. It is written from
`model/bench/results/` by `bench/run_all.py`.

`benchmark_report.md` is the historical v1 matrix report retained for context.
It is no longer the active report target.

## Testing

Run the package tests with:

```bash
./.venv/bin/python -m pytest model/tests -q
```

The tests cover:

* VSA binding, unbinding, bundling, positions, and BEF initialisation
* packed Boolean layers and BEP backward wiring
* BSR recurrence and streaming parity
* episodic slot causality, windowing, and margin supervision
* full-model forward/backward, sampling, and checkpoint round-tripping
* source-level efficiency contracts that guard against old packed-path regressions

## Where To Start When Changing The Model

If you are changing the core architecture, start with:

1. [`vsa.py`](./vsa.py)
2. [`bep.py`](./bep.py)
3. [`layers.py`](./layers.py)
4. [`model.py`](./model.py)

That is the shortest path to understanding how the packed representation,
memory blocks, and update rule fit together.

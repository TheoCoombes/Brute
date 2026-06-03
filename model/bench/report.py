"""Generate the canonical `benchmark-report.md` from `bench/results/` JSON."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .experiments import GROUP_ORDER
from .runner import ExperimentResult, load_results


REPORT_MD = Path(__file__).resolve().parent.parent / "benchmark-report.md"

GROUP_TITLES = {
    "ablation": "Component Ablations",
    "bsr": "BSR Sweep",
    "episodic": "Episodic Sweep",
    "hopfield": "Hopfield Sweep",
    "bep": "BEP Hyperparameters",
    "codebook": "Codebook",
    "arch": "Architecture (D and Depth)",
}


def _pass_fail(acc: float, threshold: float = 0.80) -> str:
    return "PASS" if acc >= threshold else "FAIL"


def _fmt_float(x: float) -> str:
    if x == float("inf"):
        return "inf"
    return f"{x:.3f}"


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _normalize_legacy_codebook_error(text: str) -> str:
    """Rewrite old codebook flags in timeout command strings to the new mode flag."""
    legacy_flags = {
        "--offline-codebook",
        "--no-offline-codebook",
        "--structured-codebook",
        "--no-structured-codebook",
    }
    if not any(flag in text for flag in legacy_flags):
        return text

    match = re.search(r"Command '(?P<cmd>\[.*?\])' timed out", text)
    if match is None:
        return text

    try:
        cmd = ast.literal_eval(match.group("cmd"))
    except Exception:
        return text
    if not isinstance(cmd, list) or not all(isinstance(item, str) for item in cmd):
        return text

    mode = None
    insert_at = None
    cleaned: List[str] = []
    for token in cmd:
        if token in legacy_flags:
            if insert_at is None:
                insert_at = len(cleaned)
            if token == "--offline-codebook":
                mode = "offline"
            elif token == "--no-offline-codebook":
                mode = "structured"
            elif token == "--structured-codebook":
                mode = "structured"
            elif token == "--no-structured-codebook":
                mode = "random"
            continue
        cleaned.append(token)

    if mode is None:
        return text
    if insert_at is None:
        insert_at = len(cleaned)
    cleaned[insert_at:insert_at] = ["--codebook-mode", mode]
    return text[:match.start("cmd")] + repr(cleaned) + text[match.end("cmd"):]


def load_results_by_name(results: Optional[Iterable[ExperimentResult]] = None) -> Dict[str, ExperimentResult]:
    by_name: Dict[str, ExperimentResult] = {}
    for result in results if results is not None else load_results():
        old = by_name.get(result.name)
        if old is None or result.timestamp >= old.timestamp:
            by_name[result.name] = result
    return by_name


def format_group_table(results: List[ExperimentResult], group: str) -> str:
    group_results = sorted(
        [r for r in results if r.group == group],
        key=lambda r: (r.name, r.timestamp),
    )
    latest = load_results_by_name(group_results).values()
    rows = sorted(latest, key=lambda r: r.name)
    if not rows:
        return "_No completed results yet._"
    lines = [
        "| Experiment | Task | seq_len | Acc | Loss | Hypothesis | Verdict |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for r in rows:
        verdict = "ERROR" if r.error else _pass_fail(r.acc)
        hyp = f"ERROR: {_normalize_legacy_codebook_error(r.error)}" if r.error else r.hypothesis
        lines.append(
            f"| {_cell(r.name)} | {_cell(r.task)} | {r.seq_len} | "
            f"{_fmt_float(r.acc)} | {_fmt_float(r.loss)} | {_cell(hyp)} | {verdict} |"
        )
    return "\n".join(lines)


def _get(by_name: Dict[str, ExperimentResult], name: str) -> Optional[ExperimentResult]:
    return by_name.get(name)


def _acc(by_name: Dict[str, ExperimentResult], name: str) -> str:
    result = _get(by_name, name)
    return "not run" if result is None else _fmt_float(result.acc)


def _best(results: Iterable[ExperimentResult]) -> Optional[ExperimentResult]:
    ok = [r for r in results if r.error is None]
    return max(ok, key=lambda r: r.acc, default=None)


def _flag(result: ExperimentResult, flag: str, default: str = "default") -> str:
    return str(result.cfg_overrides.get(flag, default))


def format_summary(results_by_name: Dict[str, ExperimentResult]) -> str:
    no_bsr = _get(results_by_name, "abl_no_bsr_marker")
    no_epi = _get(results_by_name, "abl_no_episodic_marker")
    no_hop = _get(results_by_name, "abl_no_hopfield_marker")

    arch_best = _best(
        r for r in results_by_name.values()
        if r.name.startswith("arch_D") and r.task == "marker"
    )
    bep_best = _best(r for r in results_by_name.values() if r.group == "bep")
    bsr_best = _best(r for r in results_by_name.values() if r.name.startswith("bsr_decay_"))
    epi_best = _best(
        r for r in results_by_name.values()
        if r.name.startswith("epi_read_k") and r.task == "marker"
    )

    def component_line(label: str, result: Optional[ExperimentResult]) -> str:
        if result is None:
            return f"- **{label}**: not run"
        status = "OPTIONAL" if result.acc >= 0.80 else "REQUIRED"
        return f"- **{label}**: {status} - marker acc under ablation: {_fmt_float(result.acc)}"

    if arch_best is None:
        dim_line = "- No dimension sweep results yet."
    else:
        dim_line = (
            f"- D={_flag(arch_best, '--D')} is the best completed marker dimension "
            f"(acc {_fmt_float(arch_best.acc)})."
        )

    if bep_best is None:
        bep_line = "- No BEP sweep results yet."
    else:
        bep_line = (
            f"- r={_flag(bep_best, '--r')}, gate_open={_flag(bep_best, '--gate-open')}, "
            f"bits={_flag(bep_best, '--bits', '15')} (acc {_fmt_float(bep_best.acc)})."
        )

    if bsr_best is None:
        bsr_line = "- No BSR decay sweep results yet."
    else:
        bsr_line = f"- Decay palette {_flag(bsr_best, '--decay-shifts')} (acc {_fmt_float(bsr_best.acc)})."

    if epi_best is None:
        epi_line = "- No epi_read_k sweep results yet."
    else:
        epi_line = f"- k={_flag(epi_best, '--epi-read-k')} (acc {_fmt_float(epi_best.acc)})."

    return "\n".join([
        "### Component Necessity (from ablations)",
        component_line("BSR", no_bsr),
        component_line("EpisodicSlotMemory", no_epi),
        component_line("HopfieldBank", no_hop),
        "",
        "### Minimum Useful Dimension",
        dim_line,
        "",
        "### Position Codes",
        f"- position_probe acc WITH positions: not run | WITHOUT: {_acc(results_by_name, 'cb_positionprobe_no_pos')}",
        "- Positions optional for position-dependent tasks.",
        "",
        "### Best BEP Config",
        bep_line,
        "",
        "### Best BSR Decay",
        bsr_line,
        "",
        "### Recommended epi_read_k",
        epi_line,
    ])


def render_report(results: List[ExperimentResult]) -> str:
    by_name = load_results_by_name(results)
    sections = [
        "# Benchmark Report",
        "",
        "> Auto-generated from `model/bench/results/`. Run `python model/bench/run_all.py` to refresh.",
        "",
        "## Summary of Findings",
        "",
        "<!-- BEGIN_AUTO:summary -->",
        format_summary(by_name),
        "<!-- END_AUTO:summary -->",
    ]
    for group in GROUP_ORDER:
        sections.extend([
            "",
            "---",
            "",
            f"## Group: {GROUP_TITLES[group]}",
            "",
            f"<!-- BEGIN_AUTO:{group} -->",
            format_group_table(results, group),
            f"<!-- END_AUTO:{group} -->",
        ])
    return "\n".join(sections).rstrip() + "\n"


def update_benchmark_report(results: List[ExperimentResult]) -> None:
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(render_report(results))

"""Reads JSON results from bench/results/, formats and writes model/AGENTS.md."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .experiments import GROUP_ORDER
from .runner import ExperimentResult, load_results


AGENTS_MD = Path(__file__).resolve().parent.parent / "AGENTS.md"

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
        "|---|---|---:|---:|---:|---|---|",
    ]
    for r in rows:
        verdict = "ERROR" if r.error else _pass_fail(r.acc)
        hyp = f"ERROR: {r.error}" if r.error else r.hypothesis
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
    bsr_best = _best(
        r for r in results_by_name.values()
        if r.name.startswith("bsr_decay_")
    )
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
        bsr_line = (
            f"- Decay palette {_flag(bsr_best, '--decay-shifts')} "
            f"(acc {_fmt_float(bsr_best.acc)})."
        )

    if epi_best is None:
        epi_line = "- No epi_read_k sweep results yet."
    else:
        epi_line = f"- k={_flag(epi_best, '--epi-read-k')} (acc {_fmt_float(epi_best.acc)})."

    pos_no = _get(results_by_name, "cb_positionprobe_no_pos")
    pos_status = "not run"
    if pos_no is not None:
        pos_status = "REQUIRED" if pos_no.acc < 0.80 else "OPTIONAL"

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
        f"- Positions {pos_status} for position-dependent tasks.",
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


def _template() -> str:
    sections = []
    for group in GROUP_ORDER:
        sections.append(
            f"## Group: {GROUP_TITLES[group]}\n\n"
            f"<!-- BEGIN_AUTO:{group} -->\n"
            "_No completed results yet._\n"
            f"<!-- END_AUTO:{group} -->"
        )
    return (
        "# HÆMMR v2 - Architecture Validation Notes\n\n"
        "> Auto-generated sections are between `<!-- BEGIN_AUTO:group -->` and "
        "`<!-- END_AUTO:group -->` delimiters.\n"
        "> Hand-written notes outside those delimiters survive re-runs.\n\n"
        "> Last updated: never\n\n"
        "## Summary of Findings\n\n"
        "<!-- BEGIN_AUTO:summary -->\n"
        "_No completed results yet._\n"
        "<!-- END_AUTO:summary -->\n\n"
        "---\n\n"
        + "\n\n---\n\n".join(sections)
        + "\n\n---\n\n"
        "## Hand-Written Notes\n\n"
        "_Add architecture decisions and observations below - this section is never overwritten._\n"
    )


def _replace_section(text: str, key: str, body: str) -> str:
    pattern = re.compile(
        rf"(<!-- BEGIN_AUTO:{re.escape(key)} -->)(.*?)(<!-- END_AUTO:{re.escape(key)} -->)",
        re.DOTALL,
    )
    if pattern.search(text):
        return pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)
    return text.rstrip() + f"\n\n<!-- BEGIN_AUTO:{key} -->\n{body}\n<!-- END_AUTO:{key} -->\n"


def update_agents_md(results: List[ExperimentResult]) -> None:
    text = AGENTS_MD.read_text() if AGENTS_MD.exists() else _template()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if re.search(r"^> Last updated: .*$", text, flags=re.MULTILINE):
        text = re.sub(r"^> Last updated: .*$", f"> Last updated: {now}", text, flags=re.MULTILINE)
    else:
        text = text.replace(
            "> Hand-written notes outside those delimiters survive re-runs.\n",
            "> Hand-written notes outside those delimiters survive re-runs.\n"
            f"> Last updated: {now}\n",
            1,
        )
    by_name = load_results_by_name(results)
    text = _replace_section(text, "summary", format_summary(by_name))
    for group in GROUP_ORDER:
        text = _replace_section(text, group, format_group_table(results, group))
    AGENTS_MD.write_text(text)

"""Aggregate pytest-benchmark JSON output into a polished markdown report.

The report is designed for fast iteration: every (device, op) row shows
all three scales (dispatch / medium / huge) side-by-side, with the speedup
(bool / bit1) at each. The summary panel at the top calls out regressions
across the whole matrix.

Two modes
---------
Single-run mode (default)
    python -m tests.benchmarks.make_report .benchmarks/latest.json
    Renders one report from the given JSON.

Version-diff mode
    python -m tests.benchmarks.make_report base.json head.json
    Renders deltas (head vs base) for every (device, op, scale, pack):
    if bit1 got faster, the cell shows  +X.XX×; if slower, -X.XX×.

Usage
-----
    # 1. Run benchmarks
    pytest tests/benchmarks --benchmark-only \
        --benchmark-group-by=group --benchmark-sort=name \
        --benchmark-json=.benchmarks/$(date +%Y%m%d_%H%M).json

    # 2. Render report
    python -m tests.benchmarks.make_report                         # newest .json
    python -m tests.benchmarks.make_report .benchmarks/run.json    # single run
    python -m tests.benchmarks.make_report base.json head.json     # diff
    python -m tests.benchmarks.make_report -o report.md *.json     # write to file
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# JSON discovery
# ─────────────────────────────────────────────────────────────────────────────

def _latest_json(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = list(root.rglob("*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ─────────────────────────────────────────────────────────────────────────────
# Entry parsing
# ─────────────────────────────────────────────────────────────────────────────

_BIT1_RE = re.compile(r"^test_bit1(?:_(.+))?$")
_BOOL_RE = re.compile(r"^test_bool(?:_(.+))?$")
_SCALES = ("01_dispatch", "02_medium", "03_huge")
_DEVICES = ("cpu", "cuda", "mps")
_PACK_IDS = ("pw_08", "pw_32", "pw_64")


def _classify(name: str) -> tuple[Optional[str], Optional[str]]:
    """Return (kind, op) for a benchmark name (handles both naming styles).

    Old style:  test_bit1_and       → ("bit1", "and")
    New style:  test_bit1           → ("bit1", "")   (op comes from params)
    """
    core = name.split("[", 1)[0]
    if (m := _BIT1_RE.match(core)) is not None:
        return "bit1", m.group(1) or ""
    if (m := _BOOL_RE.match(core)) is not None:
        return "bool", m.group(1) or ""
    return None, None


def _parse_suffix(name: str) -> dict:
    """Pull device / scale / pack / op from the [...] suffix of a benchmark name.

    bench_catalog.py emits names like:
        test_bit1[cpu-bitwise_and-01_dispatch-pw_08]
    so we have to extract each piece.
    """
    out = {"device": "?", "scale": "?", "pack": "?", "op": ""}
    if "[" not in name:
        return out
    inside = name.split("[", 1)[1].rstrip("]")
    parts = inside.split("-")
    op_parts = []
    for tok in parts:
        if tok in _DEVICES:
            out["device"] = tok
        elif tok in _SCALES:
            out["scale"] = tok
        elif tok in _PACK_IDS:
            out["pack"] = tok
        else:
            op_parts.append(tok)
    if op_parts:
        out["op"] = "_".join(op_parts)
    return out


def _stat_seconds(entry: dict) -> float:
    s = entry["stats"]
    return float(s.get("median", s["mean"]))


def _group_key(entry: dict) -> str:
    """The benchmark.group string ('bitwise/and'). Used as the canonical op id."""
    return entry.get("group") or ""


def _format_seconds(s: float) -> str:
    if s >= 1.0:
        return f"{s:.3f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.2f} ms"
    if s >= 1e-6:
        return f"{s * 1e6:.1f} µs"
    return f"{s * 1e9:.0f} ns"


def _format_speedup(speedup: Optional[float]) -> str:
    if speedup is None:
        return "—"
    if speedup >= 1.0:
        return f"**{speedup:.2f}×**"
    return f"<u>{speedup:.2f}×</u>"


def _scale_label(scale: str) -> str:
    return scale.split("_", 1)[1] if "_" in scale else scale


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate(entries: list[dict]) -> dict:
    """Index entries by (device, group, pack, scale) → {bit1, bool, throughput}.

    `group` is the canonical op identifier (matches make_report's row label).
    """
    table: dict = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))))
    for e in entries:
        kind, _ = _classify(e["name"])
        if kind is None:
            continue
        info = _parse_suffix(e["name"])
        device = info["device"]
        scale = info["scale"]
        pack = info["pack"]
        group = _group_key(e) or info["op"]
        cell = table[device][group][pack][scale]
        cell[kind] = _stat_seconds(e)
        # Throughput annotation lives in extra_info (set via set_throughput).
        extra = e.get("extra_info") or {}
        if extra:
            cell.setdefault("throughput", extra)
    return table


def _all(table, device=None, group=None):
    """Iterate (device, group, pack, scale, cell) over the table."""
    devs = [device] if device else table.keys()
    for d in devs:
        groups = [group] if group else table[d].keys()
        for g in groups:
            for pack in table[d][g]:
                for scale in table[d][g][pack]:
                    yield d, g, pack, scale, table[d][g][pack][scale]


# ─────────────────────────────────────────────────────────────────────────────
# Rendering — single run
# ─────────────────────────────────────────────────────────────────────────────

def _render_single(table: dict) -> str:
    out = ["# brute benchmark report", ""]
    out.append(f"_3 scales × paired bit1/bool, devices: {sorted(table.keys())}_")
    out.append("")

    # --- regression summary -------------------------------------------------
    regressions = []
    for d, g, pack, scale, c in _all(table):
        b1 = c.get("bit1")
        bo = c.get("bool")
        if b1 and bo and bo / b1 < 1.0:
            regressions.append((bo / b1, d, g, pack, scale, b1, bo))
    if regressions:
        regressions.sort()
        out.append("## Regressions — bit1 slower than bool")
        out.append("")
        out.append("| Speedup | Device | Op | Pack | Scale | bit1 | bool |")
        out.append("|---|---|---|---|---|---|---|")
        for sp, d, g, pack, scale, b1, bo in regressions[:25]:
            pack_disp = pack if pack != "?" else "—"
            out.append(f"| <u>{sp:.2f}×</u> | {d} | `{g}` | {pack_disp} | "
                       f"{_scale_label(scale)} | {_format_seconds(b1)} "
                       f"| {_format_seconds(bo)} |")
        if len(regressions) > 25:
            out.append(f"| _… {len(regressions)-25} more …_ | | | | | | |")
        out.append("")

    # --- per-device, per-op tables -----------------------------------------
    for device in sorted(table.keys()):
        out.append(f"## Device: `{device}`")
        out.append("")

        # Group ops by category prefix (everything before the first '/').
        ops_by_cat: dict[str, list[str]] = defaultdict(list)
        for g in table[device]:
            cat = g.split("/", 1)[0] if "/" in g else g
            ops_by_cat[cat].append(g)

        for cat in sorted(ops_by_cat):
            out.append(f"### {cat}")
            out.append("")
            # Determine which pack widths appear for any op in this category.
            packs = set()
            for g in ops_by_cat[cat]:
                packs.update(table[device][g].keys())
            packs = sorted(packs)
            multi_pack = len(packs) > 1

            # Build the header: one column per (pack × scale) combination, but
            # only if there's >1 pack; otherwise just scales.
            if multi_pack:
                scale_cols = [(p, s) for p in packs for s in _SCALES]
                hdr = (["Op"]
                       + [f"{p}<br>{_scale_label(s)}" for p, s in scale_cols]
                       + ["best speedup"])
            else:
                scale_cols = [(packs[0], s) for s in _SCALES]
                hdr = (["Op"]
                       + [_scale_label(s) for _, s in scale_cols]
                       + ["best speedup"])
            out.append("| " + " | ".join(hdr) + " |")
            out.append("|" + "---|" * len(hdr))

            for g in sorted(ops_by_cat[cat]):
                row = [f"`{g.split('/', 1)[1] if '/' in g else g}`"]
                speedups: list[float] = []
                for pack, scale in scale_cols:
                    cell = table[device][g].get(pack, {}).get(scale, {})
                    b1 = cell.get("bit1")
                    bo = cell.get("bool")
                    if b1 is None and bo is None:
                        row.append("—")
                        continue
                    if b1 and bo:
                        sp = bo / b1
                        speedups.append(sp)
                        # Format: "1.34× ⏱ 12.3µs"
                        row.append(f"{sp:.2f}×<br>_{_format_seconds(b1)}_")
                    elif b1:
                        row.append(f"bit1 only<br>_{_format_seconds(b1)}_")
                    else:
                        row.append(f"bool only<br>_{_format_seconds(bo)}_")
                if speedups:
                    best = max(speedups)
                    row.append(_format_speedup(best))
                else:
                    row.append("—")
                out.append("| " + " | ".join(row) + " |")
            out.append("")

    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering — diff (head vs base)
# ─────────────────────────────────────────────────────────────────────────────

def _render_diff(base: dict, head: dict, base_name: str, head_name: str) -> str:
    out = [f"# brute benchmark report — {head_name} vs {base_name}", ""]
    out.append(f"Δ = head_speedup / base_speedup. Values > 1.0 mean bit1 "
               f"got *better* relative to bool between runs.")
    out.append("")

    biggest_wins: list[tuple] = []
    biggest_regressions: list[tuple] = []
    keys = set()
    for d, g, pack, scale, _ in _all(base):
        keys.add((d, g, pack, scale))
    for d, g, pack, scale, _ in _all(head):
        keys.add((d, g, pack, scale))

    for d, g, pack, scale in keys:
        bc = base.get(d, {}).get(g, {}).get(pack, {}).get(scale, {})
        hc = head.get(d, {}).get(g, {}).get(pack, {}).get(scale, {})
        if not (bc.get("bit1") and bc.get("bool") and hc.get("bit1") and hc.get("bool")):
            continue
        base_sp = bc["bool"] / bc["bit1"]
        head_sp = hc["bool"] / hc["bit1"]
        if base_sp <= 0:
            continue
        delta = head_sp / base_sp
        if delta >= 1.05:
            biggest_wins.append((delta, d, g, pack, scale, base_sp, head_sp))
        elif delta <= 0.95:
            biggest_regressions.append((delta, d, g, pack, scale, base_sp, head_sp))

    def _panel(title, rows, ascending):
        out.append(f"## {title}")
        out.append("")
        if not rows:
            out.append("_(none)_")
            out.append("")
            return
        rows.sort(reverse=not ascending)
        out.append("| Δ | Device | Op | Pack | Scale | base bit1/bool | head bit1/bool |")
        out.append("|---|---|---|---|---|---|---|")
        for delta, d, g, pack, scale, b, h in rows[:25]:
            arrow = "↑" if delta >= 1.0 else "↓"
            out.append(f"| {arrow} {delta:.2f}× | {d} | `{g}` | {pack} | "
                       f"{_scale_label(scale)} | {b:.2f}× | {h:.2f}× |")
        out.append("")

    _panel("Top improvements", biggest_wins, ascending=False)
    _panel("Top regressions", biggest_regressions, ascending=True)

    # Full delta matrix per (device, op).
    for d in sorted(set(list(base.keys()) + list(head.keys()))):
        out.append(f"## Device: `{d}`")
        out.append("")
        groups = sorted(set(list(base.get(d, {}).keys()) + list(head.get(d, {}).keys())))
        ops_by_cat: dict[str, list[str]] = defaultdict(list)
        for g in groups:
            ops_by_cat[g.split("/", 1)[0] if "/" in g else g].append(g)
        for cat in sorted(ops_by_cat):
            out.append(f"### {cat}")
            out.append("")
            out.append("| Op | scale | pack | base | head | Δ |")
            out.append("|---|---|---|---|---|---|")
            for g in sorted(ops_by_cat[cat]):
                packs = sorted(set(
                    list(base.get(d, {}).get(g, {}).keys())
                    + list(head.get(d, {}).get(g, {}).keys())
                ))
                for pack in packs:
                    for scale in _SCALES:
                        bc = base.get(d, {}).get(g, {}).get(pack, {}).get(scale, {})
                        hc = head.get(d, {}).get(g, {}).get(pack, {}).get(scale, {})
                        b1b = bc.get("bit1")
                        bob = bc.get("bool")
                        b1h = hc.get("bit1")
                        boh = hc.get("bool")
                        if not (b1b and bob and b1h and boh):
                            continue
                        bsp = bob / b1b
                        hsp = boh / b1h
                        delta = hsp / bsp if bsp > 0 else None
                        d_s = f"{delta:.2f}×" if delta else "—"
                        out.append(f"| `{g.split('/', 1)[1] if '/' in g else g}` | "
                                   f"{_scale_label(scale)} | {pack} | "
                                   f"{bsp:.2f}× | {hsp:.2f}× | {d_s} |")
            out.append("")

    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="make_report",
        description="Render a benchmark report from pytest-benchmark JSON.",
    )
    parser.add_argument("json_paths", nargs="*",
                        help="One JSON for single-run mode, or two for diff "
                             "(base, head). Defaults to newest .benchmarks/*.json.")
    parser.add_argument("--markdown", "-o", default=None,
                        help="Write the report to this file (default: stdout).")
    args = parser.parse_args(argv)

    if not args.json_paths:
        repo_root = Path(__file__).resolve().parents[2]
        path = _latest_json(repo_root / ".benchmarks")
        if path is None:
            print("No JSON found under .benchmarks/. "
                  "Run `pytest tests/benchmarks --benchmark-json=...` first.",
                  file=sys.stderr)
            return 2
        paths = [path]
    else:
        paths = [Path(p) for p in args.json_paths]
        for p in paths:
            if not p.exists():
                print(f"File not found: {p}", file=sys.stderr)
                return 2

    if len(paths) == 1:
        with open(paths[0]) as fh:
            data = json.load(fh)
        entries = data.get("benchmarks", [])
        if not entries:
            print(f"No benchmarks in {paths[0]}", file=sys.stderr)
            return 1
        report = _render_single(_aggregate(entries))
        n_summary = f"{len(entries)} benchmark entries from {paths[0].name}"
    elif len(paths) == 2:
        with open(paths[0]) as fh:
            base = json.load(fh)
        with open(paths[1]) as fh:
            head = json.load(fh)
        report = _render_diff(
            _aggregate(base.get("benchmarks", [])),
            _aggregate(head.get("benchmarks", [])),
            paths[0].name, paths[1].name,
        )
        n_summary = f"diff: {paths[0].name} → {paths[1].name}"
    else:
        print("Pass at most two JSON files (base + head for diff).", file=sys.stderr)
        return 2

    if args.markdown:
        Path(args.markdown).write_text(report)
        print(f"Wrote {args.markdown} ({n_summary})", file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

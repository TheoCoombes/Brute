"""Aggregate pytest-benchmark JSON output into markdown speedup tables.

Usage:
    # Run the benchmarks once, dumping JSON.
    pytest tests/benchmarks --benchmark-only \
        --benchmark-group-by=group \
        --benchmark-sort=name \
        --benchmark-json=.benchmarks/full.json

    # Render the report.
    python -m tests.benchmarks.make_report                       # newest *.json
    python -m tests.benchmarks.make_report .benchmarks/full.json
    python -m tests.benchmarks.make_report --markdown report.md  # write to file

The report groups results by (device, size) and prints one table per
(device, size) pair. Each row pairs `test_bit1_<op>` against `test_bool_<op>`
in the same `benchmark.group` and reports the speedup (bool / bit1).

A regression summary at the end lists every (op, device, size) pair where
bit1 is slower than bool, ordered by severity (worst first).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


# JSON discovery

def _latest_json(root: Path) -> Optional[Path]:
    """Find the newest pytest-benchmark JSON under .benchmarks/."""
    if not root.exists():
        return None
    candidates = list(root.rglob("*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# Benchmark-entry parsing

_BIT1_RE = re.compile(r"^test_bit1_(.+)$")
_BOOL_RE = re.compile(r"^test_bool_(.+)$")


def _classify(name: str) -> tuple[Optional[str], Optional[str]]:
    """Return (kind, op) for a benchmark name.

    `kind` is 'bit1' / 'bool' / None.  `op` is the trailing identifier (the
    string after the prefix), e.g. 'and', 'xor', 'invert', 'index_row'.
    """
    # Strip parameter suffix like '[cpu-05_huge]' first.
    core = name.split("[", 1)[0]
    if (m := _BIT1_RE.match(core)) is not None:
        return "bit1", m.group(1)
    if (m := _BOOL_RE.match(core)) is not None:
        return "bool", m.group(1)
    return None, None


def _stat_seconds(entry: dict) -> float:
    """Pick the most stable single-number representative of the timing."""
    s = entry["stats"]
    # Median is more stable than mean against the slow-warmup tail.
    return float(s.get("median", s["mean"]))


def _device(entry: dict) -> str:
    params = entry.get("params") or {}
    if "device" in params:
        return str(params["device"])
    # Fallback: pull from the parameter suffix in the name.
    suffix = entry["name"].split("[", 1)
    if len(suffix) == 2:
        for tok in suffix[1].rstrip("]").split("-"):
            if tok in ("cpu", "cuda", "mps"):
                return tok
    return "?"


def _size_id(entry: dict) -> str:
    """A short, lexically-sortable label for the benchmark scale.

    Pytest-benchmark stores both the raw parameter values (in `params`) and
    the human-readable pytest ID (in the `[...]` suffix on `name`). We prefer
    the latter because it preserves the `01_tiny` → `05_huge` ordering.
    """
    suffix = entry["name"].split("[", 1)
    if len(suffix) == 2:
        tokens = [t for t in suffix[1].rstrip("]").split("-") if t not in ("cpu", "cuda", "mps")]
        if tokens:
            return "-".join(tokens)
    return str(entry.get("params") or {})


def _format_seconds(s: float) -> str:
    if s >= 1.0:
        return f"{s:.3f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.2f} ms"
    if s >= 1e-6:
        return f"{s * 1e6:.1f} µs"
    return f"{s * 1e9:.0f} ns"


# Aggregation

def _aggregate(entries: list[dict]) -> dict:
    """Group entries by (device, size, op) and pair bit1 ↔ bool."""
    # table[device][size][op] = {'bit1': sec, 'bool': sec, 'group': str}
    table: dict[str, dict[str, dict[str, dict[str, float | str]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    for e in entries:
        kind, op = _classify(e["name"])
        if kind is None or op is None:
            continue
        dev = _device(e)
        size = _size_id(e)
        cell = table[dev][size][op]
        cell[kind] = _stat_seconds(e)
        cell.setdefault("group", e.get("group", op))
    return table


# Rendering

def _render_pair_table(device: str, size: str,
                       cells: dict[str, dict[str, float | str]]) -> list[str]:
    rows = []
    for op, c in sorted(cells.items()):
        bit1 = c.get("bit1")
        boo  = c.get("bool")
        if bit1 is None and boo is None:
            continue
        speedup = (boo / bit1) if (bit1 and boo) else None
        rows.append((c.get("group", op), op, bit1, boo, speedup))

    if not rows:
        return []

    out = [f"### {device} — {size}", ""]
    out.append("| Op | bit1 | bool | speedup (bool / bit1) |")
    out.append("|---|---|---|---|")
    for group, op, bit1, boo, speedup in rows:
        bit1_s = _format_seconds(bit1) if bit1 is not None else "—"
        boo_s  = _format_seconds(boo)  if boo  is not None else "—"
        if speedup is None:
            sp_s = "—"
        else:
            sp_s = f"**{speedup:.2f}×**" if speedup < 1.0 else f"{speedup:.2f}×"
        out.append(f"| `{group}` | {bit1_s} | {boo_s} | {sp_s} |")
    out.append("")
    return out


def _render_regressions(table: dict) -> list[str]:
    regressions = []
    for device, sizes in table.items():
        for size, cells in sizes.items():
            for op, c in cells.items():
                bit1 = c.get("bit1")
                boo  = c.get("bool")
                if bit1 and boo and (boo / bit1) < 1.0:
                    regressions.append((boo / bit1, device, size, op, bit1, boo, c.get("group", op)))
    if not regressions:
        return []
    regressions.sort()
    out = ["## Regressions (bit1 slower than bool)", ""]
    out.append("| Speedup | Device | Size | Op | bit1 | bool |")
    out.append("|---|---|---|---|---|---|")
    for speedup, device, size, op, bit1, boo, group in regressions:
        out.append(f"| **{speedup:.2f}×** | {device} | {size} | `{group}` "
                   f"| {_format_seconds(bit1)} | {_format_seconds(boo)} |")
    out.append("")
    return out


def _render_report(table: dict) -> str:
    out = ["# brute benchmark report", ""]
    for device in sorted(table.keys()):
        for size in sorted(table[device].keys()):
            section = _render_pair_table(device, size, table[device][size])
            out.extend(section)
    out.extend(_render_regressions(table))
    return "\n".join(out)


# CLI

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="make_report",
        description="Aggregate pytest-benchmark JSON into markdown speedup tables.",
    )
    parser.add_argument(
        "json_path", nargs="?", default=None,
        help="Path to a pytest-benchmark JSON file. Default: newest .benchmarks/*.json.",
    )
    parser.add_argument(
        "--markdown", "-o", default=None,
        help="Write the report to this file (default: stdout).",
    )
    args = parser.parse_args(argv)

    if args.json_path:
        path = Path(args.json_path)
    else:
        repo_root = Path(__file__).resolve().parents[2]
        path = _latest_json(repo_root / ".benchmarks")
        if path is None:
            print("No JSON found under .benchmarks/. "
                  "Run `pytest tests/benchmarks --benchmark-json=...` first.",
                  file=sys.stderr)
            return 2

    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    with open(path) as fh:
        data = json.load(fh)

    entries = data.get("benchmarks", [])
    if not entries:
        print(f"No benchmarks in {path}", file=sys.stderr)
        return 1

    report = _render_report(_aggregate(entries))

    if args.markdown:
        Path(args.markdown).write_text(report)
        print(f"Wrote {args.markdown} ({len(entries)} benchmark entries from {path.name})",
              file=sys.stderr)
    else:
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

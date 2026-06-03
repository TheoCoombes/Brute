"""Subprocess-based experiment executor with ProcessPoolExecutor parallelism."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODEL_DIR = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


@dataclass
class ExperimentSpec:
    name: str
    group: str
    task: str
    seq_len: int
    steps: int
    hypothesis: str
    cfg_overrides: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    name: str
    group: str
    task: str
    seq_len: int
    acc: float
    loss: float
    elapsed_s: float
    hypothesis: str
    cfg_overrides: dict
    timestamp: str
    error: Optional[str] = None


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())


def build_command(spec: ExperimentSpec, json_out: Path) -> List[str]:
    cmd = [
        PYTHON,
        str(MODEL_DIR / "probe_synthetic.py"),
        "--tasks",
        spec.task,
        "--lengths",
        str(spec.seq_len),
        "--steps",
        str(spec.steps),
        "--json-out",
        str(json_out),
    ]
    for flag, value in spec.cfg_overrides.items():
        cmd.append(flag)
        if value != "":
            cmd.append(str(value))
    return cmd


def run_one(spec: ExperimentSpec) -> ExperimentResult:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()
    tmp_json = RESULTS_DIR / f".{spec.name}_{os.getpid()}_{ts}.tmp.json"
    t0 = time.time()
    try:
        cmd = build_command(spec, tmp_json)
        proc = subprocess.run(
            cmd,
            cwd=str(MODEL_DIR),
            text=True,
            capture_output=True,
            timeout=90,
        )
        elapsed = time.time() - t0
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or f"exit code {proc.returncode}").strip()
            return ExperimentResult(
                name=spec.name, group=spec.group, task=spec.task, seq_len=spec.seq_len,
                acc=0.0, loss=float("inf"), elapsed_s=elapsed,
                hypothesis=spec.hypothesis, cfg_overrides=spec.cfg_overrides,
                timestamp=ts, error=err[-4000:],
            )
        rows = json.loads(tmp_json.read_text())
        row = next((r for r in rows if r.get("model") == "haemmr"), rows[0] if rows else None)
        if row is None:
            raise ValueError("probe_synthetic wrote no result rows")
        return ExperimentResult(
            name=spec.name,
            group=spec.group,
            task=spec.task,
            seq_len=spec.seq_len,
            acc=float(row.get("acc", 0.0)),
            loss=float(row.get("loss", float("inf"))),
            elapsed_s=elapsed,
            hypothesis=spec.hypothesis,
            cfg_overrides=dict(spec.cfg_overrides),
            timestamp=ts,
        )
    except Exception as exc:  # subprocess timeout and parse failures land here
        return ExperimentResult(
            name=spec.name, group=spec.group, task=spec.task, seq_len=spec.seq_len,
            acc=0.0, loss=float("inf"), elapsed_s=time.time() - t0,
            hypothesis=spec.hypothesis, cfg_overrides=spec.cfg_overrides,
            timestamp=ts, error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        tmp_json.unlink(missing_ok=True)


def _result_path(result: ExperimentResult) -> Path:
    return RESULTS_DIR / f"{result.name}_{result.timestamp}.json"


def _save_result(result: ExperimentResult) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _result_path(result).write_text(json.dumps(asdict(result), indent=2, sort_keys=True))
    if result.error is None:
        (RESULTS_DIR / f"{result.name}.done").touch()


def is_done(name: str) -> bool:
    return (RESULTS_DIR / f"{name}.done").exists()


def _latest_result(name: str) -> Optional[ExperimentResult]:
    paths = sorted(RESULTS_DIR.glob(f"{name}_*.json"))
    if not paths:
        return None
    data = json.loads(paths[-1].read_text())
    return ExperimentResult(**data)


def run_group(specs: List[ExperimentSpec], n_workers: int = 4) -> List[ExperimentResult]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: List[ExperimentResult] = []
    pending: List[ExperimentSpec] = []
    for spec in specs:
        cached = _latest_result(spec.name) if is_done(spec.name) else None
        if cached is not None:
            results.append(cached)
        else:
            pending.append(spec)

    if not pending:
        return results

    if n_workers <= 1:
        for spec in pending:
            result = run_one(spec)
            _save_result(result)
            results.append(result)
        return results

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        fut_to_spec = {pool.submit(run_one, spec): spec for spec in pending}
        for fut in as_completed(fut_to_spec):
            result = fut.result()
            _save_result(result)
            results.append(result)
    return results


def load_results() -> List[ExperimentResult]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out: List[ExperimentResult] = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            out.append(ExperimentResult(**json.loads(path.read_text())))
        except Exception:
            continue
    return out

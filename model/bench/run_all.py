"""Main entry point: run experiment groups, update AGENTS.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.experiments import ALL_EXPERIMENTS, EXPERIMENTS_BY_GROUP, GROUP_ORDER
from bench.report import update_agents_md
from bench.runner import RESULTS_DIR, load_results, run_group


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--groups",
        nargs="+",
        default=GROUP_ORDER,
        choices=GROUP_ORDER + ["all"],
        help="Which groups to run (default: all)",
    )
    p.add_argument("--workers", type=int, default=4, help="Parallel workers")
    p.add_argument("--force", action="store_true", help="Re-run already-done experiments")
    p.add_argument("--dry-run", action="store_true", help="Print what would run, do not execute")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    groups = GROUP_ORDER if "all" in args.groups else args.groups
    if args.dry_run:
        print(f"Catalog contains {len(ALL_EXPERIMENTS)} experiments.")

    for group in groups:
        specs = EXPERIMENTS_BY_GROUP.get(group, [])
        if args.dry_run:
            for s in specs:
                print(f"  [{group}] {s.name}: {s.task} len={s.seq_len} steps={s.steps}")
            continue

        print(f"\n=== Group: {group} ({len(specs)} experiments) ===")
        if args.force:
            for s in specs:
                (RESULTS_DIR / f"{s.name}.done").unlink(missing_ok=True)
        results = run_group(specs, n_workers=args.workers)
        print(f"  completed {sum(1 for r in results if r.error is None)}/{len(results)}")

    if args.dry_run:
        return

    print("\nUpdating AGENTS.md ...")
    update_agents_md(load_results())
    print("Done.")


if __name__ == "__main__":
    main()

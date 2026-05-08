#!/usr/bin/env python3
"""Re-verify and aggregate eval results.

Walks the canonical layout
    results/<model>/<tier>/<cond>.json    (consolidated per-cell)

For each cell file:
  1. Re-parses raw_response into (submitted_satisfiable, submitted_solution).
  2. Re-runs validate_answer against the corresponding shipped dataset
     (`mathconstraint_easy/` for tier=easy, `mathconstraint/` for tier=hard/hard2).
  3. Updates the cell record in place with refreshed
     `submitted_satisfiable / submitted_solution / correct /
     satisfiability_correct / solution_correct / validation_details`,
     and recomputes the cell's `metrics` block.
  4. Prints a per-(model, tier, condition) accuracy table at the end.

Default mode is in-place update. Use `--dry-run` to print what would change
without touching files.

Usage:
    scripts/verify.py
    scripts/verify.py --results-dir results --dry-run
    scripts/verify.py --models gpt-5.5 sonnet-4.6
    scripts/verify.py --tiers easy
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from math_constraint.eval import (  # noqa: E402
    EvalResult,
    compute_metrics,
    parse_response,
    validate_answer,
)

TIER_TO_DATASET = {
    "easy":  REPO / "mathconstraint_easy",
    "hard":  REPO / "mathconstraint",
    "hard2": REPO / "mathconstraint",
}


def _load_dataset(path: Path) -> dict[str, dict]:
    pj = path / "problems.jsonl"
    if not pj.exists():
        raise FileNotFoundError(f"{pj} not found")
    return {json.loads(l)["name"]: json.loads(l) for l in pj.open()}


def _process_cell(arg) -> tuple[str, dict]:
    cell_path, dry_run = arg
    cell_path = Path(cell_path)
    parts = cell_path.parts
    # results/<model>/<tier>/<cond>.json
    model = parts[-3]
    tier = parts[-2]
    cond = cell_path.stem
    dataset_dir = TIER_TO_DATASET.get(tier)
    if dataset_dir is None:
        return str(cell_path), {"error": f"unknown tier {tier!r}"}

    gt = _load_dataset(dataset_dir)

    d = json.loads(cell_path.read_text())
    if not isinstance(d, dict) or "results" not in d:
        return str(cell_path), {"error": "missing 'results'"}

    records = d["results"]
    n_changed = 0
    n_correct = 0
    flips = 0
    for r in records:
        problem = gt.get(r["problem_name"])
        if problem is None:
            continue

        content = parse_response(r.get("raw_response"))
        if content is None:
            new_sat, new_sol = None, None
        else:
            new_sat = content.get("satisfiable")
            new_sol = content.get("solution")
        v = validate_answer(problem, new_sat, new_sol)
        new_correct = v["correct"]

        old_tuple = (
            r.get("submitted_satisfiable"),
            r.get("correct"),
            r.get("satisfiability_correct"),
            r.get("solution_correct"),
        )
        new_tuple = (
            new_sat,
            new_correct,
            v["satisfiability_correct"],
            v["solution_correct"],
        )
        if old_tuple != new_tuple:
            n_changed += 1
            if bool(r.get("correct")) != bool(new_correct):
                flips += 1
            r["submitted_satisfiable"] = new_sat
            r["submitted_solution"] = new_sol
            r["correct"] = new_correct
            r["satisfiability_correct"] = v["satisfiability_correct"]
            r["solution_correct"] = v["solution_correct"]
            r["validation_details"] = v["details"]

        if new_correct:
            n_correct += 1

    if n_changed:
        er = [
            EvalResult(**{k: r.get(k) for k in EvalResult.__dataclass_fields__ if k in r})
            for r in records
        ]
        metrics = compute_metrics(er)
        from math_constraint.cli import _metrics_to_dict  # noqa: E402
        d["metrics"] = _metrics_to_dict(metrics)
        if not dry_run:
            cell_path.write_text(json.dumps(d, indent=2, default=str))

    return str(cell_path), {
        "model": model,
        "tier": tier,
        "cond": cond,
        "n": len(records),
        "correct": n_correct,
        "changed": n_changed,
        "flipped": flips,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results-dir", default="results", help="root of model/tier/cond.json layout")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="don't write changes back")
    ap.add_argument("--models", nargs="*", help="restrict to these model dirs (default: all)")
    ap.add_argument("--tiers", nargs="*", choices=["easy", "hard", "hard2"], help="restrict tiers")
    args = ap.parse_args()

    root = Path(args.results_dir)
    if not root.exists():
        print(f"results dir not found: {root}", file=sys.stderr)
        return 1

    cells: list[Path] = []
    model_glob = args.models or [d.name for d in root.iterdir() if d.is_dir()]
    tiers = args.tiers or ["easy", "hard", "hard2"]
    for m in sorted(model_glob):
        for tier in tiers:
            for cond in ("tools", "no_tools"):
                p = root / m / tier / f"{cond}.json"
                if p.exists():
                    cells.append(p)
    if not cells:
        print("no cells found", file=sys.stderr)
        return 1

    print(f"[verify] {len(cells)} cells; workers={args.workers}; dry_run={args.dry_run}", flush=True)

    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(_process_cell, [(str(p), args.dry_run) for p in cells])

    total_changed = sum(r[1].get("changed", 0) for r in results if "error" not in r[1])
    total_flipped = sum(r[1].get("flipped", 0) for r in results if "error" not in r[1])

    # Aggregate per (model, tier, cond)
    print()
    print(f"{'model':<28} {'tier':<6} {'cond':<10} {'n':>4} {'correct':>8} {'acc':>8} {'flipped':>8}")
    print("-" * 78)
    for path, info in sorted(results):
        if "error" in info:
            print(f"  ERROR {path}: {info['error']}")
            continue
        n = info["n"]; c = info["correct"]
        acc = (c / n) if n else 0
        print(f"  {info['model']:<26} {info['tier']:<6} {info['cond']:<10} {n:>4} {c:>8} {acc:>8.3f} {info['flipped']:>8}")

    print()
    print(f"[verify] {total_changed} record-level updates; {total_flipped} correctness flips; "
          f"{'DRY RUN' if args.dry_run else 'written in place'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

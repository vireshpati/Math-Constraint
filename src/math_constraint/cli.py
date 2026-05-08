"""Unified CLI for MathConstraint.

Subcommands: build / eval / sweep / verify / stats. Reads .env from the repo
root if present so secrets (OPENROUTER_API_KEY, HF_TOKEN, WANDB_*) load
without explicit export.

Invoke as: ``python -m math_constraint <subcommand>`` (or via the
``scripts/{build,eval,sweep}.sh`` wrappers).
"""

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from math_constraint import load_dataset
from math_constraint.eval import (
    EvalConfig,
    EvalResult,
    Evaluator,
    compute_metrics,
)
from math_constraint.generate import run_build
from math_constraint.llm import create_client
from math_constraint.problems import PROBLEMS, get_backend, solve, list_problems


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------


# Walk up from src/math_constraint/cli.py to the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    path = REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    output = Path(args.output)
    profile_name, profile = _load_build_profile(args.profile)
    if profile:
        args.seed = profile.get("seed", args.seed)
        args.samples_per_type = profile.get("samples_per_type", args.samples_per_type)
        args.problems = profile.get("problem_types", args.problems)
        args.sat_hint_fraction = profile.get("sat_hint_fraction", args.sat_hint_fraction)
        args.partial_assignment_fraction = profile.get(
            "partial_assignment_fraction",
            args.partial_assignment_fraction,
        )
    manifest = run_build(
        output_dir=output,
        seed=args.seed,
        samples_per_type=args.samples_per_type,
        workers=args.workers,
        problem_types=args.problems,
        domain_overrides=(profile or {}).get("param_domains"),
        sat_hint_fraction=args.sat_hint_fraction,
        partial_assignment_fraction=args.partial_assignment_fraction,
        profile_name=profile_name,
        profile=profile,
    )
    print(f"[build] {manifest['total_instances']} instances "
          f"({manifest['satisfiable_instances']} SAT / "
          f"{manifest['unsatisfiable_instances']} UNSAT) "
          f"across {manifest['total_problem_types']} types -> {output}")
    print(f"[build] SAT instance %: {manifest['sat_unsat_split']['instances_sat_pct']}")
    if args.stats:
        _print_stats(manifest)
    if args.verify:
        rc = _verify_dataset(output, workers=args.workers)
        if rc != 0:
            return rc
    # Hard checks per the plan's constraint list.
    sat_pct = manifest["sat_unsat_split"]["instances_sat_pct"]
    if sat_pct > 80.0:
        print(f"[build] WARNING: SAT instance% = {sat_pct} > 80")
    if manifest["total_instances"] > 300:
        print(f"[build] WARNING: total_instances = {manifest['total_instances']} > 300")
    return 0


def _load_build_profile(profile_ref: str | None) -> tuple[str | None, dict[str, Any] | None]:
    if not profile_ref:
        return None, None
    if ":" in profile_ref:
        path_str, name = profile_ref.split(":", 1)
    else:
        path_str, name = "configs/dataset_profiles.yaml", profile_ref
    blob = yaml.safe_load(Path(path_str).read_text())
    profiles = blob.get("profiles", blob)
    if name not in profiles:
        raise KeyError(f"Unknown dataset profile {name!r} in {path_str}")
    profile = {**(blob.get("defaults") or {}), **profiles[name]}
    profile["name"] = name
    return name, profile


def _print_stats(manifest: dict) -> None:
    print()
    print(f"  PROBLEM TYPE TABLE  ({manifest['total_problem_types']} types, "
          f"{manifest['total_instances']} instances)")
    print("  " + "-" * 100)
    print(f"  {'type':<32s} {'bk':<5s} {'inst':>5s} {'sat':>4s} {'uns':>4s} "
          f"{'hint':>4s} {'easy':>5s} {'med':>5s} {'hard':>5s}")
    for ptype, m in manifest["problem_types"].items():
        tc = m["solve_tier_counts"]
        print(f"  {ptype:<32s} {m['backend']:<5s} {m['instances']:>5d} "
              f"{m['satisfiable_instances']:>4d} {m['unsatisfiable_instances']:>4d} "
              f"{m['hinted_instances']:>4d} {tc['easy']:>5d} {tc['medium']:>5d} {tc['hard']:>5d}")
    print(f"  thresholds (ms): p33={manifest['solve_tier_thresholds_ms']['p33_ms']} "
          f"p66={manifest['solve_tier_thresholds_ms']['p66_ms']}")


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    return _verify_dataset(Path(args.dataset), workers=args.workers)


def _verify_dataset(dataset_dir: Path, workers: int) -> int:
    instances = load_dataset(dataset_dir)
    sat = [i for i in instances if i["satisfiable"]]
    print(f"[verify] {len(instances)} instances ({len(sat)} SAT to re-solve)")

    def _resolve(i: dict) -> tuple[str, bool, str]:
        try:
            r = solve(i["problem_type"], **i["params"])
            return (i["name"], r.satisfiable, "")
        except Exception as e:
            return (i["name"], False, str(e))

    bad: list[tuple[str, str]] = []
    if workers <= 1:
        for inst in sat:
            name, ok, err = _resolve(inst)
            if not ok:
                bad.append((name, err or "solver disagreed: returned UNSAT"))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for fut in as_completed([pool.submit(_resolve, i) for i in sat]):
                name, ok, err = fut.result()
                if not ok:
                    bad.append((name, err or "solver disagreed: returned UNSAT"))

    if bad:
        print(f"[verify] FAIL: {len(bad)} of {len(sat)} SAT instances did not re-solve")
        for n, e in bad[:20]:
            print(f"  {n}: {e}")
        return 1

    h = sum(1 for i in instances if i.get("partial_assignment") is not None)
    types = sorted({i["problem_type"] for i in instances})
    print(f"[verify] OK: {len(instances)} instances, "
          f"{len(sat)} SAT (verified) / {len(instances) - len(sat)} UNSAT, "
          f"{h} hinted, {len(types)} problem types")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    manifest_path = Path(args.dataset) / "manifest.json"
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}")
        return 1
    _print_stats(json.loads(manifest_path.read_text()))
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-")


def _build_provider_routing(args: argparse.Namespace) -> dict | None:
    """OpenRouter routing. We DON'T set data_collection here — the user's
    account-level privacy settings already govern what providers are allowed.
    Adding a request-level filter on top can yield empty-intersection 404s.
    """
    if args.provider != "openrouter":
        return None
    return {"allow_fallbacks": True}


def _build_reasoning(args: argparse.Namespace) -> dict | None:
    r: dict[str, Any] = {}
    if args.reasoning_effort:
        r["effort"] = args.reasoning_effort
    if args.reasoning_max_tokens:
        r["max_tokens"] = args.reasoning_max_tokens
    if not r:
        return None
    if not args.reasoning_include:
        r["exclude"] = True
    return r


def _evaluate_cell(
    *,
    model: str,
    dataset: list[dict],
    provider: str,
    base_url: str | None,
    api_key: str | None,
    enable_tools: bool,
    max_tokens: int,
    temperature: float,
    seed: int | None,
    max_tool_rounds: int,
    reasoning: dict | None,
    provider_routing: dict | None,
    models_fallback: list[str] | None,
    workers: int,
    wandb_run: Any,
    trace_dir: Path | None = None,
) -> tuple[list[EvalResult], float]:
    """Run a single (model, condition) cell. Returns results + wall_seconds."""
    config = EvalConfig.from_provider(
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        base_url=base_url,
        enable_tools=enable_tools,
        max_tool_rounds=max_tool_rounds,
        reasoning=reasoning,
        seed=seed,
        provider_routing=provider_routing,
        models_fallback=models_fallback,
    )
    client_kwargs: dict[str, Any] = {}
    if provider == "openrouter":
        if provider_routing:
            client_kwargs["provider_routing"] = provider_routing
        if models_fallback:
            client_kwargs["models_fallback"] = models_fallback
    client = create_client(provider, model, base_url=config.base_url,
                           api_key=api_key, **client_kwargs)
    evaluator = Evaluator(config, client)

    results: list[EvalResult] = []
    # Per-instance resume: load any existing trace_dir/{name}.json into results
    # and skip those instances on the API path. Trace dir is unique per
    # (output_dir, model, condition), so cross-talk between cells is impossible.
    pending = list(dataset)
    if trace_dir:
        resumed = 0
        keep = []
        for inst in dataset:
            tp = trace_dir / f"{inst['name']}.json"
            if tp.exists():
                try:
                    results.append(EvalResult.from_dict(json.loads(tp.read_text())))
                    resumed += 1
                    continue
                except Exception:
                    pass  # malformed trace, fall through and re-run
            keep.append(inst)
        pending = keep
        if resumed:
            print(f"  [resume] {resumed}/{len(dataset)} instances loaded from {trace_dir}; running {len(pending)} fresh")
    t0 = time.time()
    if workers <= 1:
        for i, inst in enumerate(pending):
            r = evaluator.evaluate(inst)
            results.append(r)
            if trace_dir: (trace_dir / f"{r.problem_name}.json").write_text(json.dumps(r.to_dict(), default=str))
            wandb_run.log(_wandb_row(r))
            if (i + 1) % 10 == 0 or i == len(pending) - 1:
                print(f"  [{i + 1}/{len(pending)}] elapsed={time.time() - t0:.1f}s")
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(evaluator.evaluate, inst): inst for inst in pending}
            for n, fut in enumerate(as_completed(futures)):
                try:
                    r = fut.result()
                except Exception as e:
                    inst = futures[fut]
                    r = EvalResult(
                        problem_name=inst["name"], problem_type=inst["problem_type"],
                        params=inst.get("params", {}), difficulty=inst.get("difficulty", {}),
                        is_hinted=inst.get("partial_assignment") is not None,
                        is_satisfiable=inst["satisfiable"], model=model,
                        provider=provider, condition="tools" if enable_tools else "no_tools",
                        error=f"worker_crash: {type(e).__name__}: {e}",
                        failure_bucket="api_error",
                    )
                results.append(r)
                if trace_dir: (trace_dir / f"{r.problem_name}.json").write_text(json.dumps(r.to_dict(), default=str))
                wandb_run.log(_wandb_row(r))
                if (n + 1) % 10 == 0 or n == len(pending) - 1:
                    print(f"  [{n + 1}/{len(pending)}] elapsed={time.time() - t0:.1f}s")

    return results, time.time() - t0


def _wandb_row(r: EvalResult) -> dict[str, Any]:
    """Per-instance row logged to wandb. Stage 7 fields."""
    return {
        "problem_name": r.problem_name,
        "problem_type": r.problem_type,
        "is_hinted": r.is_hinted,
        "is_satisfiable": r.is_satisfiable,
        "solve_tier": r.difficulty.get("solve_tier"),
        "solve_pct_global": r.difficulty.get("solve_pct_global"),
        "correct": r.correct,
        "satisfiability_correct": r.satisfiability_correct,
        "solution_correct": r.solution_correct,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "reasoning_tokens": r.reasoning_tokens,
        "cached_tokens": r.cached_tokens,
        "cost_usd": r.cost_usd,
        "latency_ms": r.latency_ms,
        "tool_calls_made": r.tool_calls_made,
        "tool_rounds": r.tool_rounds,
        "n_execute_python": r.n_execute_python,
        "n_submit_answer": r.n_submit_answer,
        "finish_reason": r.finish_reason,
        "refusal_present": bool(r.refusal),
        "failure_bucket": r.failure_bucket,
        "forced_submit": r.forced_submit,
    }


def _wandb_summary(metrics: Any, wall_seconds: float) -> dict[str, Any]:
    """Per-cell summary written to wandb.summary."""
    flat: dict[str, Any] = {
        "accuracy": metrics.accuracy,
        "sat_accuracy": metrics.sat_accuracy,
        "sol_accuracy": metrics.sol_accuracy,
        "error_rate": metrics.error_rate,
        "wall_seconds": wall_seconds,
        **{f"tokens/{k}": v for k, v in metrics.tokens.items()},
        **{f"cost/{k}": v for k, v in metrics.cost.items()},
        **{f"timing/{k}": v for k, v in metrics.timing.items()},
        **{f"tool_use/{k}": v for k, v in metrics.tool_use.items()},
        **{f"hint_effect/{k}": v for k, v in metrics.hint_effect.items()},
        **{f"failure/{k}": v for k, v in metrics.failure_bucket_counts.items()},
    }
    for dim, label in [
        (metrics.by_problem_type, "by_type"),
        (metrics.by_solve_tier, "by_solve_tier"),
        (metrics.by_backend, "by_backend"),
        (metrics.by_hint_condition, "by_hint_condition"),
        (metrics.by_is_satisfiable, "by_is_satisfiable"),
    ]:
        for bucket, stats in dim.items():
            for stat_k, stat_v in stats.items():
                flat[f"{label}/{bucket}/{stat_k}"] = stat_v
    return flat


def _init_wandb(*, project: str, name: str, group: str | None,
                condition: str, model: str, config_extra: dict) -> Any:
    """Always init wandb. Entity, mode, etc. come from env vars (.env).
    Set WANDB_MODE=disabled to skip logging entirely."""
    import wandb
    return wandb.init(
        project=project,
        name=name,
        group=group,
        job_type="eval",
        tags=[condition, model.split("/")[0]],
        config={"model": model, "condition": condition, **config_extra},
        reinit=True,
    )


def cmd_eval(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    if args.problems:
        dataset = [d for d in dataset if d["problem_type"] in set(args.problems)]
    print(f"[eval] {args.model} | {'tools' if args.enable_tools else 'no_tools'} | "
          f"n={len(dataset)} workers={args.workers}")

    reasoning = _build_reasoning(args)
    provider_routing = _build_provider_routing(args)
    models_fallback = args.models_fallback

    wandb_run = _init_wandb(
        project=args.wandb_project,
        name=f"{_slug(args.model)}-{'tools' if args.enable_tools else 'no_tools'}",
        group=args.wandb_group,
        condition="tools" if args.enable_tools else "no_tools",
        model=args.model,
        config_extra={
            "dataset": str(args.dataset),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_tool_rounds": args.max_tool_rounds,
            "reasoning": reasoning,
            "provider_routing": provider_routing,
        },
    )

    out_path = Path(args.output) if args.output else Path(f"results/{_slug(args.model)}-{'tools' if args.enable_tools else 'no_tools'}.json")
    trace_dir = out_path.parent / "traces" / _slug(args.model) / ("tools" if args.enable_tools else "no_tools")
    trace_dir.mkdir(parents=True, exist_ok=True)
    results, wall = _evaluate_cell(
        model=args.model,
        dataset=dataset,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        enable_tools=args.enable_tools,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        max_tool_rounds=args.max_tool_rounds,
        reasoning=reasoning,
        provider_routing=provider_routing,
        models_fallback=models_fallback,
        workers=args.workers,
        wandb_run=wandb_run,
        trace_dir=trace_dir,
    )
    metrics = compute_metrics(results)

    wandb_run.summary.update(_wandb_summary(metrics, wall))
    wandb_run.finish()

    blob = {
        "model": args.model,
        "condition": "tools" if args.enable_tools else "no_tools",
        "dataset": str(args.dataset),
        "wall_seconds": round(wall, 2),
        "metrics": _metrics_to_dict(metrics),
        "results": [r.to_dict() for r in results],
    }
    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2, default=str))

    cost_total = metrics.cost.get("cost_total_usd", 0.0)
    print(f"[done] acc={metrics.accuracy:.3f} ({metrics.correct}/{metrics.total}) "
          f"cost=${cost_total:.4f} wall={wall:.1f}s -> {out}")
    return 0


def _metrics_to_dict(m: Any) -> dict[str, Any]:
    return {
        "total": m.total, "correct": m.correct,
        "accuracy": m.accuracy, "sat_accuracy": m.sat_accuracy,
        "sol_accuracy": m.sol_accuracy, "error_rate": m.error_rate,
        "by_problem_type": m.by_problem_type,
        "by_solve_tier": m.by_solve_tier,
        "by_backend": m.by_backend,
        "by_hint_condition": m.by_hint_condition,
        "by_is_satisfiable": m.by_is_satisfiable,
        "failure_bucket_counts": m.failure_bucket_counts,
        "tokens": m.tokens, "cost": m.cost, "timing": m.timing,
        "tool_use": m.tool_use, "hint_effect": m.hint_effect,
    }


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


def cmd_sweep(args: argparse.Namespace) -> int:
    cfg = yaml.safe_load(Path(args.models).read_text())
    defaults = cfg.get("defaults", {})
    models = cfg["models"]
    if args.models_filter:
        models = [m for m in models if m["id"] in set(args.models_filter)]
    conditions = cfg["conditions"]
    if args.conditions_filter:
        conditions = [c for c in conditions if c["name"] in set(args.conditions_filter)]
    models = [m for m in models if m.get("provider", "openrouter") == "openrouter"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = [(m, c, out_dir / f"{_slug(m['id'])}-{c['name']}.json")
             for m in models for c in conditions]

    if args.resume:
        cells = [(m, c, p) for (m, c, p) in cells if not p.exists()]
        print(f"[sweep] resume: {len(cells)} cells remaining")

    if not cells:
        print("[sweep] no cells to run")
        return 0

    print(f"[sweep] {len(cells)} cells, cell_parallelism={args.cell_parallelism}")

    def _run_cell(m: dict, c: dict, out_path: Path) -> tuple[str, str, int]:
        # Build a fresh argparse.Namespace per cell, reusing cmd_eval.
        ns = argparse.Namespace(
            model=m["id"],
            dataset=args.dataset,
            problems=args.problems,
            output=str(out_path),
            provider="openrouter",
            base_url=None,
            api_key=None,
            workers=defaults.get("workers", 4),
            max_tokens=defaults.get("max_tokens", 4096),
            temperature=defaults.get("temperature", 0.0),
            seed=defaults.get("seed"),
            enable_tools=bool(c.get("tools")),
            max_tool_rounds=defaults.get("max_tool_rounds", 5),
            reasoning_effort=(m.get("reasoning") or {}).get("effort"),
            reasoning_max_tokens=(m.get("reasoning") or {}).get("max_tokens"),
            reasoning_include=(m.get("reasoning") or {}).get("exclude") is False,
            models_fallback=None,
            wandb_project=args.wandb_project,
            wandb_group=args.wandb_group or out_dir.name,
        )
        try:
            cmd_eval(ns)
            return (m["id"], c["name"], 0)
        except Exception as e:
            print(f"  [fail] {m['id']} / {c['name']}: {e}")
            return (m["id"], c["name"], 1)

    completed, failed = 0, 0
    with ThreadPoolExecutor(max_workers=args.cell_parallelism) as pool:
        futs = [pool.submit(_run_cell, m, c, p) for m, c, p in cells]
        for fut in as_completed(futs):
            mid, cname, rc = fut.result()
            if rc == 0:
                completed += 1
                print(f"  [ok] {mid} / {cname}")
            else:
                failed += 1

    _write_costs_csv(out_dir, cells)
    print(f"[sweep] ok={completed} fail={failed}")
    return 0 if failed == 0 else 1


def _write_costs_csv(out_dir: Path, cells: list[tuple]) -> None:
    rows = []
    for m, c, path in cells:
        if not path.exists():
            continue
        blob = json.loads(path.read_text())
        rows.append({
            "model": m["id"],
            "condition": c["name"],
            "n": blob["metrics"]["total"],
            "accuracy": round(blob["metrics"]["accuracy"], 4),
            "cost_usd": round(blob["metrics"].get("cost", {}).get("cost_total_usd", 0.0), 4),
            "wall_seconds": blob.get("wall_seconds", 0.0),
        })
    if not rows:
        return
    with (out_dir / "costs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    total = sum(r["cost_usd"] for r in rows)
    print(f"[sweep] total_cost=${total:.2f} -> {out_dir / 'costs.csv'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser(prog="python -m math_constraint", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="generate dataset")
    pb.add_argument("--output", default="dataset")
    pb.add_argument("--seed", type=int, default=20260423)
    pb.add_argument("--samples-per-type", type=int, default=6)
    pb.add_argument("--workers", type=int, default=6)
    pb.add_argument("--problems", nargs="*", default=None)
    pb.add_argument("--sat-hint-fraction", type=float, default=0.5,
                    help="Fraction of SAT instances that receive partial-assignment hints.")
    pb.add_argument("--partial-assignment-fraction", type=float, default=0.2,
                    help="Fraction of each hinted solution revealed in the prompt.")
    pb.add_argument("--profile", default=None,
                    help="Dataset profile name or path:name (default path configs/dataset_profiles.yaml).")
    pb.add_argument("--stats", action="store_true")
    pb.add_argument("--verify", action="store_true")

    pv = sub.add_parser("verify", help="re-verify dataset")
    pv.add_argument("--dataset", default="dataset")
    pv.add_argument("--workers", type=int, default=6)

    ps = sub.add_parser("stats", help="print dataset stats")
    ps.add_argument("--dataset", default="dataset")

    pe = sub.add_parser("eval", help="single (model, condition) cell")
    pe.add_argument("--model", required=True)
    pe.add_argument("--dataset", required=True)
    pe.add_argument("--problems", nargs="*", default=None)
    pe.add_argument("--output", default=None)
    pe.add_argument("--provider", default="openrouter",
                    choices=["openrouter", "huggingface", "vllm", "generic"])
    pe.add_argument("--base-url", default=None)
    pe.add_argument("--api-key", default=None)
    pe.add_argument("--workers", type=int, default=4)
    pe.add_argument("--max-tokens", type=int, default=4096)
    pe.add_argument("--temperature", type=float, default=0.0)
    pe.add_argument("--seed", type=int, default=None)
    pe.add_argument("--enable-tools", action="store_true")
    pe.add_argument("--max-tool-rounds", type=int, default=5)
    pe.add_argument("--reasoning-effort", default=None,
                    choices=["minimal", "low", "medium", "high", "xhigh"])
    pe.add_argument("--reasoning-max-tokens", type=int, default=None)
    pe.add_argument("--reasoning-include", action="store_true")
    pe.add_argument("--models-fallback", nargs="*", default=None)
    pe.add_argument("--wandb-project", default="math-constraint")
    pe.add_argument("--wandb-group", default=None)

    pw = sub.add_parser("sweep", help="full sweep from configs/models.yaml")
    pw.add_argument("--models", default="configs/models.yaml")
    pw.add_argument("--dataset", default="dataset")
    pw.add_argument("--output-dir", required=True)
    pw.add_argument("--problems", nargs="*", default=None,
                    help="Filter to a subset of problem types (forwarded to each cell's eval).")
    pw.add_argument("--models-filter", nargs="*", default=None)
    pw.add_argument("--conditions-filter", nargs="*", default=None)
    pw.add_argument("--cell-parallelism", type=int, default=2)
    pw.add_argument("--resume", action="store_true")
    pw.add_argument("--wandb-project", default="math-constraint")
    pw.add_argument("--wandb-group", default=None)

    args = ap.parse_args()
    if args.cmd == "build":
        return cmd_build(args)
    if args.cmd == "verify":
        return cmd_verify(args)
    if args.cmd == "stats":
        return cmd_stats(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    if args.cmd == "sweep":
        return cmd_sweep(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

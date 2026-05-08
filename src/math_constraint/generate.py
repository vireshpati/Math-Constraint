"""Dataset generator: sample params, solve, write instances.

One unique problem becomes one instance. Hint condition is assigned
deterministically by a seed-driven 50/50 split across SAT instances. No
paired _h/_nh expansion. UNSAT instances are unhinted singletons.

Pipeline:
    sample_unique_params  -- rejection sampling using problems.is_valid_params
    solve                 -- run csp_solver or sms_solver
    assign_hint           -- 50/50 across SAT pool, seed-deterministic
    write_instances       -- one JSON per instance + manifest + jsonl
    annotate_solve_tiers  -- post-hoc easy/medium/hard from solve_time_ms
"""

import hashlib
import json
import logging
import multiprocessing as mp
import random
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from math_constraint.config import ProblemInstance
from math_constraint.problems import (
    PROBLEMS,
    get_backend,
    get_param_domain,
    get_prompt,
    get_search_space,
    get_spec,
    list_problems,
    solve,
)

log = logging.getLogger(__name__)


PARTIAL_ASSIGNMENT_FRACTION = 0.20

_NON_INSTANCE_FILES = {"manifest.json", "problems.jsonl", "generation_config.json"}
_SKIP_DIRS = {"raw", "_raw"}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _normalize_domain(domain: tuple[int, int] | list[Any]) -> tuple[int, int] | list[Any]:
    if (
        isinstance(domain, list)
        and len(domain) == 2
        and all(isinstance(x, int) and not isinstance(x, bool) for x in domain)
    ):
        return (domain[0], domain[1])
    return domain


def _sample_param_value(rng: random.Random, domain: tuple[int, int] | list[Any]) -> Any:
    domain = _normalize_domain(domain)
    if isinstance(domain, tuple):
        low, high = domain
        return rng.randint(low, high)
    return rng.choice(domain)


def _sample_unique_params(
    problem_name: str,
    count: int,
    rng: random.Random,
    domain_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Rejection-sample N unique valid parameter sets.

    Uses each problem's optional ``is_valid_params`` predicate from problems.py
    to skip mathematically ill-defined combinations. Returns fewer than N
    only if the parameter domain is too small.
    """
    spec = get_spec(problem_name)
    overrides = (domain_overrides or {}).get(problem_name, {})
    domains = {
        k: _normalize_domain(overrides.get(k, get_param_domain(problem_name, k)))
        for k in sorted(spec["params"].keys())
    }
    keys = list(domains.keys())
    if not keys:
        return [{}]

    is_valid = spec.get("is_valid_params")
    out: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    max_attempts = max(1000, count * 200)

    for _ in range(max_attempts):
        if len(out) >= count:
            break
        cand = {k: _sample_param_value(rng, domains[k]) for k in keys}
        if is_valid is not None and not is_valid(cand):
            continue
        sig = tuple((k, cand[k]) for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(cand)

    if len(out) < count:
        log.warning(
            "Only %d/%d unique valid params for '%s' (domain too small)",
            len(out), count, problem_name,
        )
    return out


# ---------------------------------------------------------------------------
# Solving + writing one instance
# ---------------------------------------------------------------------------

def _instance_name(problem_type: str, params: dict[str, Any], idx: int, with_hint: bool, sat: bool) -> str:
    param_str = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    if param_str and len(param_str) > 120:
        param_str = "h" + hashlib.sha1(param_str.encode("utf-8")).hexdigest()[:12]
    base = f"{problem_type}_{param_str}__v{idx}" if param_str else f"{problem_type}__v{idx}"
    if not sat:
        return base
    return f"{base}_h" if with_hint else f"{base}_nh"


def _build_partial_assignment(
    instance_name: str,
    solution: dict[str, Any],
    fraction: float = PARTIAL_ASSIGNMENT_FRACTION,
) -> dict[str, Any] | None:
    seed = int(hashlib.sha256(instance_name.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    partial: dict[str, Any] = {}
    for var, value in solution.items():
        if not isinstance(value, list) or not value:
            continue
        if all(not isinstance(v, (list, tuple, dict)) for v in value):
            count = max(1, int(len(value) * fraction))
            indices = sorted(rng.sample(range(len(value)), min(len(value), count)))
            partial[var] = {str(i): value[i] for i in indices}
        elif all(isinstance(v, (list, tuple)) and len(v) == 2 for v in value):
            count = max(1, int(len(value) * fraction))
            sample = rng.sample(value, min(len(value), count))
            partial[var] = [list(edge) for edge in sample]
    return partial or None


def _add_partial_to_prompt(prompt: str, partial: dict[str, Any]) -> str:
    lines = ["", "Partial assignment (fixed values that must be respected):"]
    for var, value in partial.items():
        if isinstance(value, dict):
            fixed = ", ".join(
                f"{var}[{i}]={v}" for i, v in sorted(value.items(), key=lambda x: int(x[0]))
            )
            lines.append(f"- {fixed}")
        elif isinstance(value, list):
            edges = ", ".join(f"({u},{v})" for u, v in value)
            lines.append(f"- Known present {var}: {edges}")
    lines.append("Return a complete solution consistent with these fixed assignments.")
    return f"{prompt}\n" + "\n".join(lines)


def generate(
    name: str,
    idx: int = 0,
    with_hint: bool = False,
    partial_assignment_fraction: float = PARTIAL_ASSIGNMENT_FRACTION,
    **params,
) -> ProblemInstance:
    """Solve the problem instance and return a populated ProblemInstance."""
    result = solve(name, **params)
    sat = result.satisfiable
    inst_name = _instance_name(name, params, idx, with_hint, sat)
    backend = get_backend(name)
    # pysms returns num_variables=num_constraints=-1; the natural complexity
    # signal is the satisfying graph's edge count.
    num_edges = -1
    if backend == "pysms" and sat and isinstance(result.solution, dict):
        edges = result.solution.get("edges")
        if isinstance(edges, list):
            num_edges = len(edges)
    difficulty = {
        "solve_time_ms": round(result.solve_time_ms, 1),
        "search_space": get_search_space(name, **params),
        "num_variables": result.num_variables,
        "num_constraints": result.num_constraints,
        "num_edges": num_edges,
        "backend": backend,
    }
    partial = None
    prompt = get_prompt(name, params)
    if sat and with_hint and result.solution is not None:
        partial = _build_partial_assignment(
            inst_name,
            result.solution,
            fraction=partial_assignment_fraction,
        )
        if partial:
            prompt = _add_partial_to_prompt(prompt, partial)
    return ProblemInstance(
        name=inst_name,
        problem_type=name,
        params=params,
        prompt=prompt,
        satisfiable=sat,
        solution=result.solution if sat else None,
        difficulty=difficulty,
        partial_assignment=partial,
        artifacts=result.artifacts,
    )


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

def _solve_task(task: tuple[str, int, bool, float, dict[str, Any]]) -> dict[str, Any] | None:
    """Worker entrypoint. Returns the JSON-ready instance dict."""
    name, idx, with_hint, partial_assignment_fraction, params = task
    try:
        inst = generate(
            name,
            idx=idx,
            with_hint=with_hint,
            partial_assignment_fraction=partial_assignment_fraction,
            **params,
        )
    except Exception as exc:
        log.warning("solve failed for %s params=%s: %s", name, params, exc)
        return None
    return {
        "name": inst.name,
        "problem_type": inst.problem_type,
        "params": inst.params,
        "prompt": inst.prompt,
        "satisfiable": inst.satisfiable,
        "solution": inst.solution,
        "difficulty": inst.difficulty,
        "partial_assignment": inst.partial_assignment,
        "_backend": get_backend(name),
        "_artifacts": (
            {
                "model": inst.artifacts.model,
                "stdout": inst.artifacts.stdout,
                "stderr": inst.artifacts.stderr,
                "command": inst.artifacts.command,
            } if inst.artifacts else None
        ),
    }


def build_tasks(
    seed: int,
    samples_per_type: int | dict[str, int] = 6,
    problem_types: list[str] | None = None,
    domain_overrides: dict[str, dict[str, Any]] | None = None,
    partial_assignment_fraction: float = PARTIAL_ASSIGNMENT_FRACTION,
) -> list[tuple[str, int, bool, float, dict[str, Any]]]:
    """Sample params for every problem type and assign hint conditions.

    samples_per_type may be a single int (uniform) or a dict mapping
    problem_type -> count for per-type weighting.
    """
    rng = random.Random(seed)
    types = list_problems()
    if problem_types:
        types = [t for t in types if t in problem_types]
    tasks: list[tuple[str, int, bool, float, dict[str, Any]]] = []
    default_n = samples_per_type if isinstance(samples_per_type, int) else 6
    for ptype in types:
        n = samples_per_type.get(ptype, default_n) if isinstance(samples_per_type, dict) else samples_per_type
        if n <= 0:
            continue
        for i, params in enumerate(
            _sample_unique_params(ptype, n, rng, domain_overrides)
        ):
            # with_hint=False placeholder; flipped post-solve based on
            # seed-deterministic 50/50 split over SAT instances.
            tasks.append((ptype, i, False, partial_assignment_fraction, params))
    return tasks


def _annotate_solve_tiers(instances: list[dict[str, Any]]) -> dict[str, float]:
    """Add solve_tier, solve_pct_global, solve_pct_type. Returns p33/p66 ms."""
    def _ms(inst: dict[str, Any]) -> float:
        return float(inst.get("difficulty", {}).get("solve_time_ms") or 0.0)

    def _rank(values: list[float], target: float) -> float:
        if not values:
            return 0.0
        below = sum(1 for v in values if v < target)
        equal = sum(1 for v in values if v == target)
        return 100.0 * (below + 0.5 * equal) / len(values)

    all_ms = sorted(_ms(i) for i in instances)
    if len(all_ms) >= 3:
        q = statistics.quantiles(all_ms, n=3, method="inclusive")
        p33, p66 = q[0], q[1]
    else:
        p33 = p66 = 0.0

    by_type_ms: dict[str, list[float]] = defaultdict(list)
    for inst in instances:
        by_type_ms[inst["problem_type"]].append(_ms(inst))
    for v in by_type_ms.values():
        v.sort()

    for inst in instances:
        ms = _ms(inst)
        diff = inst.setdefault("difficulty", {})
        diff["solve_tier"] = "easy" if ms <= p33 else ("medium" if ms <= p66 else "hard")
        diff["solve_pct_global"] = round(_rank(all_ms, ms), 2)
        diff["solve_pct_type"] = round(_rank(by_type_ms[inst["problem_type"]], ms), 2)

    return {"p33_ms": round(p33, 2), "p66_ms": round(p66, 2)}


def run_build(
    output_dir: Path,
    seed: int,
    samples_per_type: int,
    workers: int,
    problem_types: list[str] | None = None,
    domain_overrides: dict[str, dict[str, Any]] | None = None,
    sat_hint_fraction: float = 0.5,
    partial_assignment_fraction: float = PARTIAL_ASSIGNMENT_FRACTION,
    profile_name: str | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full build pipeline. Returns the manifest dict."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    instances_dir = output_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(
        seed,
        samples_per_type,
        problem_types,
        domain_overrides,
        partial_assignment_fraction,
    )
    log.info("[build] %d tasks across %d types", len(tasks), len(set(t[0] for t in tasks)))

    # Solve in parallel. Stream payloads to a checkpoint file so a wall-time
    # kill or OOM doesn't lose all work.
    raw_instances: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "_checkpoint.jsonl"
    with checkpoint_path.open("w") as ckpt:
        if workers <= 1:
            for task in tasks:
                payload = _solve_task(task)
                if payload is not None:
                    raw_instances.append(payload)
                    ckpt.write(json.dumps({k: v for k, v in payload.items() if not k.startswith("_")}) + "\n")
                    ckpt.flush()
        else:
            with mp.Pool(processes=workers) as pool:
                for payload in pool.imap_unordered(_solve_task, tasks):
                    if payload is not None:
                        raw_instances.append(payload)
                        ckpt.write(json.dumps({k: v for k, v in payload.items() if not k.startswith("_")}) + "\n")
                        ckpt.flush()

    # Assign hint condition: deterministic 50/50 split across SAT instances.
    rng = random.Random(seed)
    sat_indices = [i for i, inst in enumerate(raw_instances) if inst["satisfiable"]]
    hint_count = int(len(sat_indices) * max(0.0, min(1.0, sat_hint_fraction)))
    hinted = set(rng.sample(sat_indices, k=hint_count)) if sat_indices else set()

    # Re-process SAT instances that need a hint (rebuild prompt + partial_assignment).
    finalized: list[dict[str, Any]] = []
    for i, payload in enumerate(raw_instances):
        if i in hinted:
            # Rebuild partial assignment + prompt with hint, regenerate name.
            inst = generate(
                payload["problem_type"],
                idx=int(payload["name"].split("__v")[-1].split("_")[0]),
                with_hint=True,
                partial_assignment_fraction=partial_assignment_fraction,
                **payload["params"],
            )
            payload = {
                **payload,
                "name": inst.name,
                "prompt": inst.prompt,
                "partial_assignment": inst.partial_assignment,
            }
        finalized.append(payload)
    finalized.sort(key=lambda x: x["name"])

    solve_thresholds = _annotate_solve_tiers(finalized)

    # Write instances + jsonl.
    jsonl_path = output_dir / "problems.jsonl"
    with jsonl_path.open("w") as jl:
        for inst in finalized:
            clean = {k: v for k, v in inst.items() if not k.startswith("_")}
            (instances_dir / f"{inst['name']}.json").write_text(json.dumps(clean, indent=2) + "\n")
            jl.write(json.dumps(clean) + "\n")

    # Write raw artifacts (gitignored).
    raw_dir = output_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for inst in finalized:
        artifacts = inst.get("_artifacts")
        if not artifacts:
            continue
        ext = ".xml" if inst.get("_backend") == "pycsp" else ".cnf"
        (raw_dir / f"{inst['name']}{ext}").write_text(artifacts["model"])
        (raw_dir / f"{inst['name']}.out.json").write_text(json.dumps({
            "stdout": artifacts["stdout"],
            "stderr": artifacts["stderr"],
            "command": artifacts["command"],
        }, indent=2))

    # Build manifest.
    manifest = _build_manifest(
        finalized,
        seed,
        samples_per_type,
        solve_thresholds,
        sat_hint_fraction=sat_hint_fraction,
        partial_assignment_fraction=partial_assignment_fraction,
        profile_name=profile_name,
        profile=profile,
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _build_manifest(
    instances: list[dict[str, Any]],
    seed: int,
    samples_per_type: int,
    solve_thresholds: dict[str, float],
    sat_hint_fraction: float = 0.5,
    partial_assignment_fraction: float = PARTIAL_ASSIGNMENT_FRACTION,
    profile_name: str | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for inst in instances:
        by_type[inst["problem_type"]].append(inst)

    sat_inst = sum(1 for i in instances if i["satisfiable"])
    unsat_inst = len(instances) - sat_inst
    hinted = sum(1 for i in instances if i["partial_assignment"] is not None)
    unhinted_sat = sat_inst - hinted

    def _tier_counts(items: list[dict[str, Any]], key: str) -> dict[str, int]:
        c: dict[str, int] = defaultdict(int)
        for x in items:
            c[x.get("difficulty", {}).get(key, "unknown")] += 1
        return {k: c.get(k, 0) for k in ("easy", "medium", "hard", "unknown")}

    types_meta = {}
    for ptype, items in sorted(by_type.items()):
        s = sum(1 for x in items if x["satisfiable"])
        types_meta[ptype] = {
            "backend": get_backend(ptype),
            "params": sorted(PROBLEMS.get(ptype, {}).get("params", {}).keys()),
            "instances": len(items),
            "satisfiable_instances": s,
            "unsatisfiable_instances": len(items) - s,
            "hinted_instances": sum(1 for x in items if x["partial_assignment"] is not None),
            "solve_tier_counts": _tier_counts(items, "solve_tier"),
        }

    return {
        "name": "MathConstraint",
        "version": "5.0",
        "description": (
            "Combinatorial constraint reasoning benchmark for LLMs. "
            "Solver-verified instances across 25 NP-hard problem types. "
            "One instance per unique problem; SAT instances split 50/50 "
            "into hinted/unhinted by deterministic seed. SAT/UNSAT mix is "
            "the natural distribution of each problem's parameter domain."
        ),
        "total_instances": len(instances),
        "satisfiable_instances": sat_inst,
        "unsatisfiable_instances": unsat_inst,
        "hint_distribution": {
            "hinted_sat": hinted,
            "unhinted_sat": unhinted_sat,
            "unsat": unsat_inst,
        },
        "sat_unsat_split": {
            "instances_sat_pct": round(100 * sat_inst / max(1, len(instances)), 2),
        },
        "difficulty_distribution": {
            "by_solve_tier": _tier_counts(instances, "solve_tier"),
        },
        "solve_tier_thresholds_ms": solve_thresholds,
        "total_problem_types": len(by_type),
        "generator": {
            "seed": seed,
            "samples_per_type": samples_per_type,
            "partial_assignment_fraction": partial_assignment_fraction,
            "sat_hint_fraction": sat_hint_fraction,
            "profile_name": profile_name,
            "profile": profile,
        },
        "problem_types": types_meta,
    }


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load instance JSONs from a dataset directory or single .json/.jsonl file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {p}")
    if p.is_file():
        if p.suffix == ".jsonl":
            return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        return [json.loads(p.read_text())]
    files = sorted(
        f for f in p.rglob("*.json")
        if f.name not in _NON_INSTANCE_FILES
        and not any(part in _SKIP_DIRS for part in f.relative_to(p).parts)
    )
    if not files:
        raise FileNotFoundError(f"No .json instance files in {p}")
    return [json.loads(f.read_text()) for f in files]

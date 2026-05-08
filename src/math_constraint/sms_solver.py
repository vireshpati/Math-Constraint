"""PySMS solver interface -- constraint tables, solve(), verify()."""

import ast
import os
import subprocess
import tempfile
import time
from itertools import combinations as _combinations
from pathlib import Path
from typing import Any

from pysms.graph_builder import GraphEncodingBuilder

from .log import get_logger, solver_timeout_seconds
from .config import ProblemResult, SolverArtifacts

log = get_logger(__name__)


# Patch pysms bug: minConnectivity() references undefined `args.connectivity_low`
def _minConnectivity_fixed(self, connectivity_low) -> None:
    assert self.n > connectivity_low
    V = self.V
    var_edge = self.var_edge
    reachable = {
        (v, t, I): self.id()
        for k in range(connectivity_low)
        for I in _combinations(sorted(set(V)), k)
        for v in set(V) - {min(set(V) - set(I))} - set(I)
        for t in V
    }
    reachable_via = {
        (v, w, t, I): self.id()
        for k in range(connectivity_low)
        for I in _combinations(sorted(set(V)), k)
        for v in set(V) - {min(set(V) - set(I))} - set(I)
        for t in V
        for w in set(V) - {min(set(V) - set(I)), v} - set(I)
    }

    def var_reachable(v, t, I):
        return reachable[(v, t, I)]

    def var_reachable_via(v, w, t, I):
        return reachable_via[(v, w, t, I)]

    for k in range(connectivity_low):
        for I in _combinations(sorted(set(V)), k):
            u = min(set(V) - set(I))
            for v in set(V) - {u} - set(I):
                for t in V:
                    if t == 0:
                        self.append([-var_edge(v, u), +var_reachable(v, 0, I)])
                        self.append([+var_edge(v, u), -var_reachable(v, 0, I)])
                    else:
                        self.append(
                            [-var_reachable(v, t, I), +var_reachable(v, t - 1, I)]
                            + [+var_reachable_via(v, w, t, I) for w in set(V) - set(I) - {v, u}]
                        )
                        self.append([+var_reachable(v, t, I), -var_reachable(v, t - 1, I)])
                        for w in set(V) - set(I) - {v, u}:
                            self.append([+var_reachable(v, t, I), -var_reachable_via(v, w, t, I)])
                            self.append([+var_reachable_via(v, w, t, I), -var_reachable(w, t - 1, I), -var_edge(w, v)])
                            self.append([-var_reachable_via(v, w, t, I), +var_reachable(w, t - 1, I)])
                            self.append([-var_reachable_via(v, w, t, I), +var_edge(w, v)])
                self.append([+var_reachable(v, max(V), I)])

GraphEncodingBuilder.minConnectivity = _minConnectivity_fixed

_SPECS: dict[str, dict[str, Any]] = {
    "pysms_min_degree": {
        "constraints": {"minDegree": "min_degree"},
    },
    "pysms_degree_bounds": {
        "constraints": {"minDegree": "min_degree", "maxDegree": "max_degree"},
    },
    "pysms_num_edges_bounds": {
        "constraints": {"numEdgesLow": "min_edges", "numEdgesUpp": "max_edges"},
    },
    "pysms_min_connectivity": {
        "constraints": {"minConnectivity": "min_connectivity"},
    },
    "pysms_min_girth": {
        "constraints": {"minGirth": "min_girth"},
    },
    "pysms_contains_cliques": {
        "constraints": {"contains_cliques": ["num_cliques", "clique_size"]},
    },
    "pysms_mtf": {
        "constraints": {"mtf": None},
    },
    "pysms_graph_builder": {
        "optional_constraints": {
            "minDegree": "delta_low",
            "maxDegree": "Delta_upp",
            "numEdgesLow": "num_edges_low",
            "numEdgesUpp": "num_edges_upp",
            "maxChromaticNumber": "max_chromatic_number",
        },
        "optional_raw_params": {
            "min-chromatic-number": "min_chromatic_number",
        },
    },
    "pysms_combined_graph": {
        "optional_constraints": {
            "minDegree": "min_degree",
            "maxDegree": "max_degree",
            "numEdgesLow": "min_edges",
            "numEdgesUpp": "max_edges",
            "maxClique": "max_clique",
            "maxIndependentSet": "max_independent_set",
            "maxChromaticNumber": "max_chromatic_number",
            "minGirth": "min_girth",
            "ckFree": "k",
            "minConnectivity": "min_connectivity",
        },
        "optional_raw_params": {
            "min-chromatic-number": "min_chromatic_number",
        },
        "flag_constraints": {
            "mtf": "maximal_triangle_free",
        },
        "multi_arg_constraints": {
            "contains_cliques": ["num_cliques", "clique_size"],
        },
    },
    # -- New compound types (non-trivial: all require edges) --
    "pysms_clique_coloring": {
        "constraints": {
            "maxClique": "max_clique",
            "maxChromaticNumber": "max_chromatic_number",
            "minDegree": "min_degree",
        },
    },
    "pysms_independent_connectivity": {
        "constraints": {
            "maxIndependentSet": "max_independent_set",
            "minConnectivity": "min_connectivity",
        },
    },
    "pysms_girth_degree": {
        "constraints": {
            "minGirth": "min_girth",
            "minDegree": "min_degree",
            "maxDegree": "max_degree",
        },
    },
    "pysms_chromatic_girth": {
        "constraints": {
            "maxChromaticNumber": "max_chromatic_number",
            "minGirth": "min_girth",
            "numEdgesLow": "min_edges",
        },
    },
    # New v3 types: structurally hard, fully CNF-encoded (Cadical-verifiable).
    "pysms_ramsey": {
        # No clique of size r AND no independent set of size s.
        # UNSAT iff vertices >= R(r, s); SAT iff vertices < R(r, s).
        # Encoded via maxClique(r-1) + maxIndependentSet(s-1).
        "constraints": {
            "maxClique": "clique_avoid_minus_one",
            "maxIndependentSet": "indset_avoid_minus_one",
        },
    },
    "pysms_diameter2critical": {
        # Diameter exactly 2 AND removing any edge increases diameter.
        "constraints": {"diameter2critical": None},
    },
}


def _derived_params(params: dict) -> dict:
    """Compute -1 derivations needed by Ramsey / color-critical specs."""
    out = dict(params)
    if "clique_avoid" in params:
        out["clique_avoid_minus_one"] = params["clique_avoid"] - 1
    if "indset_avoid" in params:
        out["indset_avoid_minus_one"] = params["indset_avoid"] - 1
    if "chromatic" in params:
        out["chromatic_minus_one"] = params["chromatic"] - 1
    return out


def apply_constraints(builder: Any, spec: dict, params: dict) -> None:
    """Apply constraints from spec to a GraphEncodingBuilder."""
    params = _derived_params(params)
    # Required single/multi-arg constraints
    for method, param_key in spec.get("constraints", {}).items():
        if param_key is None:
            getattr(builder, method)()
        elif isinstance(param_key, list):
            getattr(builder, method)(*(params[k] for k in param_key))
        else:
            getattr(builder, method)(params[param_key])

    # Required raw SMS params
    for sms_key, param_key in spec.get("raw_params", {}).items():
        builder.paramsSMS[sms_key] = params[param_key]

    # Optional single-arg constraints (skip if param is None)
    for method, param_key in spec.get("optional_constraints", {}).items():
        val = params.get(param_key)
        if val is not None:
            getattr(builder, method)(val)

    # Optional raw SMS params
    for sms_key, param_key in spec.get("optional_raw_params", {}).items():
        val = params.get(param_key)
        if val is not None:
            builder.paramsSMS[sms_key] = val

    # Boolean flag constraints (e.g., mtf)
    for method, param_key in spec.get("flag_constraints", {}).items():
        if params.get(param_key):
            getattr(builder, method)()

    # Optional multi-arg constraints (all args must be present)
    for method, param_keys in spec.get("multi_arg_constraints", {}).items():
        values = [params.get(k) for k in param_keys]
        if all(v is not None for v in values):
            getattr(builder, method)(*values)


def _parse_edges(solution: Any, vertices: int) -> tuple[list[tuple[int, int]] | None, str]:
    if not isinstance(solution, dict):
        return None, "Solution must be a dict with 'edges' key"
    edges = solution.get("edges")
    if edges is None:
        return None, "Solution must contain 'edges' key"
    if not isinstance(edges, list):
        return None, "Edges must be a list"
    parsed_edges = []
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            return None, f"Invalid edge format: {edge}"
        u, v = edge
        if not (0 <= u < vertices and 0 <= v < vertices):
            return None, f"Edge {edge} has vertices out of range [0, {vertices})"
        if u == v:
            return None, f"Self-loop found: {edge}"
        parsed_edges.append((min(u, v), max(u, v)))
    if len(parsed_edges) != len(set(parsed_edges)):
        return None, "Duplicate edges found"
    return parsed_edges, ""


def _parse_edges_output(output: str) -> list[tuple[int, int]] | None:
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            try:
                edges = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue
            if isinstance(edges, list):
                return edges
    return None


def _build_smsg_cmd(builder: Any, cnf_file: str, directed: bool = False) -> list[str]:
    cmd = ["smsg"]
    if directed:
        cmd.append("--directed")
    for param, value in builder.paramsSMS.items():
        if value == "":
            cmd.append(f"--{param}")
        else:
            cmd.extend([f"--{param}", str(value)])
    cmd.extend(["--dimacs", cnf_file])
    return cmd


def solve(problem_name: str, **params) -> ProblemResult:
    """Build CNF via GraphEncodingBuilder, run smsg, parse edges."""
    spec = _SPECS[problem_name]
    vertices = params.get("vertices")
    if vertices is None:
        raise RuntimeError(f"{problem_name} requires 'vertices'")

    directed = params.get("directed", False)
    multigraph = params.get("multi_graph")
    builder = GraphEncodingBuilder(vertices, directed=directed, multiGraph=multigraph)
    builder.DEBUG = False

    apply_constraints(builder, spec, params)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
        cnf_file = f.name
        builder.print_dimacs(f)

    try:
        cnf_content = Path(cnf_file).read_text()
        cmd = _build_smsg_cmd(builder, cnf_file, directed)
        cmd_str = " ".join(cmd)

        log.debug(f"[pysms] {problem_name}: {cmd_str}")
        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=solver_timeout_seconds(),
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Solver timeout for {problem_name}")
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"smsg binary not found on PATH: {exc}. "
                f"Install pysms-solver or load the cluster module."
            ) from exc
        solve_time_ms = (time.time() - start) * 1000

        if result.returncode != 0 and "unsat" not in result.stdout.lower():
            stderr_head = (result.stderr or "").strip().splitlines()[:3]
            raise RuntimeError(
                f"smsg exited with code {result.returncode} for {problem_name} "
                f"{params}: {' | '.join(stderr_head) or '(no stderr)'}"
            )

        output = result.stdout
        edges = _parse_edges_output(output)
        satisfiable = edges is not None
        solution = {"edges": edges} if edges is not None else None
        log.info(f"[pysms] {problem_name}: sat={satisfiable} edges={len(edges) if edges else 0} time={solve_time_ms:.1f}ms")

        artifacts = SolverArtifacts(
            model=cnf_content,
            stdout=result.stdout,
            stderr=result.stderr,
            command=cmd_str,
        )
    finally:
        if os.path.exists(cnf_file):
            os.remove(cnf_file)

    return ProblemResult(
        satisfiable=satisfiable,
        solution=solution,
        solve_time_ms=solve_time_ms,
        num_variables=-1,
        num_constraints=-1,
        artifacts=artifacts,
    )


def verify(problem_name: str, params: dict, solution: Any) -> tuple[bool, str]:
    """Re-run with edge variables fixed as unit clauses."""
    spec = _SPECS[problem_name]
    vertices = params.get("vertices", 0)
    edges, err = _parse_edges(solution, vertices)
    if edges is None:
        return False, err

    directed = params.get("directed", False)
    multigraph = params.get("multi_graph")
    builder = GraphEncodingBuilder(vertices, directed=directed, multiGraph=multigraph)
    builder.DEBUG = False

    apply_constraints(builder, spec, params)

    # Fix the submitted edges
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    for i in range(vertices):
        for j in range(i + 1, vertices):
            edge_var = builder.var_edge(i, j)
            if (i, j) in edge_set:
                builder.append([edge_var])
            else:
                builder.append([-edge_var])

    # Verify with a plain SAT solver instead of smsg. smsg enforces
    # canonical-form symmetry breaking which rejects valid (non-canonical) graphs.
    # We only need to check whether the constraint clauses are satisfiable
    # given the model's edge assignment.
    try:
        from pysat.solvers import Cadical153
        # GraphEncodingBuilder appends clauses to self.clauses (list of lists)
        # builder.clauses contains all constraint clauses + the unit clauses we added
        clauses = list(builder)  # GraphEncodingBuilder inherits from list
        with Cadical153(bootstrap_with=clauses) as solver:
            if solver.solve():
                return True, "Solution verified by SAT solver (constraints satisfied)"
            return False, "Solution violates constraints (SAT solver returned UNSAT)"
    except ImportError:
        return False, "pysat not available for verification"
    except Exception as e:
        return False, f"Verification error: {e}"

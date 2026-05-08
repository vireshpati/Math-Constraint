"""Evaluator: run an LLM on a CSP instance, parse the answer, verify via solver.

Strict mode: malformed tool calls or empty content count as failures. No
text-shaped tool-call recovery, no retry nudges, no truncated-JSON repair.

Tool roster (when ``--enable-tools``):
    execute_python(code, timeout_seconds): run code in an isolated subprocess
    submit_answer(satisfiable, solution, reasoning): final structured answer

Public API:
    EvalConfig       — per-cell knobs
    EvalResult       — per-instance record (all Stage-7 metrics)
    Evaluator        — runs one instance through the loop
    parse_response   — strict JSON / fenced JSON / brace-balanced fallback
    validate_answer  — solver-backed verification (re-runs solver)
    compute_metrics  — per-cell aggregates including failure-bucket counts
"""

import json
import ast
import os
import re
import sys
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from math_constraint.llm import LLMClient
from math_constraint.problems import get_backend, verify


PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "openrouter":  {"default_model": "anthropic/claude-opus-4.7"},
    "huggingface": {"default_model": "meta-llama/Llama-3.1-70B-Instruct"},
    "vllm":        {"default_model": "meta-llama/Llama-3.1-8B-Instruct",
                    "base_url": "http://localhost:8000/v1"},
    "generic":     {"default_model": "default",
                    "base_url": "http://localhost:8000/v1"},
}


# ---------------------------------------------------------------------------
# Config + result records
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    model: str
    provider: str = "openrouter"
    max_tokens: int = 4096
    temperature: float = 0.0
    base_url: str | None = None
    enable_tools: bool = False
    max_tool_rounds: int = 5
    reasoning: dict | None = None
    seed: int | None = None
    provider_routing: dict | None = None
    models_fallback: list[str] | None = None

    @classmethod
    def from_provider(cls, provider: str = "openrouter",
                      model: str | None = None, **overrides: Any) -> "EvalConfig":
        defaults = PROVIDER_DEFAULTS.get(provider, {})
        if model is None:
            model = defaults.get("default_model", "anthropic/claude-opus-4.7")
        return cls(
            model=model,
            provider=provider,
            base_url=overrides.pop("base_url", defaults.get("base_url")),
            **{k: v for k, v in overrides.items() if k in cls.__dataclass_fields__},
        )


@dataclass
class EvalResult:
    """Per-instance record. Schema mirrors Stage 7 of the plan exactly."""
    # identity
    problem_name: str
    problem_type: str
    params: dict[str, Any]
    difficulty: dict[str, Any]
    is_hinted: bool = False
    is_satisfiable: bool = False
    model: str = ""
    provider: str = ""
    condition: str = ""

    # model output
    submitted_satisfiable: bool | None = None
    submitted_solution: Any = None
    reasoning: str | None = None
    raw_response: str | None = None
    cot_trace: str | None = None
    finish_reason: str | None = None
    native_finish_reason: str | None = None
    refusal: str | None = None

    # correctness
    correct: bool = False
    satisfiability_correct: bool = False
    solution_correct: bool = False
    validation_details: str = ""

    # tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    # cost
    cost_usd: float = 0.0
    cost_source: str = "openrouter"

    # timing
    latency_ms: float = 0.0
    tool_dispatch_ms_total: float = 0.0

    # tool use
    tool_calls_made: int = 0
    tool_rounds: int = 0
    n_execute_python: int = 0
    n_submit_answer: int = 0
    tool_call_chain: list[dict[str, Any]] = field(default_factory=list)

    # failure
    error: str | None = None
    failure_bucket: str = "none"
    forced_submit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalResult":
        # Filter to known fields, tolerate extras / missing
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_response(text: str | None) -> dict[str, Any] | None:
    """Strict JSON → fenced JSON → brace-balanced extraction. None = no parse."""
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    fences = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    for cand in reversed(fences):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    start = text.find('"satisfiable"')
    if start == -1:
        return None
    depth, i = 0, start
    while i >= 0 and not (text[i] == "{" and depth == 0):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
        i -= 1
    if i < 0:
        return None
    depth = 1
    for j in range(i + 1, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        if depth == 0:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                return None
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _primary_var(problem: dict) -> str:
    """Single source of truth for the primary variable name:
        pysms → "edges"
        pycsp → the first VarArray name in the model code
    """
    backend = get_backend(problem["problem_type"])
    if backend == "pysms":
        return "edges"
    from .csp_solver import _MODELS, _extract_variable_name
    spec = _MODELS.get(problem["problem_type"])
    if spec is None:
        return ""
    return _extract_variable_name(spec["model"](**problem["params"])) or ""


def _normalize_solution(problem: dict, solution: Any) -> Any:
    """Bring a submitted solution to canonical dict form keyed by primary var.

    Generous parsing handles all the shapes models actually emit:
        - JSON-encoded string (qwen)
        - already-correct dict (e.g., {"edges": [[u,v],...]})
        - bare list (most common; prompt says "return a list of ...")
        - 2D nested list for grid problems (flattened row-major)
        - flat-int list of edges (re-paired into [u,v] pairs for pysms)
    Returns a dict {primary_var: list-of-values}. Never alters underlying
    values or guesses intent.
    """
    if isinstance(solution, str):
        try:
            solution = json.loads(solution)
        except (json.JSONDecodeError, ValueError):
            return solution

    primary = _primary_var(problem)
    backend = get_backend(problem["problem_type"])

    # Already a dict
    if isinstance(solution, dict):
        if primary and primary in solution:
            values = solution[primary]
        elif len(solution) == 1:
            values = next(iter(solution.values()))
        else:
            return solution  # multi-key dict — pass through (verify handles)
    elif isinstance(solution, list):
        values = solution
    else:
        return solution

    if not isinstance(values, list):
        return solution

    # Per-backend shape coercion on the values list
    if backend == "pycsp":
        if values and isinstance(values[0], list):
            values = [v for row in values for v in row]  # flatten 2D
    elif backend == "pysms":
        if (values and isinstance(values[0], int)
                and len(values) % 2 == 0):
            values = [[values[i], values[i + 1]]
                      for i in range(0, len(values), 2)]

    return {primary: values} if primary else values


def _check_partial_assignment(problem: dict, submitted: Any) -> tuple[bool, str]:
    """Verify partial-assignment hints against the normalized dict submission.

    Skips any partial var the submission didn't supply — derived variables
    (e.g., graceful_graph's 'd' computed from 'x') are re-derived by the
    solver during the verify step and checked there.
    """
    partial = problem.get("partial_assignment")
    if not isinstance(partial, dict) or not partial:
        return True, ""
    if not isinstance(submitted, dict):
        return False, "Submitted not a dict after normalization"
    for var, fixed in partial.items():
        values = submitted.get(var)
        if values is None:
            continue  # derived var or not requested; verify will catch any inconsistency
        if isinstance(fixed, dict):
            if isinstance(values, dict):
                for idx, expected in fixed.items():
                    if values.get(idx) != expected:
                        return False, f"Partial mismatch: {var}[{idx}]={expected} required"
            elif isinstance(values, list):
                for idx, expected in fixed.items():
                    i = int(idx)
                    if not (0 <= i < len(values)) or values[i] != expected:
                        return False, f"Partial mismatch: {var}[{idx}]={expected} required"
            else:
                return False, f"Unsupported value shape for '{var}'"
        elif isinstance(fixed, list):
            if not isinstance(values, list):
                return False, f"Unsupported list partial for '{var}'"
            for needed in fixed:
                if needed not in values:
                    return False, f"Partial mismatch: expected {var} to contain {needed}"
    return True, ""


def validate_answer(problem: dict, sat: bool | None, solution: Any) -> dict[str, Any]:
    expected = problem["satisfiable"]
    out = {"correct": False, "satisfiability_correct": sat == expected,
           "solution_correct": False, "details": ""}
    if sat != expected:
        out["details"] = f"claimed {'SAT' if sat else 'UNSAT'}, expected {'SAT' if expected else 'UNSAT'}"
        return out
    if not expected:
        out["correct"] = True
        out["solution_correct"] = True
        out["details"] = "Correct UNSAT"
        return out
    if solution is None:
        out["details"] = "Claimed SAT but no solution provided"
        return out
    solution = _normalize_solution(problem, solution)
    ok, msg = _check_partial_assignment(problem, solution)
    if not ok:
        out["details"] = msg
        return out
    try:
        valid, details = verify(problem["problem_type"], problem["params"], solution)
    except KeyError:
        valid, details = False, f"Unknown problem type: {problem['problem_type']}"
    except Exception as e:
        valid, details = False, f"Verification error: {e}"
    out["solution_correct"] = valid
    out["correct"] = valid
    out["details"] = details
    return out


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You are solving constraint satisfaction problems. Analyze the problem carefully and determine:
1. Whether the problem is satisfiable (SAT) or unsatisfiable (UNSAT)
2. If satisfiable, provide a valid solution

Respond with a JSON object in the following format:
{
    "satisfiable": true/false,
    "solution": [list of values] or null if unsatisfiable,
    "reasoning": "Brief explanation of your reasoning"
}

Important:
- For satisfiability, true means a solution exists, false means it's mathematically impossible
- Solutions should be provided as a list of integers
- Be precise with your answer - incorrect solutions will be rejected"""

TOOL_SYSTEM_PROMPT_APPENDIX = """
You may optionally call tools before submitting your final answer:
- execute_python(code, timeout_seconds): run Python code in an isolated
  subprocess. The interpreter has the standard library plus generic solver
  packages when installed (pysat, z3, pycosat). It cannot import this benchmark's
  implementation packages (math_constraint, pycsp3, pysms), access the network,
  or spawn subprocesses. Default timeout 15s, max 60s.
- submit_answer(satisfiable, solution, reasoning): submit your final answer.

If you use tools, always finish by calling submit_answer.
"""

EVAL_TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "execute_python",
        "description": "Execute Python in an isolated subprocess; return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "timeout_seconds": {"type": "integer"},
            },
            "required": ["code"],
        },
    }},
    {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Submit your final SAT/UNSAT decision and (if SAT) solution.",
        "parameters": {
            "type": "object",
            "properties": {
                "satisfiable": {"type": "boolean"},
                "solution": {"type": ["array", "null"], "items": {"type": "integer"}},
                "reasoning": {"type": "string"},
            },
            "required": ["satisfiable"],
        },
    }},
]


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_VENV_PYTHON = REPO_ROOT / ".tool-venv" / "bin" / "python"

_TOOL_IMPORT_ALLOWLIST = {
    "bisect",
    "collections",
    "functools",
    "heapq",
    "itertools",
    "json",
    "math",
    "pycosat",
    "pysat",
    "random",
    "re",
    "statistics",
    "sys",
    "time",
    "z3",
}
_DENIED_TOOL_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}


def _tool_python_executable() -> str:
    configured = os.environ.get("MCNST_TOOL_PYTHON")
    if configured:
        return configured
    if TOOL_VENV_PYTHON.exists():
        return str(TOOL_VENV_PYTHON)
    return sys.executable


def _tool_env(tmp: str) -> dict[str, str]:
    keep = {"HOME", "LANG", "LC_ALL", "PATH", "TMPDIR"}
    env = {k: v for k, v in os.environ.items() if k in keep and v}
    env["PYTHONNOUSERSITE"] = "1"
    env["TMPDIR"] = tmp
    return env


def _validate_tool_code(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax_error: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in _TOOL_IMPORT_ALLOWLIST:
                    return f"import_not_allowed: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            if root not in _TOOL_IMPORT_ALLOWLIST:
                return f"import_not_allowed: {module or '<relative>'}"
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _DENIED_TOOL_CALLS:
                return f"call_not_allowed: {fn.id}"
            if isinstance(fn, ast.Attribute) and fn.attr in _DENIED_TOOL_CALLS:
                return f"call_not_allowed: {fn.attr}"
    return None


def _execute_python(code: str, timeout_seconds: int = 15, max_output_chars: int = 4000) -> dict[str, Any]:
    try:
        timeout = max(1, min(int(timeout_seconds), 60))
    except (TypeError, ValueError):
        timeout = 15
    validation_error = _validate_tool_code(code)
    if validation_error:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": validation_error,
            "timed_out": False,
            "blocked": True,
        }
    try:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "tool.py"
            script.write_text(code)
            proc = subprocess.run(
                [_tool_python_executable(), "-I", str(script)], cwd=tmp,
                capture_output=True, text=True, timeout=timeout, env=_tool_env(tmp),
            )
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": (proc.stdout or "")[:max_output_chars],
                "stderr": (proc.stderr or "")[:max_output_chars],
                "timed_out": False,
            }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False, "returncode": None,
            "stdout": (e.stdout or "")[:max_output_chars] if isinstance(e.stdout, str) else "",
            "stderr": (e.stderr or "")[:max_output_chars] if isinstance(e.stderr, str) else "",
            "timed_out": True, "error": f"timeout {timeout}s",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# Failure bucketing (Stage 7)
# ---------------------------------------------------------------------------


def _failure_bucket(r: EvalResult, hit_max_rounds: bool, parse_failed: bool, exception: bool) -> str:
    if exception:
        return "api_error"
    if r.finish_reason == "length":
        return "length_cutoff"
    if r.finish_reason == "content_filter" or r.refusal:
        return "content_filter"
    if hit_max_rounds and r.submitted_satisfiable is None:
        return "max_rounds"
    if parse_failed:
        return "parse_failure"
    if not r.satisfiability_correct:
        return "wrong_polarity"
    if r.is_satisfiable and not r.solution_correct:
        return "wrong_solution"
    return "none"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


# Optional callback signature: fn(result_dict, step_index)
WandbHook = Callable[[dict[str, Any], int], None] | None


class Evaluator:
    """Run an LLM through one constraint instance and record all metrics."""

    def __init__(self, config: EvalConfig, client: LLMClient, provider_name: str = ""):
        self.config = config
        self.client = client
        self.provider_name = provider_name or config.provider

    def evaluate(self, problem: dict) -> EvalResult:
        cfg = self.config
        condition = "tools" if cfg.enable_tools else "no_tools"
        result = EvalResult(
            problem_name=problem["name"],
            problem_type=problem["problem_type"],
            params=problem["params"],
            difficulty=problem.get("difficulty", {}),
            is_hinted=problem.get("partial_assignment") is not None,
            is_satisfiable=problem["satisfiable"],
            model=cfg.model,
            provider=self.provider_name,
            condition=condition,
        )

        system_prompt = SYSTEM_PROMPT
        if cfg.enable_tools:
            system_prompt += "\n\n" + TOOL_SYSTEM_PROMPT_APPENDIX
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem["prompt"]},
        ]

        rounds_budget = cfg.max_tool_rounds if cfg.enable_tools else 1
        hit_max_rounds = False
        parse_failed = False
        had_exception = False
        t0 = time.time()

        try:
            for _ in range(rounds_budget):
                result.tool_rounds += 1
                resp = self.client.chat(
                    messages=messages,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    tools=EVAL_TOOLS if cfg.enable_tools else None,
                    response_format=None,
                    reasoning=cfg.reasoning,
                    seed=cfg.seed,
                )
                self._accumulate_usage(result, resp)

                if resp.tool_calls and cfg.enable_tools:
                    done = self._dispatch_tool_calls(result, messages, resp, problem)
                    if done:
                        break
                    continue

                # No tool calls: parse content.
                if resp.content is None:
                    parse_failed = True
                    break
                result.raw_response = resp.content
                content = parse_response(resp.content)
                if content is None:
                    parse_failed = True
                    break
                self._record_answer(result, content, problem)
                break
            else:
                hit_max_rounds = cfg.enable_tools

        except Exception as e:
            had_exception = True
            result.error = f"{type(e).__name__}: {e}"

        # Force a final submission if no answer was committed (max_rounds or parse fail).
        if (hit_max_rounds or parse_failed) and result.submitted_satisfiable is None and not had_exception:
            messages.append({"role": "user", "content":
                "Submit your best answer NOW as a JSON object: "
                '{"satisfiable": true/false, "solution": [...] or null, "reasoning": "..."}. '
                "Do not call any tools."})
            try:
                resp = self.client.chat(
                    messages=messages, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
                    tools=None, response_format=None, reasoning=cfg.reasoning, seed=cfg.seed)
                self._accumulate_usage(result, resp)
                if resp.content:
                    result.raw_response = resp.content
                    content = parse_response(resp.content)
                    if content is not None:
                        self._record_answer(result, content, problem)
                        result.forced_submit = True
                        parse_failed = False
            except Exception as e:
                result.error = (result.error or "") + f" | forced_submit_failed: {type(e).__name__}: {e}"

        result.latency_ms = (time.time() - t0) * 1000.0

        # Validate whenever we got an answer (including forced submissions).
        if not had_exception and result.submitted_satisfiable is not None:
            try:
                self._validate(result, problem)
            except Exception as e:
                result.error = (result.error or "") + f" | validate_failed: {type(e).__name__}: {e}"

        result.failure_bucket = _failure_bucket(result, hit_max_rounds, parse_failed, had_exception)
        if hit_max_rounds and not result.error:
            result.error = f"max_tool_rounds={cfg.max_tool_rounds} exceeded"
        if parse_failed and not result.error:
            result.error = f"parse_failure: {result.raw_response[:200] if result.raw_response else 'empty'}"

        return result

    def _accumulate_usage(self, r: EvalResult, resp: Any) -> None:
        r.prompt_tokens += resp.prompt_tokens
        r.completion_tokens += resp.completion_tokens
        if resp.reasoning_tokens:
            r.reasoning_tokens += resp.reasoning_tokens
        if resp.cached_tokens:
            r.cached_tokens += resp.cached_tokens
        if resp.cost is not None:
            r.cost_usd += resp.cost
        if resp.finish_reason:
            r.finish_reason = resp.finish_reason
        if resp.native_finish_reason:
            r.native_finish_reason = resp.native_finish_reason
        if resp.refusal:
            r.refusal = resp.refusal
        if resp.reasoning:
            r.cot_trace = (r.cot_trace + "\n---\n" + resp.reasoning) if r.cot_trace else resp.reasoning

    def _dispatch_tool_calls(self, r: EvalResult, messages: list[dict],
                             resp: Any, problem: dict) -> bool:
        """Run tool calls; append role:tool messages. Return True if submit_answer fired."""
        r.tool_calls_made += len(resp.tool_calls)
        # Echo assistant turn back into the conversation.
        messages.append({
            "role": "assistant",
            "content": resp.content or "",
            "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": tc["function"]} for tc in resp.tool_calls
            ],
        })
        for tc in resp.tool_calls:
            name = tc["function"]["name"]
            args = _parse_tool_arguments(tc["function"]["arguments"])
            t_tool = time.time()
            if name == "submit_answer":
                r.n_submit_answer += 1
                self._record_answer(r, args, problem)
                r.tool_call_chain.append({"name": name, "args": args, "ok": True,
                                          "latency_ms": round((time.time() - t_tool) * 1000, 2)})
                return True
            elif name == "execute_python":
                r.n_execute_python += 1
                output = _execute_python(str(args.get("code", "")),
                                         timeout_seconds=args.get("timeout_seconds", 15))
            else:
                output = {"ok": False, "error": f"unknown tool: {name}"}
            tool_ms = (time.time() - t_tool) * 1000
            r.tool_dispatch_ms_total += tool_ms
            r.tool_call_chain.append({
                "name": name, "args": args, "ok": bool(output.get("ok")),
                "latency_ms": round(tool_ms, 2),
            })
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(output)})
        return False

    def _record_answer(self, r: EvalResult, content: dict, problem: dict) -> None:
        sat = content.get("satisfiable")
        if isinstance(sat, bool):
            r.submitted_satisfiable = sat
        elif sat is not None:
            r.submitted_satisfiable = str(sat).lower() in ("true", "1", "yes")
        r.submitted_solution = content.get("solution")
        r.reasoning = content.get("reasoning")

    def _validate(self, r: EvalResult, problem: dict) -> None:
        v = validate_answer(problem, r.submitted_satisfiable, r.submitted_solution)
        r.correct = v["correct"]
        r.satisfiability_correct = v["satisfiability_correct"]
        r.solution_correct = v["solution_correct"]
        r.validation_details = v["details"]


# ---------------------------------------------------------------------------
# Per-cell aggregation
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkMetrics:
    """Per-cell aggregates. Stage 7 of the plan defines every field."""
    total: int
    correct: int
    accuracy: float
    sat_accuracy: float
    sol_accuracy: float
    error_rate: float
    by_problem_type: dict[str, dict[str, float]] = field(default_factory=dict)
    by_solve_tier: dict[str, dict[str, float]] = field(default_factory=dict)
    by_backend: dict[str, dict[str, float]] = field(default_factory=dict)
    by_hint_condition: dict[str, dict[str, float]] = field(default_factory=dict)
    by_is_satisfiable: dict[str, dict[str, float]] = field(default_factory=dict)
    failure_bucket_counts: dict[str, int] = field(default_factory=dict)
    tokens: dict[str, float] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    tool_use: dict[str, float] = field(default_factory=dict)
    hint_effect: dict[str, float] = field(default_factory=dict)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _strat(results: list[EvalResult], key: Callable[[EvalResult], str]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        groups[key(r)].append(r)
    out = {}
    for g, items in groups.items():
        n = len(items)
        out[g] = {
            "n": n,
            "accuracy": sum(1 for r in items if r.correct) / n,
            "sat_accuracy": sum(1 for r in items if r.satisfiability_correct) / n,
            "sol_accuracy": sum(1 for r in items if r.solution_correct) / n,
        }
    return out


def compute_metrics(results: list[EvalResult]) -> BenchmarkMetrics:
    n = len(results)
    if n == 0:
        return BenchmarkMetrics(0, 0, 0, 0, 0, 0)

    correct = sum(1 for r in results if r.correct)
    sat_correct = sum(1 for r in results if r.satisfiability_correct)
    sol_correct = sum(1 for r in results if r.solution_correct)
    errors = sum(1 for r in results if r.error)

    bucket_counts: dict[str, int] = defaultdict(int)
    for r in results:
        bucket_counts[r.failure_bucket] += 1

    pt = sum(r.prompt_tokens for r in results)
    ct = sum(r.completion_tokens for r in results)
    rt = sum(r.reasoning_tokens for r in results)
    cached = sum(r.cached_tokens for r in results)
    tokens = {
        "prompt_tokens_total": pt,
        "completion_tokens_total": ct,
        "reasoning_tokens_total": rt,
        "cached_tokens_total": cached,
        "total_tokens": pt + ct,
        "cache_hit_rate": cached / pt if pt else 0.0,
        "reasoning_token_ratio": rt / ct if ct else 0.0,
    }

    cost_total = sum(r.cost_usd for r in results)
    cost = {
        "cost_total_usd": cost_total,
        "cost_per_instance_usd": cost_total / n,
        "cost_per_correct_usd": (cost_total / correct) if correct else float("inf"),
    }

    latencies = [r.latency_ms for r in results]
    timing = {
        "avg_latency_ms": sum(latencies) / n,
        "median_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "p99_latency_ms": _percentile(latencies, 99),
    }

    tool_calls = [r.tool_calls_made for r in results]
    tool_use = {
        "tool_use_rate": sum(1 for x in tool_calls if x > 0) / n,
        "avg_tool_calls_per_instance": sum(tool_calls) / n,
        "avg_tool_rounds": sum(r.tool_rounds for r in results) / n,
        "total_execute_python_calls": sum(r.n_execute_python for r in results),
        "total_submit_answer_calls": sum(r.n_submit_answer for r in results),
        "submit_answer_rate": sum(1 for r in results if r.n_submit_answer > 0) / n,
    }

    sat_results = [r for r in results if r.is_satisfiable]
    hinted = [r for r in sat_results if r.is_hinted]
    unhinted = [r for r in sat_results if not r.is_hinted]
    h_acc = (sum(1 for r in hinted if r.correct) / len(hinted)) if hinted else 0.0
    u_acc = (sum(1 for r in unhinted if r.correct) / len(unhinted)) if unhinted else 0.0
    hint_effect = {
        "hinted_accuracy": h_acc,
        "unhinted_accuracy": u_acc,
        "hint_delta": h_acc - u_acc,
        "n_hinted": len(hinted),
        "n_unhinted": len(unhinted),
    }

    by_type = _strat(results, lambda r: r.problem_type)
    by_solve_tier = _strat(results, lambda r: r.difficulty.get("solve_tier", "unknown"))
    def _safe_backend(r: EvalResult) -> str:
        try:
            return get_backend(r.problem_type)
        except KeyError:
            return "unknown"
    by_backend = _strat(results, _safe_backend)
    by_hint = _strat(results, lambda r: "hinted" if r.is_hinted else "unhinted")
    by_sat = _strat(results, lambda r: "sat" if r.is_satisfiable else "unsat")

    return BenchmarkMetrics(
        total=n,
        correct=correct,
        accuracy=correct / n,
        sat_accuracy=sat_correct / n,
        sol_accuracy=sol_correct / n,
        error_rate=errors / n,
        by_problem_type=by_type,
        by_solve_tier=by_solve_tier,
        by_backend=by_backend,
        by_hint_condition=by_hint,
        by_is_satisfiable=by_sat,
        failure_bucket_counts=dict(bucket_counts),
        tokens=tokens,
        cost=cost,
        timing=timing,
        tool_use=tool_use,
        hint_effect=hint_effect,
    )

"""End-to-end smoke tests for the build + eval pipeline.

Uses an in-process stub LLMClient so no API calls are made.
"""

import json
from pathlib import Path

import pytest

from math_constraint.eval import (
    EvalConfig,
    Evaluator,
    compute_metrics,
    parse_response,
    validate_answer,
    _failure_bucket,
    _execute_python,
)
from math_constraint.generate import (
    _annotate_solve_tiers,
    _sample_unique_params,
    build_tasks,
    load_dataset,
)
from math_constraint.llm import LLMResponse


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_response_strict_json():
    text = '{"satisfiable": true, "solution": [1, 2, 3], "reasoning": "x"}'
    assert parse_response(text) == {
        "satisfiable": True, "solution": [1, 2, 3], "reasoning": "x",
    }


def test_parse_response_markdown_fence():
    text = 'preamble\n```json\n{"satisfiable": false}\n```\nepilogue'
    assert parse_response(text) == {"satisfiable": False}


def test_parse_response_brace_balanced_fallback():
    text = 'reasoning\n{"satisfiable": true, "solution": null, "reasoning": "x"}'
    out = parse_response(text)
    assert out is not None and out["satisfiable"] is True


def test_parse_response_returns_none_on_garbage():
    assert parse_response("no json anywhere") is None
    assert parse_response("") is None
    assert parse_response(None) is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_unsat_correctly_identified():
    problem = {
        "problem_type": "pigeons",
        "params": {"n": 4},
        "satisfiable": False,
    }
    out = validate_answer(problem, sat=False, solution=None)
    assert out["correct"] is True
    assert out["satisfiability_correct"] is True
    assert "Correct UNSAT" in out["details"]


def test_validate_wrong_polarity():
    problem = {
        "problem_type": "queens",
        "params": {"n": 5},
        "satisfiable": True,
    }
    out = validate_answer(problem, sat=False, solution=None)
    assert out["correct"] is False
    assert out["satisfiability_correct"] is False
    assert "claimed" in out["details"]


# ---------------------------------------------------------------------------
# Sampling (rejection)
# ---------------------------------------------------------------------------


import random as _random


def test_sample_unique_params_returns_valid():
    rng = _random.Random(42)
    params = _sample_unique_params("queens", 4, rng)
    assert len(params) == 4
    assert len({tuple(p.items()) for p in params}) == 4
    for p in params:
        assert 2 <= p["n"] <= 27


def test_sample_unique_params_rejects_invalid():
    """For pysms_min_degree, _is_valid_graph_params should reject min_degree > V-1."""
    rng = _random.Random(0)
    params = _sample_unique_params("pysms_min_degree", 8, rng)
    for p in params:
        assert p["min_degree"] <= p["vertices"] - 1


def test_build_tasks_distributes_across_types():
    tasks = build_tasks(seed=0, samples_per_type=2, problem_types=["queens", "pigeons"])
    types = {t[0] for t in tasks}
    assert types == {"queens", "pigeons"}
    assert len(tasks) == 4


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def test_annotate_solve_tiers_labels_p33_p66():
    instances = [
        {"problem_type": "x", "difficulty": {"solve_time_ms": ms}}
        for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ]
    thresholds = _annotate_solve_tiers(instances)
    assert thresholds["p33_ms"] < thresholds["p66_ms"]
    tiers = [i["difficulty"]["solve_tier"] for i in instances]
    assert tiers.count("easy") + tiers.count("medium") + tiers.count("hard") == 10
    # Slowest should be hard, fastest should be easy.
    assert instances[0]["difficulty"]["solve_tier"] == "easy"
    assert instances[-1]["difficulty"]["solve_tier"] == "hard"


# ---------------------------------------------------------------------------
# Failure bucketing
# ---------------------------------------------------------------------------


class _R:
    """Tiny stand-in for EvalResult for bucketing tests."""

    def __init__(self, **kw):
        self.finish_reason = kw.get("finish_reason")
        self.refusal = kw.get("refusal")
        self.satisfiability_correct = kw.get("satisfiability_correct", True)
        self.is_satisfiable = kw.get("is_satisfiable", False)
        self.solution_correct = kw.get("solution_correct", True)
        self.submitted_satisfiable = kw.get("submitted_satisfiable")


def test_failure_bucket_priority():
    assert _failure_bucket(_R(), False, False, True) == "api_error"
    assert _failure_bucket(_R(finish_reason="length"), False, False, False) == "length_cutoff"
    assert _failure_bucket(_R(refusal="x"), False, False, False) == "content_filter"
    assert _failure_bucket(_R(), True, False, False) == "max_rounds"
    assert _failure_bucket(_R(submitted_satisfiable=True), True, False, False) == "none"
    assert _failure_bucket(_R(), False, True, False) == "parse_failure"
    assert _failure_bucket(_R(satisfiability_correct=False), False, False, False) == "wrong_polarity"
    assert _failure_bucket(_R(is_satisfiable=True, solution_correct=False),
                           False, False, False) == "wrong_solution"
    assert _failure_bucket(_R(), False, False, False) == "none"


# ---------------------------------------------------------------------------
# Tool sandbox
# ---------------------------------------------------------------------------


def test_execute_python_blocks_benchmark_imports():
    for module in ("math_constraint", "pycsp3", "pysms"):
        out = _execute_python(f"import {module}")
        assert out["ok"] is False
        assert out.get("blocked") is True
        assert "import_not_allowed" in out["stderr"]


def test_execute_python_blocks_escape_hatches():
    for code in ("import socket", "import subprocess", "open('/etc/passwd').read()"):
        out = _execute_python(code)
        assert out["ok"] is False
        assert out.get("blocked") is True


def test_execute_python_allows_generic_solver_packages_when_installed():
    checks = [
        "from pysat.solvers import Glucose3\nprint('pysat-ok')",
        "from z3 import Solver\nprint('z3-ok')",
        "import pycosat\nprint('pycosat-ok')",
    ]
    for code in checks:
        out = _execute_python(code)
        if not out["ok"] and "ModuleNotFoundError" in out.get("stderr", ""):
            pytest.skip("generic solver packages are not installed in the tool environment")
        assert out["ok"] is True


# ---------------------------------------------------------------------------
# Evaluator with a stub client
# ---------------------------------------------------------------------------


class StubClient:
    """Minimal LLMClient stand-in: returns a canned LLMResponse."""

    def __init__(self, response: LLMResponse):
        self._response = response

    def chat(self, **_kwargs):
        return self._response


def test_evaluator_correct_unsat_no_tools():
    problem = {
        "name": "pigeons_n4__v0",
        "problem_type": "pigeons",
        "params": {"n": 4},
        "prompt": "Place 4 pigeons in 3 holes...",
        "satisfiable": False,
        "solution": None,
        "difficulty": {"solve_time_ms": 1.0, "solve_tier": "easy"},
        "partial_assignment": None,
    }
    response = LLMResponse(
        content='{"satisfiable": false, "solution": null, "reasoning": "pigeonhole"}',
        tool_calls=None, prompt_tokens=10, completion_tokens=20,
        finish_reason="stop", cost=0.001,
    )
    cfg = EvalConfig.from_provider("openrouter", "test-model")
    r = Evaluator(cfg, StubClient(response)).evaluate(problem)
    assert r.correct is True
    assert r.satisfiability_correct is True
    assert r.failure_bucket == "none"
    assert r.cost_usd == 0.001
    assert r.prompt_tokens == 10
    assert r.completion_tokens == 20


def test_evaluator_records_parse_failure():
    problem = {
        "name": "x", "problem_type": "queens", "params": {"n": 5},
        "prompt": "p", "satisfiable": True, "solution": [0, 2, 4, 1, 3],
        "difficulty": {"solve_time_ms": 1.0}, "partial_assignment": None,
    }
    response = LLMResponse(
        content="not json at all", tool_calls=None,
        prompt_tokens=5, completion_tokens=5, finish_reason="stop",
    )
    cfg = EvalConfig.from_provider("openrouter", "test-model")
    r = Evaluator(cfg, StubClient(response)).evaluate(problem)
    assert r.correct is False
    assert r.failure_bucket == "parse_failure"


# ---------------------------------------------------------------------------
# Metrics shape
# ---------------------------------------------------------------------------


def test_compute_metrics_full_schema():
    """Spot-check that compute_metrics returns every Stage-7 field family."""
    from math_constraint.eval import EvalResult
    results = [
        EvalResult(
            problem_name=f"p{i}", problem_type="queens",
            params={"n": 5}, difficulty={"solve_tier": "easy" if i < 2 else "hard"},
            is_hinted=(i % 2 == 0), is_satisfiable=True,
            correct=(i < 3), satisfiability_correct=(i < 3), solution_correct=(i < 3),
            prompt_tokens=10, completion_tokens=20, cost_usd=0.001 * (i + 1),
            latency_ms=100.0 * (i + 1), failure_bucket="none" if i < 3 else "wrong_solution",
        )
        for i in range(4)
    ]
    m = compute_metrics(results)
    assert m.total == 4
    assert m.correct == 3
    assert m.accuracy == 0.75
    assert m.tokens["prompt_tokens_total"] == 40
    assert m.tokens["total_tokens"] == 120
    assert m.cost["cost_total_usd"] == pytest.approx(0.010)
    assert m.cost["cost_per_correct_usd"] == pytest.approx(0.010 / 3)
    assert "easy" in m.by_solve_tier and "hard" in m.by_solve_tier
    assert m.failure_bucket_counts["none"] == 3
    assert m.failure_bucket_counts["wrong_solution"] == 1
    assert m.hint_effect["n_hinted"] == 2
    assert m.hint_effect["n_unhinted"] == 2


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_excludes_metadata(tmp_path):
    instances_dir = tmp_path / "instances"
    instances_dir.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"name": "X"}))
    (tmp_path / "problems.jsonl").write_text("garbage\n")
    (tmp_path / "generation_config.json").write_text(json.dumps({"seed": 0}))
    (instances_dir / "a.json").write_text(json.dumps({"name": "a", "problem_type": "x"}))
    (instances_dir / "b.json").write_text(json.dumps({"name": "b", "problem_type": "x"}))
    out = load_dataset(tmp_path)
    names = {d["name"] for d in out}
    assert names == {"a", "b"}

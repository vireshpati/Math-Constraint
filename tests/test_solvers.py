"""Solver correctness tests: small instances on representative problem types.

Marked slow because each test launches the actual ACE / smsg subprocess.
Run subset with: pytest tests/test_solvers.py -k "queens or pigeons"
"""

import os

import pytest

from math_constraint.eval import _execute_python
from math_constraint.problems import solve, verify


pytestmark = pytest.mark.skipif(
    os.environ.get("MCNST_SKIP_SOLVER_TESTS") == "1",
    reason="solver tests disabled via env",
)


# ---------------------------------------------------------------------------
# pycsp backend
# ---------------------------------------------------------------------------


def test_queens_n5_sat():
    r = solve("queens", n=5)
    assert r.satisfiable is True
    valid, _ = verify("queens", {"n": 5}, r.solution)
    assert valid is True


def test_queens_n3_unsat():
    r = solve("queens", n=3)
    assert r.satisfiable is False


def test_pigeons_unsat():
    r = solve("pigeons", n=4)
    assert r.satisfiable is False


def test_langford_n3_sat():
    """Langford's theorem: n=3 satisfies n mod 4 in {0,3}."""
    r = solve("langford", n=3)
    assert r.satisfiable is True


def test_langford_n5_unsat():
    """n=5 mod 4 = 1, not in {0,3}, so UNSAT."""
    r = solve("langford", n=5)
    assert r.satisfiable is False


# ---------------------------------------------------------------------------
# pysms backend
# ---------------------------------------------------------------------------


def test_pysms_min_degree_sat():
    r = solve("pysms_min_degree", vertices=8, min_degree=2)
    assert r.satisfiable is True


def test_pysms_min_degree_unsat_overconstrained():
    """min_degree > vertices-1 should be unreachable, but the solver handles it."""
    r = solve("pysms_min_degree", vertices=4, min_degree=4)
    assert r.satisfiable is False


# ---------------------------------------------------------------------------
# execute_python tool
# ---------------------------------------------------------------------------


def test_execute_python_basic():
    out = _execute_python("print(2 + 3)")
    assert out["ok"] is True
    assert out["stdout"].strip() == "5"


def test_execute_python_timeout():
    out = _execute_python("import time; time.sleep(10)", timeout_seconds=2)
    assert out["timed_out"] is True


def test_execute_python_traceback_captured():
    out = _execute_python("raise ValueError('boom')")
    assert out["ok"] is False
    assert "ValueError" in out["stderr"]
    assert "boom" in out["stderr"]

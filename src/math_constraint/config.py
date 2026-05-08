"""Shared dataclasses used by solvers + generator."""

from dataclasses import dataclass
from typing import Any


@dataclass
class SolverArtifacts:
    model: str
    stdout: str
    stderr: str
    command: str | None = None


@dataclass
class ProblemResult:
    satisfiable: bool
    solution: dict[str, Any] | None
    solve_time_ms: float = -1
    num_variables: int = -1
    num_constraints: int = -1
    artifacts: SolverArtifacts | None = None


@dataclass
class ProblemInstance:
    name: str
    problem_type: str
    params: dict[str, Any]
    prompt: str
    satisfiable: bool
    solution: dict[str, Any] | None
    difficulty: dict[str, Any]
    partial_assignment: dict[str, Any] | None = None
    artifacts: SolverArtifacts | None = None

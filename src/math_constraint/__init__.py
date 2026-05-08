"""Math-Constraint: CSP benchmark for LLM evaluation.

Public API: ``load_dataset(path)`` returns a list of instance dicts. For
running an LLM eval, see ``run.py eval`` or import :class:`math_constraint.eval.Evaluator`.
"""

from math_constraint.generate import load_dataset

__all__ = ["load_dataset"]

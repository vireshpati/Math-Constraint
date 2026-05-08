"""Logging configuration and shared utilities."""

import logging
import os


def get_logger(name: str = "math_constraint") -> logging.Logger:
    """Get a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = os.environ.get("MATH_CONSTRAINT_LOG_LEVEL", "INFO")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname).1s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.propagate = False
    return logger


def solver_timeout_seconds() -> float | None:
    """Read solver timeout from MCNST_SOLVER_TIMEOUT_SEC env var.

    Returns None (no timeout) when the variable is unset, empty, or <= 0.
    """
    raw = os.getenv("MCNST_SOLVER_TIMEOUT_SEC", "0").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    return None if value <= 0 else value

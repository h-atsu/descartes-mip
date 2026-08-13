"""Shared public typing contracts."""

from collections.abc import Callable

from pulp import LpSolver

type SolverFactory = Callable[[float | None], LpSolver]
"""Create an independent PuLP solver with an optional remaining-time limit."""

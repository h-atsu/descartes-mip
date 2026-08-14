"""Shared public typing contracts."""

from pulp import LpSolver

type SolverClass = type[LpSolver]
"""A supported PuLP solver class instantiated once per subproblem solve."""

"""Supported PuLP solver backends and instance construction."""

from collections.abc import Mapping
from typing import Any

from pulp import COIN_CMD, HiGHS, LpSolver

from descartes_mip.types import SolverClass

SUPPORTED_SOLVERS: tuple[SolverClass, ...] = (HiGHS, COIN_CMD)
"""Solver classes covered by the descartes-mip compatibility tests."""


def create_solver(
    solver: SolverClass,
    solver_options: Mapping[str, object],
    remaining_time: float | None,
) -> LpSolver:
    """Create an independent supported solver with the current time budget."""
    options: dict[str, Any] = {"msg": False, **solver_options}
    options["timeLimit"] = remaining_time
    return solver(**options)

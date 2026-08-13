"""Solver factories shipped with descartes-mip."""

from pulp import HiGHS, LpSolver


def highs_factory(remaining_time: float | None) -> LpSolver:
    """Create a quiet HiGHS solver, optionally bounded by remaining wall time."""
    if remaining_time is None:
        return HiGHS(msg=False)
    return HiGHS(msg=False, timeLimit=max(0.0, remaining_time))

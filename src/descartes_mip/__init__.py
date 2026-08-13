"""Automatic Lagrangian decomposition for PuLP models."""

from descartes_mip.exceptions import (
    DescartesMipError,
    ModelValidationError,
    SubproblemSolveError,
)
from descartes_mip.solvers import highs_factory
from descartes_mip.types import SolverFactory

__all__ = [
    "DescartesMipError",
    "ModelValidationError",
    "SolverFactory",
    "SubproblemSolveError",
    "highs_factory",
]


def main() -> None:
    """Print a short pointer to the Python API."""
    print("descartes-mip: use the Python API; see README.md")

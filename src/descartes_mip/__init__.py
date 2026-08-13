"""Automatic Lagrangian decomposition for PuLP models."""

from descartes_mip.decomposition import (
    Decomposition,
    LinearTerm,
    RelaxedConstraint,
    RelaxedTerm,
    SoftConstraint,
    Subproblem,
    decompose,
)
from descartes_mip.exceptions import (
    DescartesMipError,
    ModelValidationError,
    SubproblemSolveError,
)
from descartes_mip.result import IterationRecord, LagrangianResult, SubproblemRecord
from descartes_mip.solver import IterationCallback, solve
from descartes_mip.solvers import highs_factory
from descartes_mip.types import SolverFactory

__all__ = [
    "Decomposition",
    "DescartesMipError",
    "IterationCallback",
    "IterationRecord",
    "LagrangianResult",
    "LinearTerm",
    "ModelValidationError",
    "RelaxedConstraint",
    "RelaxedTerm",
    "SoftConstraint",
    "SolverFactory",
    "Subproblem",
    "SubproblemRecord",
    "SubproblemSolveError",
    "decompose",
    "highs_factory",
    "solve",
]


def main() -> None:
    """Print a short pointer to the Python API."""
    print("descartes-mip: use the Python API; see README.md")

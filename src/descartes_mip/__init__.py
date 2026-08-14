"""Automatic Lagrangian decomposition for PuLP models."""

from descartes_mip.backends import SUPPORTED_SOLVERS
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
from descartes_mip.types import SolverClass

__all__ = [
    "SUPPORTED_SOLVERS",
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
    "SolverClass",
    "Subproblem",
    "SubproblemRecord",
    "SubproblemSolveError",
    "decompose",
    "solve",
]


def main() -> None:
    """Print a short pointer to the Python API."""
    print("descartes-mip: use the Python API; see README.md")

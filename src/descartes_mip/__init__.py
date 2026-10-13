"""Solver-independent modeling and automatic Lagrangian decomposition."""

from descartes_mip.algorithms import (
    IterationCallback,
    IterationRecord,
    LagrangianResult,
    SubproblemRecord,
)
from descartes_mip.decomposition import (
    Decomposition,
    LinearTerm,
    RelaxedConstraint,
    RelaxedTerm,
    Subproblem,
)
from descartes_mip.exceptions import (
    DescartesMipError,
    ModelValidationError,
    SubproblemSolveError,
)
from descartes_mip.modeling import (
    Constraint,
    LinearExpr,
    Model,
    Variable,
    VariableType,
    quicksum,
)

__all__ = [
    "Constraint",
    "Decomposition",
    "DescartesMipError",
    "IterationCallback",
    "IterationRecord",
    "LagrangianResult",
    "LinearExpr",
    "LinearTerm",
    "Model",
    "ModelValidationError",
    "RelaxedConstraint",
    "RelaxedTerm",
    "Subproblem",
    "SubproblemRecord",
    "SubproblemSolveError",
    "Variable",
    "VariableType",
    "quicksum",
]


def main() -> None:
    """Print a short pointer to the Python API."""
    print("descartes-mip: use the Model API; see README.md")

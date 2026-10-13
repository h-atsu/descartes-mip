"""Public records produced by the Lagrangian algorithm."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from descartes_mip.decomposition.records import Decomposition
from descartes_mip.exceptions import ModelValidationError
from descartes_mip.modeling.expressions import Constraint, Variable

type TerminationReason = Literal["max_iterations", "time_limit", "callback"]


@dataclass(frozen=True, slots=True)
class SubproblemRecord:
    subproblem_id: int
    status: str
    objective_value: float
    elapsed_time: float


@dataclass(frozen=True, slots=True)
class IterationRecord:
    iteration: int
    primal_bound: float
    dual_bound: float
    best_primal_bound: float
    best_dual_bound: float
    absolute_gap: float
    step_size: float
    multipliers: dict[str, float]
    residuals: dict[str, float]
    violations: dict[str, float]
    variable_values: dict[str, float]
    subproblems: tuple[SubproblemRecord, ...]
    elapsed_time: float


type IterationCallback = Callable[[IterationRecord], bool]


@dataclass(frozen=True, slots=True)
class LagrangianResult:
    termination_reason: TerminationReason
    iterations: int
    best_primal_bound: float
    best_dual_bound: float
    absolute_gap: float
    relative_gap: float
    variable_values: dict[str, float]
    violations: dict[str, float]
    hard_feasible: bool
    soft_feasible: bool
    has_certified_bounds: bool
    multipliers: dict[str, float]
    history: tuple[IterationRecord, ...]
    decomposition: Decomposition
    elapsed_time: float
    timed_out_subproblem: int | None
    _model_owner: object = field(repr=False, compare=False)
    _variable_handles: tuple[Variable, ...] = field(repr=False, compare=False)
    _constraint_handles: tuple[Constraint, ...] = field(repr=False, compare=False)
    _relaxed_indices: frozenset[int] = field(repr=False, compare=False)

    def value(self, variable: Variable) -> float:
        index = self._validate_variable(variable)
        try:
            return self.variable_values[self._variable_handles[index].name]
        except KeyError as error:
            raise KeyError(
                "no variable value is available because no iteration completed"
            ) from error

    def violation(self, constraint: Constraint) -> float:
        index = self._validate_constraint(constraint)
        if index not in self._relaxed_indices:
            raise ModelValidationError("constraint was not relaxed for this result")
        try:
            return self.violations[self._constraint_handles[index].name]
        except KeyError as error:
            raise KeyError("no violation is available because no iteration completed") from error

    def multiplier(self, constraint: Constraint) -> float:
        index = self._validate_constraint(constraint)
        if index not in self._relaxed_indices:
            raise ModelValidationError("constraint was not relaxed for this result")
        return self.multipliers[self._constraint_handles[index].name]

    def _validate_variable(self, variable: Variable) -> int:
        if not isinstance(variable, Variable) or variable._owner is not self._model_owner:
            raise ModelValidationError("variable does not belong to the solved model")
        index = variable._index
        if index >= len(self._variable_handles) or self._variable_handles[index] is not variable:
            raise ModelValidationError("variable was added after this result was created")
        return index

    def _validate_constraint(self, constraint: Constraint) -> int:
        if not isinstance(constraint, Constraint) or constraint._owner is not self._model_owner:
            raise ModelValidationError("constraint does not belong to the solved model")
        index = constraint._index
        if (
            index is None
            or index >= len(self._constraint_handles)
            or self._constraint_handles[index] is not constraint
        ):
            raise ModelValidationError("constraint was added after this result was created")
        return index

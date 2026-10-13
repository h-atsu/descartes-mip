"""Public model builder."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from descartes_mip.exceptions import ModelValidationError
from descartes_mip.modeling.expressions import (
    Constraint,
    LinearExpr,
    LinearOperand,
    Variable,
    as_expression,
)
from descartes_mip.modeling.ir import (
    ConstraintIR,
    LinearTermIR,
    ModelIR,
    ObjectiveSense,
    RelaxationIR,
    VariableIR,
    VariableType,
)

if TYPE_CHECKING:
    from descartes_mip.algorithms.records import IterationCallback, LagrangianResult
    from descartes_mip.decomposition.records import Decomposition


class Model:
    """Build, inspect, and solve a linear mixed-integer model."""

    def __init__(self, name: str = "model") -> None:
        if not isinstance(name, str) or not name:
            raise ModelValidationError("model name must be a non-empty string")
        self.name = name
        self._variables: list[Variable] = []
        self._constraints: list[Constraint] = []
        self._variable_names: set[str] = set()
        self._constraint_names: set[str] = set()
        self._objective = LinearExpr()
        self._sense = ObjectiveSense.MINIMIZE
        self._relaxations: dict[int, float] = {}

    def add_var(
        self,
        name: str | None = None,
        *,
        lb: float | None = None,
        ub: float | None = None,
        vartype: VariableType = VariableType.CONTINUOUS,
    ) -> Variable:
        if not isinstance(vartype, VariableType):
            raise ModelValidationError("vartype must be a VariableType")
        resolved_name = self._resolve_name(name, "var", self._variable_names)
        low_bound = self._bound(lb, resolved_name, "lower")
        up_bound = self._bound(ub, resolved_name, "upper")
        if vartype is VariableType.BINARY:
            low_bound = 0.0 if low_bound is None else low_bound
            up_bound = 1.0 if up_bound is None else up_bound
            if low_bound < 0 or up_bound > 1:
                raise ModelValidationError("binary variable bounds must lie within [0, 1]")
        if low_bound is not None and up_bound is not None and low_bound > up_bound:
            raise ModelValidationError(f"variable {resolved_name!r} has inconsistent bounds")
        variable = Variable(
            self,
            len(self._variables),
            resolved_name,
            low_bound,
            up_bound,
            vartype,
        )
        self._variables.append(variable)
        self._variable_names.add(resolved_name)
        return variable

    def add_constraint(self, constraint: Constraint, name: str | None = None) -> Constraint:
        if not isinstance(constraint, Constraint):
            raise ModelValidationError("constraint must be created by a linear comparison")
        if constraint._index is not None:
            raise ModelValidationError("constraint has already been added to a model")
        if constraint._owner is not None and constraint._owner is not self:
            raise ModelValidationError("constraint belongs to another model")
        resolved_name = self._resolve_name(name, "constraint", self._constraint_names)
        constraint._owner = self
        constraint._expression._owner = self
        constraint._index = len(self._constraints)
        constraint._name = resolved_name
        self._constraints.append(constraint)
        self._constraint_names.add(resolved_name)
        return constraint

    def minimize(self, expression: LinearOperand) -> None:
        self._set_objective(expression, ObjectiveSense.MINIMIZE)

    def maximize(self, expression: LinearOperand) -> None:
        self._set_objective(expression, ObjectiveSense.MAXIMIZE)

    def relax(
        self,
        constraints: Constraint | Iterable[Constraint],
        *,
        penalty: float,
    ) -> None:
        if isinstance(penalty, bool) or not isinstance(penalty, (int, float)):
            raise ModelValidationError("penalty must be a positive finite number")
        resolved_penalty = float(penalty)
        if not math.isfinite(resolved_penalty) or resolved_penalty <= 0:
            raise ModelValidationError("penalty must be a positive finite number")
        selected = (constraints,) if isinstance(constraints, Constraint) else tuple(constraints)
        selected_indexes: set[int] = set()
        for constraint in selected:
            if not isinstance(constraint, Constraint):
                raise ModelValidationError("relax expects registered Constraint objects")
            if constraint._owner is not self or constraint._index is None:
                raise ModelValidationError("every relaxed constraint must belong to this model")
            if constraint._index in self._relaxations:
                raise ModelValidationError(
                    f"relaxed constraint {constraint.name!r} was specified more than once"
                )
            if constraint._index in selected_indexes:
                raise ModelValidationError(
                    f"relaxed constraint {constraint.name!r} was specified more than once"
                )
            selected_indexes.add(constraint._index)
        for constraint in selected:
            assert constraint._index is not None
            self._relaxations[constraint._index] = resolved_penalty

    def decompose(self) -> Decomposition:
        from descartes_mip.decomposition.analyzer import analyze

        model, relaxations = self._snapshot()
        return analyze(model, relaxations)

    def solve(
        self,
        *,
        highs_options: Mapping[str, bool | int | float | str] | None = None,
        max_iterations: int = 100,
        time_limit: float | None = None,
        parallel: bool = False,
        workers: int | None = None,
        polyak_factor: float = 1.0,
        initial_multipliers: Mapping[Constraint, float] | None = None,
        callback: IterationCallback | None = None,
    ) -> LagrangianResult:
        from descartes_mip.algorithms.lagrangian import solve_model

        return solve_model(
            self,
            highs_options=highs_options,
            max_iterations=max_iterations,
            time_limit=time_limit,
            parallel=parallel,
            workers=workers,
            polyak_factor=polyak_factor,
            initial_multipliers=initial_multipliers,
            callback=callback,
        )

    def _set_objective(self, expression: LinearOperand, sense: ObjectiveSense) -> None:
        resolved = as_expression(expression)
        if resolved._owner is not None and resolved._owner is not self:
            raise ModelValidationError("objective expression belongs to another model")
        resolved._owner = self
        self._objective = resolved
        self._sense = sense

    def _snapshot(self) -> tuple[ModelIR, tuple[RelaxationIR, ...]]:
        variables = tuple(
            VariableIR(
                index=index,
                name=variable.name,
                low_bound=variable.low_bound,
                up_bound=variable.up_bound,
                vartype=variable.vartype,
            )
            for index, variable in enumerate(self._variables)
        )
        constraints = tuple(
            ConstraintIR(
                index=index,
                name=constraint.name,
                sense=constraint.sense,
                constant=constraint._expression._constant,
                terms=tuple(
                    LinearTermIR(variable_index, coefficient)
                    for variable_index, coefficient in sorted(constraint._expression._terms.items())
                ),
            )
            for index, constraint in enumerate(self._constraints)
        )
        model = ModelIR(
            name=self.name,
            sense=self._sense,
            variables=variables,
            objective_constant=self._objective._constant,
            objective_terms=tuple(
                LinearTermIR(variable_index, coefficient)
                for variable_index, coefficient in sorted(self._objective._terms.items())
            ),
            constraints=constraints,
        )
        relaxations = tuple(
            RelaxationIR(index, penalty) for index, penalty in sorted(self._relaxations.items())
        )
        return model, relaxations

    @staticmethod
    def _resolve_name(name: str | None, prefix: str, used: set[str]) -> str:
        if name is not None:
            if not isinstance(name, str) or not name:
                raise ModelValidationError("names must be non-empty strings when supplied")
            if name in used:
                raise ModelValidationError(f"duplicate name {name!r}")
            return name
        index = 0
        while f"{prefix}_{index}" in used:
            index += 1
        return f"{prefix}_{index}"

    @staticmethod
    def _bound(value: float | None, variable_name: str, side: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ModelValidationError(
                f"{side} bound of variable {variable_name!r} must be finite or None"
            )
        result = float(value)
        if not math.isfinite(result):
            raise ModelValidationError(
                f"{side} bound of variable {variable_name!r} must be finite or None"
            )
        return result

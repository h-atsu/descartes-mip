"""Persistent native HiGHS subproblem implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import highspy

from descartes_mip.exceptions import ModelValidationError, SubproblemSolveError
from descartes_mip.modeling.ir import (
    ConstraintSense,
    ModelIR,
    SubproblemIR,
    VariableType,
)
from descartes_mip.subproblems.base import SubproblemSolution, SubproblemStatus

type HighsOptionValue = bool | int | float | str


def validate_highs_options(options: Mapping[str, HighsOptionValue]) -> None:
    """Validate native options even when a model has no generated subproblems."""

    highs = highspy.Highs()
    for name, value in {"output_flag": False, **options}.items():
        try:
            status = highs.setOptionValue(name, value)
        except Exception as error:
            raise ModelValidationError(f"invalid HiGHS option {name!r}: {error}") from error
        if status == highspy.HighsStatus.kError:
            raise ModelValidationError(f"invalid HiGHS option {name!r}")


class HighsSubproblemFactory:
    def compile(
        self,
        model: ModelIR,
        subproblem: SubproblemIR,
        options: Mapping[str, HighsOptionValue],
    ) -> HighsCompiledSubproblem:
        return HighsCompiledSubproblem(model, subproblem, options)


class HighsCompiledSubproblem:
    """A HiGHS model whose matrix and domains persist across iterations."""

    def __init__(
        self,
        model: ModelIR,
        subproblem: SubproblemIR,
        options: Mapping[str, HighsOptionValue],
    ) -> None:
        self.id = subproblem.id
        self.variable_indices = subproblem.variable_indices
        self._highs = highspy.Highs()
        self._set_option("output_flag", False)
        for name, value in options.items():
            self._set_option(name, value)

        variables = [model.variables[index] for index in self.variable_indices]
        self._check(
            self._highs.addVars(
                len(variables),
                [
                    -highspy.kHighsInf if variable.low_bound is None else variable.low_bound
                    for variable in variables
                ],
                [
                    highspy.kHighsInf if variable.up_bound is None else variable.up_bound
                    for variable in variables
                ],
            ),
            "add variables",
        )
        for local_index, variable in enumerate(variables):
            if variable.vartype is not VariableType.CONTINUOUS:
                self._check(
                    self._highs.changeColIntegrality(local_index, highspy.HighsVarType.kInteger),
                    f"set integrality for {variable.name!r}",
                )

        local_index = {
            global_index: position for position, global_index in enumerate(self.variable_indices)
        }
        row_lower: list[float] = []
        row_upper: list[float] = []
        starts = [0]
        indices: list[int] = []
        values: list[float] = []
        for constraint_index in subproblem.hard_constraint_indices:
            constraint = model.constraints[constraint_index]
            rhs = -constraint.constant
            if constraint.sense is ConstraintSense.LE:
                row_lower.append(-highspy.kHighsInf)
                row_upper.append(rhs)
            elif constraint.sense is ConstraintSense.GE:
                row_lower.append(rhs)
                row_upper.append(highspy.kHighsInf)
            else:
                row_lower.append(rhs)
                row_upper.append(rhs)
            for term in constraint.terms:
                indices.append(local_index[term.variable_index])
                values.append(term.coefficient)
            starts.append(len(indices))
        self._check(
            self._highs.addRows(
                len(row_lower),
                row_lower,
                row_upper,
                len(indices),
                starts,
                indices,
                values,
            ),
            "add constraints",
        )

    def solve(
        self,
        objective_coefficients: Sequence[float],
        remaining_time: float | None,
    ) -> SubproblemSolution:
        if len(objective_coefficients) != len(self.variable_indices):
            raise SubproblemSolveError(f"subproblem {self.id} received an invalid objective vector")
        self._check(
            self._highs.changeColsCost(
                len(self.variable_indices),
                list(range(len(self.variable_indices))),
                list(objective_coefficients),
            ),
            "update objective",
        )
        self._set_option(
            "time_limit", highspy.kHighsInf if remaining_time is None else remaining_time
        )
        run_status = self._highs.run()
        model_status = self._highs.getModelStatus()
        detail = self._highs.modelStatusToString(model_status)
        if run_status == highspy.HighsStatus.kError:
            return SubproblemSolution(SubproblemStatus.ERROR, None, (), detail)
        if model_status == highspy.HighsModelStatus.kOptimal:
            solution = self._highs.getSolution()
            return SubproblemSolution(
                SubproblemStatus.OPTIMAL,
                float(self._highs.getObjectiveValue()),
                tuple(float(value) for value in solution.col_value),
                detail,
            )
        if model_status == highspy.HighsModelStatus.kTimeLimit:
            status = SubproblemStatus.TIME_LIMIT
        elif model_status == highspy.HighsModelStatus.kInfeasible:
            status = SubproblemStatus.INFEASIBLE
        elif model_status in (
            highspy.HighsModelStatus.kUnbounded,
            highspy.HighsModelStatus.kUnboundedOrInfeasible,
        ):
            status = SubproblemStatus.UNBOUNDED
        else:
            status = SubproblemStatus.ERROR
        return SubproblemSolution(status, None, (), detail)

    def _set_option(self, name: str, value: HighsOptionValue) -> None:
        try:
            status = self._highs.setOptionValue(name, value)
        except Exception as error:
            raise ModelValidationError(f"invalid HiGHS option {name!r}: {error}") from error
        if status == highspy.HighsStatus.kError:
            raise ModelValidationError(f"invalid HiGHS option {name!r}")

    def _check(self, status: highspy.HighsStatus, action: str) -> None:
        if status == highspy.HighsStatus.kError:
            raise SubproblemSolveError(f"subproblem {self.id} failed to {action}")

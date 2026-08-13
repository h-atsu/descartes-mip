"""Sequential projected-subgradient solver for the Lagrangian dual."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import pulp

from descartes_mip._model import ConstraintIR, ModelIR
from descartes_mip.decomposition import Decomposition, SoftConstraint, decompose
from descartes_mip.exceptions import ModelValidationError, SubproblemSolveError
from descartes_mip.result import (
    IterationRecord,
    LagrangianResult,
    SubproblemRecord,
)
from descartes_mip.solvers import highs_factory
from descartes_mip.types import SolverFactory

type IterationCallback = Callable[[IterationRecord], bool]


@dataclass(slots=True)
class _BuiltSubproblem:
    id: int
    problem: pulp.LpProblem
    variables: dict[int, pulp.LpVariable]
    objective_coefficients: dict[int, float]


def solve(
    problem: pulp.LpProblem,
    *,
    relax: Sequence[SoftConstraint],
    solver_factory: SolverFactory = highs_factory,
    max_iterations: int = 100,
    polyak_factor: float = 1.0,
    initial_multipliers: Mapping[str, float] | None = None,
    callback: IterationCallback | None = None,
) -> LagrangianResult:
    """Solve the Lagrangian dual using projected Polyak subgradient steps.

    Every generated MILP subproblem must be solved to proven optimality. Bounds
    are reported in the original objective sense: a primal upper and dual lower
    bound for minimization, and a primal lower and dual upper bound for maximization.
    """
    _validate_options(max_iterations, polyak_factor, solver_factory)
    soft_constraints = tuple(relax)
    decomposition = decompose(problem, relax=soft_constraints)
    model = decomposition._model
    relaxed = _relaxed_model_constraints(model, decomposition)
    multipliers = _initial_multipliers(decomposition, initial_multipliers)
    built_subproblems = _build_subproblems(decomposition, relaxed)
    _validate_subproblems_bounded(
        decomposition,
        built_subproblems,
        solver_factory,
    )

    best_primal_normalized = math.inf
    best_dual_normalized = -math.inf
    best_values: dict[str, float] = {}
    best_slacks: dict[str, float] = {}
    history: list[IterationRecord] = []
    termination_reason = "max_iterations"

    for iteration in range(max_iterations):
        values, subproblem_records, variable_objective = _solve_subproblems(
            built_subproblems,
            model,
            relaxed,
            multipliers,
            solver_factory,
        )
        residuals = {
            constraint.name: _normalized_constraint_value(model, constraint, values)
            for constraint in relaxed
        }
        violations = {
            constraint.name: _violation(constraint, residuals[constraint.name])
            for constraint in relaxed
        }
        dual_normalized = (
            variable_objective
            + _normalized_objective_constant(model)
            + sum(
                multipliers[constraint.name] * _normalized_constraint_constant(constraint)
                for constraint in relaxed
            )
        )
        primal_normalized = _normalized_objective_value(model, values) + sum(
            constraint.penalty * violations[constraint.name]
            for constraint in decomposition.relaxed_constraints
        )

        best_dual_normalized = max(best_dual_normalized, dual_normalized)
        if primal_normalized < best_primal_normalized:
            best_primal_normalized = primal_normalized
            best_values = dict(values)
            best_slacks = dict(violations)

        gap = max(0.0, best_primal_normalized - best_dual_normalized)
        norm_squared = sum(value * value for value in residuals.values())
        step_size = 0.0 if norm_squared == 0 else polyak_factor * gap / norm_squared
        next_multipliers = _update_multipliers(
            decomposition,
            multipliers,
            residuals,
            step_size,
        )
        record = IterationRecord(
            iteration=iteration,
            primal_bound=_from_normalized(model, primal_normalized),
            dual_bound=_from_normalized(model, dual_normalized),
            best_primal_bound=_from_normalized(model, best_primal_normalized),
            best_dual_bound=_from_normalized(model, best_dual_normalized),
            absolute_gap=gap,
            step_size=step_size,
            multipliers=dict(multipliers),
            residuals=residuals,
            violations=violations,
            variable_values=values,
            subproblems=subproblem_records,
        )
        history.append(record)
        multipliers = next_multipliers
        if callback is not None and callback(record):
            termination_reason = "callback"
            break

    relative_gap = _relative_gap(best_primal_normalized, best_dual_normalized)
    return LagrangianResult(
        termination_reason=termination_reason,
        iterations=len(history),
        best_primal_bound=_from_normalized(model, best_primal_normalized),
        best_dual_bound=_from_normalized(model, best_dual_normalized),
        absolute_gap=max(0.0, best_primal_normalized - best_dual_normalized),
        relative_gap=relative_gap,
        variable_values=best_values,
        slacks=best_slacks,
        hard_feasible=all(value <= 1e-7 for value in best_slacks.values()),
        soft_feasible=True,
        multipliers=dict(multipliers),
        history=tuple(history),
        decomposition=decomposition,
    )


def _validate_options(
    max_iterations: int,
    polyak_factor: float,
    solver_factory: SolverFactory,
) -> None:
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise ModelValidationError("max_iterations must be a positive integer")
    if max_iterations <= 0:
        raise ModelValidationError("max_iterations must be a positive integer")
    if isinstance(polyak_factor, bool) or not isinstance(polyak_factor, (int, float)):
        raise ModelValidationError("polyak_factor must be finite and in (0, 2]")
    if not math.isfinite(polyak_factor) or not 0 < polyak_factor <= 2:
        raise ModelValidationError("polyak_factor must be finite and in (0, 2]")
    if not callable(solver_factory):
        raise ModelValidationError("solver_factory must be callable")


def _relaxed_model_constraints(
    model: ModelIR, decomposition: Decomposition
) -> tuple[ConstraintIR, ...]:
    names = {constraint.name for constraint in decomposition.relaxed_constraints}
    return tuple(constraint for constraint in model.constraints if constraint.name in names)


def _initial_multipliers(
    decomposition: Decomposition,
    supplied: Mapping[str, float] | None,
) -> dict[str, float]:
    relaxed_by_name = {
        constraint.name: constraint for constraint in decomposition.relaxed_constraints
    }
    supplied_values = {} if supplied is None else dict(supplied)
    unknown = supplied_values.keys() - relaxed_by_name.keys()
    if unknown:
        raise ModelValidationError(f"unknown initial multiplier names: {sorted(unknown)!r}")

    result: dict[str, float] = {}
    for name, constraint in relaxed_by_name.items():
        value = supplied_values.get(name, 0.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ModelValidationError(f"initial multiplier for {name!r} must be finite")
        lower = -constraint.penalty if constraint.sense == "==" else 0.0
        upper = constraint.penalty
        if not lower <= value <= upper:
            raise ModelValidationError(
                f"initial multiplier for {name!r} must be in [{lower}, {upper}]"
            )
        result[name] = float(value)
    return result


def _build_subproblems(
    decomposition: Decomposition,
    relaxed: tuple[ConstraintIR, ...],
) -> tuple[_BuiltSubproblem, ...]:
    model = decomposition._model
    variable_index_by_name = {
        variable.name: index for index, variable in enumerate(model.variables)
    }
    constraint_by_name = {constraint.name: constraint for constraint in model.constraints}
    normalized_objective_coefficients = {
        term.variable_index: _objective_sign(model) * term.coefficient
        for term in model.objective_terms
    }
    built: list[_BuiltSubproblem] = []
    for subproblem in decomposition.subproblems:
        generated = pulp.LpProblem(
            f"{model.name}_subproblem_{subproblem.id}",
            pulp.LpMinimize,
        )
        variable_indexes = tuple(variable_index_by_name[name] for name in subproblem.variables)
        variables = {
            index: generated.add_variable(
                model.variables[index].name,
                lowBound=model.variables[index].low_bound,
                upBound=model.variables[index].up_bound,
                cat=model.variables[index].category,
            )
            for index in variable_indexes
        }
        for name in subproblem.hard_constraints:
            constraint = constraint_by_name[name]
            expression = pulp.lpSum(
                term.coefficient * variables[term.variable_index] for term in constraint.terms
            )
            generated += (
                pulp.LpConstraint(
                    e=expression,
                    sense=constraint.sense,
                    rhs=-constraint.constant,
                ),
                constraint.name,
            )
        built.append(
            _BuiltSubproblem(
                id=subproblem.id,
                problem=generated,
                variables=variables,
                objective_coefficients={
                    index: normalized_objective_coefficients.get(index, 0.0)
                    for index in variable_indexes
                },
            )
        )
    return tuple(built)


def _validate_subproblems_bounded(
    decomposition: Decomposition,
    subproblems: tuple[_BuiltSubproblem, ...],
    solver_factory: SolverFactory,
) -> None:
    """Require every hard feasible region to have a trivial recession cone."""
    model = decomposition._model
    diagnostic_by_id = {subproblem.id: subproblem for subproblem in decomposition.subproblems}
    constraint_by_name = {constraint.name: constraint for constraint in model.constraints}

    for subproblem in subproblems:
        variable_indexes = tuple(subproblem.variables)
        if all(
            model.variables[index].low_bound is not None
            and model.variables[index].up_bound is not None
            for index in variable_indexes
        ):
            continue
        diagnostic = diagnostic_by_id[subproblem.id]
        hard_constraints = tuple(constraint_by_name[name] for name in diagnostic.hard_constraints)
        ray = _find_recession_ray(
            subproblem.id,
            variable_indexes,
            hard_constraints,
            model,
            solver_factory,
        )
        if ray is not None:
            formatted_ray = ", ".join(f"{name}={value:.6g}" for name, value in ray.items())
            raise ModelValidationError(
                f"subproblem {subproblem.id} is not bounded after relaxation; "
                f"recession direction: {formatted_ray}. Add valid variable bounds "
                "or relax a different set of constraints"
            )


def _find_recession_ray(
    subproblem_id: int,
    variable_indexes: tuple[int, ...],
    hard_constraints: tuple[ConstraintIR, ...],
    model: ModelIR,
    solver_factory: SolverFactory,
) -> dict[str, float] | None:
    """Return a normalized nonzero recession direction, if one exists."""
    ray_problem = pulp.LpProblem(
        f"{model.name}_subproblem_{subproblem_id}_recession",
        pulp.LpMinimize,
    )
    directions = {
        index: ray_problem.add_variable(
            f"direction_{model.variables[index].name}",
            lowBound=-1,
            upBound=1,
        )
        for index in variable_indexes
    }
    positive_selectors = {
        index: ray_problem.add_variable(
            f"positive_{model.variables[index].name}",
            lowBound=0,
            upBound=1,
            cat=pulp.LpBinary,
        )
        for index in variable_indexes
    }
    negative_selectors = {
        index: ray_problem.add_variable(
            f"negative_{model.variables[index].name}",
            lowBound=0,
            upBound=1,
            cat=pulp.LpBinary,
        )
        for index in variable_indexes
    }

    ray_problem += 0
    for index, direction in directions.items():
        variable = model.variables[index]
        if variable.low_bound is not None:
            ray_problem += direction >= 0, f"lower_bound_direction_{index}"
        if variable.up_bound is not None:
            ray_problem += direction <= 0, f"upper_bound_direction_{index}"

        # With -1 <= direction <= 1, these implications force a selected
        # coordinate to +1 or -1 while leaving unselected coordinates free.
        ray_problem += (
            direction >= 2 * positive_selectors[index] - 1,
            f"select_positive_direction_{index}",
        )
        ray_problem += (
            direction <= 1 - 2 * negative_selectors[index],
            f"select_negative_direction_{index}",
        )

    ray_problem += (
        pulp.lpSum(positive_selectors.values()) + pulp.lpSum(negative_selectors.values()) == 1,
        "select_nonzero_direction",
    )
    for constraint_index, constraint in enumerate(hard_constraints):
        expression = pulp.lpSum(
            term.coefficient * directions[term.variable_index] for term in constraint.terms
        )
        ray_problem += (
            pulp.LpConstraint(
                e=expression,
                sense=constraint.sense,
                rhs=0,
            ),
            f"hard_direction_{constraint_index}_{constraint.name}",
        )

    try:
        status = ray_problem.solve(solver_factory(None))
    except Exception as error:
        raise SubproblemSolveError(
            f"recession check for subproblem {subproblem_id} raised {type(error).__name__}: {error}"
        ) from error
    if status == pulp.LpStatusInfeasible:
        return None
    status_name = pulp.LpStatus.get(status, f"Unknown({status})")
    if status != pulp.LpStatusOptimal:
        raise SubproblemSolveError(
            f"recession check for subproblem {subproblem_id} did not finish: {status_name}"
        )

    result: dict[str, float] = {}
    for index, direction in directions.items():
        value = direction.value()
        if value is None or not math.isfinite(value):
            raise SubproblemSolveError(
                f"recession check for subproblem {subproblem_id} returned no finite direction"
            )
        if abs(value) > 1e-7:
            result[model.variables[index].name] = float(value)
    if not result:
        raise SubproblemSolveError(
            f"recession check for subproblem {subproblem_id} returned a zero direction"
        )
    return result


def _solve_subproblems(
    subproblems: tuple[_BuiltSubproblem, ...],
    model: ModelIR,
    relaxed: tuple[ConstraintIR, ...],
    multipliers: Mapping[str, float],
    solver_factory: SolverFactory,
) -> tuple[dict[str, float], tuple[SubproblemRecord, ...], float]:
    values: dict[str, float] = {}
    records: list[SubproblemRecord] = []
    total_objective = 0.0
    for subproblem in subproblems:
        coefficients = dict(subproblem.objective_coefficients)
        for constraint in relaxed:
            multiplier = multipliers[constraint.name]
            term_sign = _constraint_sign(constraint)
            for term in constraint.terms:
                if term.variable_index in subproblem.variables:
                    coefficients[term.variable_index] += multiplier * term_sign * term.coefficient
        objective = pulp.lpSum(
            coefficients[index] * variable for index, variable in subproblem.variables.items()
        )
        subproblem.problem.setObjective(objective)
        try:
            solver = solver_factory(None)
            status = subproblem.problem.solve(solver)
        except Exception as error:
            raise SubproblemSolveError(
                f"subproblem {subproblem.id} solver raised {type(error).__name__}: {error}"
            ) from error
        status_name = pulp.LpStatus.get(status, f"Unknown({status})")
        if status != pulp.LpStatusOptimal:
            raise SubproblemSolveError(
                f"subproblem {subproblem.id} did not reach optimality: {status_name}"
            )
        objective_value = float(pulp.value(subproblem.problem.objective) or 0.0)
        total_objective += objective_value
        records.append(SubproblemRecord(subproblem.id, status_name, objective_value))
        for index, variable in subproblem.variables.items():
            value = variable.value()
            if value is None:
                value = _canonical_free_value(model, index)
            if not math.isfinite(value):
                raise SubproblemSolveError(
                    f"subproblem {subproblem.id} returned no finite value for {variable.name!r}"
                )
            values[model.variables[index].name] = float(value)
    return values, tuple(records), total_objective


def _canonical_free_value(model: ModelIR, variable_index: int) -> float:
    variable = model.variables[variable_index]
    lower = variable.low_bound
    upper = variable.up_bound
    candidate = 0.0
    if lower is not None:
        candidate = max(candidate, lower)
    if upper is not None:
        candidate = min(candidate, upper)
    if variable.category in (pulp.LpInteger, pulp.LpBinary):
        candidate = math.ceil(candidate)
        if upper is not None and candidate > upper:
            candidate = math.floor(upper)
    return float(candidate)


def _normalized_objective_value(model: ModelIR, values: Mapping[str, float]) -> float:
    raw = model.objective_constant + sum(
        term.coefficient * values[model.variables[term.variable_index].name]
        for term in model.objective_terms
    )
    return _objective_sign(model) * raw


def _normalized_objective_constant(model: ModelIR) -> float:
    return _objective_sign(model) * model.objective_constant


def _normalized_constraint_value(
    model: ModelIR,
    constraint: ConstraintIR,
    values: Mapping[str, float],
) -> float:
    raw = constraint.constant + sum(
        term.coefficient * values[model.variables[term.variable_index].name]
        for term in constraint.terms
    )
    return _constraint_sign(constraint) * raw


def _normalized_constraint_constant(constraint: ConstraintIR) -> float:
    return _constraint_sign(constraint) * constraint.constant


def _constraint_sign(constraint: ConstraintIR) -> float:
    return -1.0 if constraint.sense == pulp.LpConstraintGE else 1.0


def _violation(constraint: ConstraintIR, normalized_value: float) -> float:
    if constraint.sense == pulp.LpConstraintEQ:
        return abs(normalized_value)
    return max(0.0, normalized_value)


def _update_multipliers(
    decomposition: Decomposition,
    current: Mapping[str, float],
    residuals: Mapping[str, float],
    step_size: float,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for constraint in decomposition.relaxed_constraints:
        lower = -constraint.penalty if constraint.sense == "==" else 0.0
        candidate = current[constraint.name] + step_size * residuals[constraint.name]
        result[constraint.name] = min(constraint.penalty, max(lower, candidate))
    return result


def _objective_sign(model: ModelIR) -> float:
    return 1.0 if model.sense == pulp.LpMinimize else -1.0


def _from_normalized(model: ModelIR, value: float) -> float:
    return _objective_sign(model) * value


def _relative_gap(primal: float, dual: float) -> float:
    gap = max(0.0, primal - dual)
    return gap / max(1.0, abs(primal), abs(dual))

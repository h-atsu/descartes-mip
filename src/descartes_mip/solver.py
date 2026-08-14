"""Projected-subgradient solver for the Lagrangian dual."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

import pulp

from descartes_mip._model import ConstraintIR, ModelIR
from descartes_mip.backends import SUPPORTED_SOLVERS, create_solver
from descartes_mip.decomposition import Decomposition, SoftConstraint, decompose
from descartes_mip.exceptions import ModelValidationError, SubproblemSolveError
from descartes_mip.result import (
    IterationRecord,
    LagrangianResult,
    SubproblemRecord,
)
from descartes_mip.types import SolverClass

type IterationCallback = Callable[[IterationRecord], bool]


@dataclass(slots=True)
class _BuiltSubproblem:
    id: int
    problem: pulp.LpProblem
    variables: dict[int, pulp.LpVariable]
    objective_coefficients: dict[int, float]


@dataclass(slots=True)
class _SolvedSubproblem:
    values: dict[str, float]
    record: SubproblemRecord
    objective_value: float


class _TimeLimitReached(Exception):
    """Signal that an incomplete iteration must be discarded."""

    def __init__(self, subproblem_id: int | None) -> None:
        self.subproblem_id = subproblem_id


def solve(
    problem: pulp.LpProblem,
    *,
    relax: Sequence[SoftConstraint],
    solver: SolverClass = pulp.HiGHS,
    solver_options: Mapping[str, object] | None = None,
    max_iterations: int = 100,
    time_limit: float | None = None,
    parallel: bool = False,
    workers: int | None = None,
    polyak_factor: float = 1.0,
    initial_multipliers: Mapping[str, float] | None = None,
    callback: IterationCallback | None = None,
) -> LagrangianResult:
    """Solve the Lagrangian dual using projected Polyak subgradient steps.

    Every generated MILP subproblem must be solved to proven optimality. Bounds
    are reported in the original objective sense: a primal upper and dual lower
    bound for minimization, and a primal lower and dual upper bound for maximization.
    """
    _validate_options(
        max_iterations,
        time_limit,
        parallel,
        workers,
        polyak_factor,
        solver,
        solver_options,
    )
    started_at = time.monotonic()
    deadline = None if time_limit is None else started_at + time_limit
    soft_constraints = tuple(relax)
    resolved_solver_options = {} if solver_options is None else dict(solver_options)
    decomposition = decompose(problem, relax=soft_constraints)
    _validate_explicit_variable_bounds(decomposition)
    model = decomposition._model
    relaxed = _relaxed_model_constraints(model, decomposition)
    multipliers = _initial_multipliers(decomposition, initial_multipliers)
    built_subproblems = _build_subproblems(decomposition, relaxed)

    best_primal_normalized = math.inf
    best_dual_normalized = -math.inf
    best_values: dict[str, float] = {}
    best_slacks: dict[str, float] = {}
    history: list[IterationRecord] = []
    termination_reason = "max_iterations"
    timed_out_subproblem: int | None = None

    for iteration in range(max_iterations):
        if _deadline_reached(deadline):
            termination_reason = "time_limit"
            break
        try:
            values, subproblem_records, variable_objective = _solve_subproblems(
                built_subproblems,
                model,
                relaxed,
                multipliers,
                solver,
                resolved_solver_options,
                deadline,
                parallel,
                workers,
            )
        except _TimeLimitReached as timeout:
            termination_reason = "time_limit"
            timed_out_subproblem = timeout.subproblem_id
            break
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
            elapsed_time=max(0.0, time.monotonic() - started_at),
        )
        history.append(record)
        multipliers = next_multipliers
        if callback is not None and callback(record):
            termination_reason = "callback"
            break
        if _deadline_reached(deadline):
            termination_reason = "time_limit"
            break

    relative_gap = _relative_gap(best_primal_normalized, best_dual_normalized)
    has_completed_iteration = bool(history)
    return LagrangianResult(
        termination_reason=termination_reason,
        iterations=len(history),
        best_primal_bound=_from_normalized(model, best_primal_normalized),
        best_dual_bound=_from_normalized(model, best_dual_normalized),
        absolute_gap=max(0.0, best_primal_normalized - best_dual_normalized),
        relative_gap=relative_gap,
        variable_values=best_values,
        slacks=best_slacks,
        hard_feasible=has_completed_iteration
        and all(value <= 1e-7 for value in best_slacks.values()),
        soft_feasible=has_completed_iteration,
        has_certified_bounds=has_completed_iteration,
        multipliers=dict(multipliers),
        history=tuple(history),
        decomposition=decomposition,
        elapsed_time=max(0.0, time.monotonic() - started_at),
        timed_out_subproblem=timed_out_subproblem,
    )


def _validate_options(
    max_iterations: int,
    time_limit: float | None,
    parallel: bool,
    workers: int | None,
    polyak_factor: float,
    solver: SolverClass,
    solver_options: Mapping[str, object] | None,
) -> None:
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise ModelValidationError("max_iterations must be a positive integer")
    if max_iterations <= 0:
        raise ModelValidationError("max_iterations must be a positive integer")
    if time_limit is not None and (
        isinstance(time_limit, bool)
        or not isinstance(time_limit, (int, float))
        or not math.isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ModelValidationError("time_limit must be a positive finite number or None")
    if not isinstance(parallel, bool):
        raise ModelValidationError("parallel must be a boolean")
    if workers is not None and (
        isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0
    ):
        raise ModelValidationError("workers must be a positive integer or None")
    if workers is not None and not parallel:
        raise ModelValidationError("workers requires parallel=True")
    if isinstance(polyak_factor, bool) or not isinstance(polyak_factor, (int, float)):
        raise ModelValidationError("polyak_factor must be finite and in (0, 2]")
    if not math.isfinite(polyak_factor) or not 0 < polyak_factor <= 2:
        raise ModelValidationError("polyak_factor must be finite and in (0, 2]")
    if solver not in SUPPORTED_SOLVERS:
        supported = ", ".join(solver_class.__name__ for solver_class in SUPPORTED_SOLVERS)
        raise ModelValidationError(
            f"unsupported solver; expected one of these PuLP solver classes: {supported}"
        )
    if solver_options is not None and not isinstance(solver_options, Mapping):
        raise ModelValidationError("solver_options must be a mapping or None")
    if solver_options is not None:
        if not all(isinstance(name, str) for name in solver_options):
            raise ModelValidationError("solver_options keys must be strings")
        if "timeLimit" in solver_options:
            raise ModelValidationError(
                "solver_options must not contain reserved option 'timeLimit'; use time_limit"
            )


def _validate_explicit_variable_bounds(decomposition: Decomposition) -> None:
    """Require finite lower and upper bounds on every decision variable."""
    missing_bounds: list[str] = []
    for variable in decomposition._model.variables:
        missing: list[str] = []
        if variable.low_bound is None:
            missing.append("lower")
        if variable.up_bound is None:
            missing.append("upper")
        if missing:
            missing_bounds.append(f"{variable.name} ({', '.join(missing)})")
    if missing_bounds:
        raise ModelValidationError(
            "all decision variables must have finite lower and upper bounds; "
            f"missing bounds: {', '.join(missing_bounds)}"
        )


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


def _solve_subproblems(
    subproblems: tuple[_BuiltSubproblem, ...],
    model: ModelIR,
    relaxed: tuple[ConstraintIR, ...],
    multipliers: Mapping[str, float],
    solver: SolverClass,
    solver_options: Mapping[str, object],
    deadline: float | None,
    parallel: bool,
    workers: int | None,
) -> tuple[dict[str, float], tuple[SubproblemRecord, ...], float]:
    if parallel and len(subproblems) > 1:
        solver_creation_lock = Lock()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = tuple(
                executor.submit(
                    _solve_one_subproblem,
                    subproblem,
                    model,
                    relaxed,
                    multipliers,
                    solver,
                    solver_options,
                    deadline,
                    solver_creation_lock,
                )
                for subproblem in subproblems
            )
            solved = tuple(future.result() for future in futures)
    else:
        solved = tuple(
            _solve_one_subproblem(
                subproblem,
                model,
                relaxed,
                multipliers,
                solver,
                solver_options,
                deadline,
                None,
            )
            for subproblem in subproblems
        )

    values: dict[str, float] = {}
    records: list[SubproblemRecord] = []
    total_objective = 0.0
    for outcome in solved:
        values.update(outcome.values)
        records.append(outcome.record)
        total_objective += outcome.objective_value
    return values, tuple(records), total_objective


def _solve_one_subproblem(
    subproblem: _BuiltSubproblem,
    model: ModelIR,
    relaxed: tuple[ConstraintIR, ...],
    multipliers: Mapping[str, float],
    solver: SolverClass,
    solver_options: Mapping[str, object],
    deadline: float | None,
    solver_creation_lock: Lock | None,
) -> _SolvedSubproblem:
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
    subproblem_started_at = time.monotonic()
    try:
        if solver_creation_lock is None:
            remaining_time = _remaining_time(deadline)
            if remaining_time is not None and remaining_time <= 0:
                raise _TimeLimitReached(subproblem.id)
            solver_instance = create_solver(solver, solver_options, remaining_time)
        else:
            with solver_creation_lock:
                remaining_time = _remaining_time(deadline)
                if remaining_time is not None and remaining_time <= 0:
                    raise _TimeLimitReached(subproblem.id)
                solver_instance = create_solver(solver, solver_options, remaining_time)
        status = subproblem.problem.solve(solver_instance)
    except _TimeLimitReached:
        raise
    except TimeoutError as error:
        if deadline is not None:
            raise _TimeLimitReached(subproblem.id) from error
        raise SubproblemSolveError(
            f"subproblem {subproblem.id} solver raised TimeoutError: {error}"
        ) from error
    except Exception as error:
        raise SubproblemSolveError(
            f"subproblem {subproblem.id} solver raised {type(error).__name__}: {error}"
        ) from error
    status_name = pulp.LpStatus.get(status, f"Unknown({status})")
    if status != pulp.LpStatusOptimal:
        if status in (pulp.LpStatusInfeasible, pulp.LpStatusUnbounded):
            raise SubproblemSolveError(
                f"subproblem {subproblem.id} did not reach optimality: {status_name}"
            )
        if deadline is not None and (
            status == pulp.LpStatusNotSolved or _deadline_reached(deadline)
        ):
            raise _TimeLimitReached(subproblem.id)
        raise SubproblemSolveError(
            f"subproblem {subproblem.id} did not reach optimality: {status_name}"
        )
    objective_value = float(pulp.value(subproblem.problem.objective) or 0.0)
    values: dict[str, float] = {}
    for index, variable in subproblem.variables.items():
        value = variable.value()
        if value is None:
            value = _canonical_free_value(model, index)
        if not math.isfinite(value):
            raise SubproblemSolveError(
                f"subproblem {subproblem.id} returned no finite value for {variable.name!r}"
            )
        values[model.variables[index].name] = float(value)
    return _SolvedSubproblem(
        values=values,
        record=SubproblemRecord(
            subproblem.id,
            status_name,
            objective_value,
            max(0.0, time.monotonic() - subproblem_started_at),
        ),
        objective_value=objective_value,
    )


def _remaining_time(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def _deadline_reached(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


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
    if not math.isfinite(primal) or not math.isfinite(dual):
        return math.inf
    gap = max(0.0, primal - dual)
    return gap / max(1.0, abs(primal), abs(dual))

"""Projected-subgradient solver for the boxed Lagrangian dual."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from descartes_mip.algorithms.records import (
    IterationCallback,
    IterationRecord,
    LagrangianResult,
    SubproblemRecord,
)
from descartes_mip.decomposition.analyzer import analyze
from descartes_mip.decomposition.records import Decomposition
from descartes_mip.exceptions import ModelValidationError, SubproblemSolveError
from descartes_mip.modeling.expressions import Constraint
from descartes_mip.modeling.ir import (
    ConstraintIR,
    ConstraintSense,
    ModelIR,
    ObjectiveSense,
)
from descartes_mip.modeling.model import Model
from descartes_mip.subproblems.base import CompiledSubproblem, SubproblemStatus
from descartes_mip.subproblems.highs import (
    HighsOptionValue,
    HighsSubproblemFactory,
    validate_highs_options,
)


@dataclass(slots=True)
class _SolvedSubproblem:
    values: dict[str, float]
    record: SubproblemRecord
    objective_value: float


@dataclass(frozen=True, slots=True)
class _PreparedSubproblem:
    solver: CompiledSubproblem
    base_objective: tuple[float, ...]
    lagrangian_terms: tuple[tuple[int, str, float], ...]


class _TimeLimitReached(Exception):
    def __init__(self, subproblem_id: int | None) -> None:
        self.subproblem_id = subproblem_id


def solve_model(
    owner: Model,
    *,
    highs_options: Mapping[str, HighsOptionValue] | None,
    max_iterations: int,
    time_limit: float | None,
    parallel: bool,
    workers: int | None,
    polyak_factor: float,
    initial_multipliers: Mapping[Constraint, float] | None,
    callback: IterationCallback | None,
) -> LagrangianResult:
    options = _validate_options(
        max_iterations, time_limit, parallel, workers, polyak_factor, highs_options
    )
    validate_highs_options(options)
    started_at = time.monotonic()
    deadline = None if time_limit is None else started_at + time_limit
    model, relaxations = owner._snapshot()
    decomposition = analyze(model, relaxations)
    _validate_explicit_variable_bounds(model)
    relaxed = tuple(model.constraints[item.constraint_index] for item in relaxations)
    multipliers = _initial_multipliers(owner, decomposition, initial_multipliers)
    factory = HighsSubproblemFactory()
    compiled = tuple(
        _prepare_subproblem(
            factory.compile(model, subproblem, options),
            model,
            relaxed,
        )
        for subproblem in decomposition._subproblem_irs
    )

    best_primal_normalized = math.inf
    best_dual_normalized = -math.inf
    best_values: dict[str, float] = {}
    best_violations: dict[str, float] = {}
    history: list[IterationRecord] = []
    termination_reason = "max_iterations"
    timed_out_subproblem: int | None = None

    for iteration in range(max_iterations):
        if _deadline_reached(deadline):
            termination_reason = "time_limit"
            break
        try:
            values, subproblem_records, variable_objective = _solve_subproblems(
                compiled,
                model,
                multipliers,
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
            decomposition._relaxation_penalties[constraint.index] * violations[constraint.name]
            for constraint in relaxed
        )
        best_dual_normalized = max(best_dual_normalized, dual_normalized)
        if primal_normalized < best_primal_normalized:
            best_primal_normalized = primal_normalized
            best_values = dict(values)
            best_violations = dict(violations)

        gap = max(0.0, best_primal_normalized - best_dual_normalized)
        norm_squared = sum(value * value for value in residuals.values())
        step_size = 0.0 if norm_squared == 0 else polyak_factor * gap / norm_squared
        next_multipliers = _update_multipliers(decomposition, multipliers, residuals, step_size)
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

    has_completed_iteration = bool(history)
    variable_handles = tuple(owner._variables[: len(model.variables)])
    constraint_handles = tuple(owner._constraints[: len(model.constraints)])
    return LagrangianResult(
        termination_reason=termination_reason,
        iterations=len(history),
        best_primal_bound=_from_normalized(model, best_primal_normalized),
        best_dual_bound=_from_normalized(model, best_dual_normalized),
        absolute_gap=max(0.0, best_primal_normalized - best_dual_normalized),
        relative_gap=_relative_gap(best_primal_normalized, best_dual_normalized),
        variable_values=best_values,
        violations=best_violations,
        hard_feasible=has_completed_iteration
        and all(value <= 1e-7 for value in best_violations.values()),
        soft_feasible=has_completed_iteration,
        has_certified_bounds=has_completed_iteration,
        multipliers=dict(multipliers),
        history=tuple(history),
        decomposition=decomposition,
        elapsed_time=max(0.0, time.monotonic() - started_at),
        timed_out_subproblem=timed_out_subproblem,
        _model_owner=owner,
        _variable_handles=variable_handles,
        _constraint_handles=constraint_handles,
        _relaxed_indices=frozenset(item.constraint_index for item in relaxations),
    )


def _validate_options(
    max_iterations: int,
    time_limit: float | None,
    parallel: bool,
    workers: int | None,
    polyak_factor: float,
    highs_options: Mapping[str, HighsOptionValue] | None,
) -> dict[str, HighsOptionValue]:
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
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
    if (
        isinstance(polyak_factor, bool)
        or not isinstance(polyak_factor, (int, float))
        or not math.isfinite(polyak_factor)
        or not 0 < polyak_factor <= 2
    ):
        raise ModelValidationError("polyak_factor must be finite and in (0, 2]")
    if highs_options is not None and not isinstance(highs_options, Mapping):
        raise ModelValidationError("highs_options must be a mapping or None")
    options = {} if highs_options is None else dict(highs_options)
    if not all(isinstance(name, str) for name in options):
        raise ModelValidationError("highs_options keys must be strings")
    if "time_limit" in options:
        raise ModelValidationError(
            "highs_options must not contain reserved option 'time_limit'; use time_limit"
        )
    if not all(isinstance(value, (bool, int, float, str)) for value in options.values()):
        raise ModelValidationError("HiGHS option values must be bool, int, float, or str")
    return options


def _validate_explicit_variable_bounds(model: ModelIR) -> None:
    missing_bounds: list[str] = []
    for variable in model.variables:
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


def _initial_multipliers(
    owner: Model,
    decomposition: Decomposition,
    supplied: Mapping[Constraint, float] | None,
) -> dict[str, float]:
    supplied_values = {} if supplied is None else dict(supplied)
    relaxed_by_index = decomposition._relaxation_penalties
    resolved: dict[int, float] = {}
    for constraint, value in supplied_values.items():
        if not isinstance(constraint, Constraint) or constraint._owner is not owner:
            raise ModelValidationError("initial multiplier constraint belongs to another model")
        if constraint._index is None or constraint._index not in relaxed_by_index:
            raise ModelValidationError("initial multiplier constraint is not relaxed")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ModelValidationError(f"initial multiplier for {constraint.name!r} must be finite")
        resolved[constraint._index] = float(value)

    result: dict[str, float] = {}
    for constraint in decomposition._model.constraints:
        if constraint.index not in relaxed_by_index:
            continue
        penalty = relaxed_by_index[constraint.index]
        value = resolved.get(constraint.index, 0.0)
        lower = -penalty if constraint.sense is ConstraintSense.EQ else 0.0
        if not lower <= value <= penalty:
            raise ModelValidationError(
                f"initial multiplier for {constraint.name!r} must be in [{lower}, {penalty}]"
            )
        result[constraint.name] = value
    return result


def _prepare_subproblem(
    solver: CompiledSubproblem,
    model: ModelIR,
    relaxed: tuple[ConstraintIR, ...],
) -> _PreparedSubproblem:
    objective_by_index = {
        term.variable_index: _objective_sign(model) * term.coefficient
        for term in model.objective_terms
    }
    position = {index: local for local, index in enumerate(solver.variable_indices)}
    lagrangian_terms: list[tuple[int, str, float]] = []
    for constraint in relaxed:
        sign = _constraint_sign(constraint)
        for term in constraint.terms:
            local = position.get(term.variable_index)
            if local is not None:
                lagrangian_terms.append((local, constraint.name, sign * term.coefficient))
    return _PreparedSubproblem(
        solver=solver,
        base_objective=tuple(
            objective_by_index.get(index, 0.0) for index in solver.variable_indices
        ),
        lagrangian_terms=tuple(lagrangian_terms),
    )


def _solve_subproblems(
    subproblems: tuple[_PreparedSubproblem, ...],
    model: ModelIR,
    multipliers: Mapping[str, float],
    deadline: float | None,
    parallel: bool,
    workers: int | None,
) -> tuple[dict[str, float], tuple[SubproblemRecord, ...], float]:
    if parallel and len(subproblems) > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = tuple(
                executor.submit(
                    _solve_one_subproblem,
                    subproblem,
                    model,
                    multipliers,
                    deadline,
                )
                for subproblem in subproblems
            )
            solved = tuple(future.result() for future in futures)
    else:
        solved = tuple(
            _solve_one_subproblem(subproblem, model, multipliers, deadline)
            for subproblem in subproblems
        )
    values: dict[str, float] = {}
    records: list[SubproblemRecord] = []
    objective = 0.0
    for item in solved:
        values.update(item.values)
        records.append(item.record)
        objective += item.objective_value
    return values, tuple(records), objective


def _solve_one_subproblem(
    subproblem: _PreparedSubproblem,
    model: ModelIR,
    multipliers: Mapping[str, float],
    deadline: float | None,
) -> _SolvedSubproblem:
    coefficients = list(subproblem.base_objective)
    for local, constraint_name, coefficient in subproblem.lagrangian_terms:
        coefficients[local] += multipliers[constraint_name] * coefficient
    remaining_time = _remaining_time(deadline)
    if remaining_time is not None and remaining_time <= 0:
        raise _TimeLimitReached(subproblem.solver.id)
    started = time.monotonic()
    try:
        solution = subproblem.solver.solve(coefficients, remaining_time)
    except TimeoutError as error:
        if deadline is not None:
            raise _TimeLimitReached(subproblem.solver.id) from error
        raise SubproblemSolveError(
            f"subproblem {subproblem.solver.id} solver raised TimeoutError: {error}"
        ) from error
    except SubproblemSolveError:
        raise
    except Exception as error:
        raise SubproblemSolveError(
            f"subproblem {subproblem.solver.id} solver raised {type(error).__name__}: {error}"
        ) from error
    if solution.status is SubproblemStatus.TIME_LIMIT:
        raise _TimeLimitReached(subproblem.solver.id)
    if solution.status is not SubproblemStatus.OPTIMAL:
        raise SubproblemSolveError(
            f"subproblem {subproblem.solver.id} did not reach optimality: {solution.detail}"
        )
    if solution.objective_value is None or len(solution.variable_values) != len(
        subproblem.solver.variable_indices
    ):
        raise SubproblemSolveError(
            f"subproblem {subproblem.solver.id} returned an incomplete solution"
        )
    values: dict[str, float] = {}
    for index, value in zip(
        subproblem.solver.variable_indices, solution.variable_values, strict=True
    ):
        if not math.isfinite(value):
            raise SubproblemSolveError(
                f"subproblem {subproblem.solver.id} returned no finite value for "
                f"{model.variables[index].name!r}"
            )
        values[model.variables[index].name] = value
    return _SolvedSubproblem(
        values=values,
        record=SubproblemRecord(
            subproblem.solver.id,
            "Optimal",
            solution.objective_value,
            max(0.0, time.monotonic() - started),
        ),
        objective_value=solution.objective_value,
    )


def _remaining_time(deadline: float | None) -> float | None:
    return None if deadline is None else max(0.0, deadline - time.monotonic())


def _deadline_reached(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _normalized_objective_value(model: ModelIR, values: Mapping[str, float]) -> float:
    raw = model.objective_constant + sum(
        term.coefficient * values[model.variables[term.variable_index].name]
        for term in model.objective_terms
    )
    return _objective_sign(model) * raw


def _normalized_objective_constant(model: ModelIR) -> float:
    return _objective_sign(model) * model.objective_constant


def _normalized_constraint_value(
    model: ModelIR, constraint: ConstraintIR, values: Mapping[str, float]
) -> float:
    raw = constraint.constant + sum(
        term.coefficient * values[model.variables[term.variable_index].name]
        for term in constraint.terms
    )
    return _constraint_sign(constraint) * raw


def _normalized_constraint_constant(constraint: ConstraintIR) -> float:
    return _constraint_sign(constraint) * constraint.constant


def _constraint_sign(constraint: ConstraintIR) -> float:
    return -1.0 if constraint.sense is ConstraintSense.GE else 1.0


def _violation(constraint: ConstraintIR, normalized_value: float) -> float:
    if constraint.sense is ConstraintSense.EQ:
        return abs(normalized_value)
    return max(0.0, normalized_value)


def _update_multipliers(
    decomposition: Decomposition,
    current: Mapping[str, float],
    residuals: Mapping[str, float],
    step_size: float,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for constraint in decomposition._model.constraints:
        penalty = decomposition._relaxation_penalties.get(constraint.index)
        if penalty is None:
            continue
        lower = -penalty if constraint.sense is ConstraintSense.EQ else 0.0
        candidate = current[constraint.name] + step_size * residuals[constraint.name]
        result[constraint.name] = min(penalty, max(lower, candidate))
    return result


def _objective_sign(model: ModelIR) -> float:
    return 1.0 if model.sense is ObjectiveSense.MINIMIZE else -1.0


def _from_normalized(model: ModelIR, value: float) -> float:
    return _objective_sign(model) * value


def _relative_gap(primal: float, dual: float) -> float:
    if not math.isfinite(primal) or not math.isfinite(dual):
        return math.inf
    gap = max(0.0, primal - dual)
    return gap / max(1.0, abs(primal), abs(dual))

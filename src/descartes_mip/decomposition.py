"""Analyze a PuLP model and derive independent Lagrangian subproblems."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import pulp

from descartes_mip._model import (
    ConstraintIR,
    LinearTermIR,
    ModelIR,
    VariableIR,
)
from descartes_mip.exceptions import ModelValidationError

type ConstraintSense = Literal["<=", ">=", "=="]
type ProblemSense = Literal["minimize", "maximize"]


@dataclass(frozen=True, slots=True)
class SoftConstraint:
    """A PuLP constraint to relax with a positive L1 violation penalty."""

    constraint: pulp.LpConstraint
    penalty: float

    def __post_init__(self) -> None:
        if not isinstance(self.constraint, pulp.LpConstraint):
            raise ModelValidationError("constraint must be a PuLP LpConstraint")
        if isinstance(self.penalty, bool) or not isinstance(self.penalty, (int, float)):
            raise ModelValidationError("penalty must be a positive finite number")
        if not math.isfinite(self.penalty) or self.penalty <= 0:
            raise ModelValidationError("penalty must be a positive finite number")
        object.__setattr__(self, "penalty", float(self.penalty))


@dataclass(frozen=True, slots=True)
class LinearTerm:
    """A named coefficient in a diagnostic linear expression."""

    variable: str
    coefficient: float


@dataclass(frozen=True, slots=True)
class RelaxedTerm:
    """The portion of one relaxed constraint owned by a subproblem."""

    constraint: str
    terms: tuple[LinearTerm, ...]


@dataclass(frozen=True, slots=True)
class Subproblem:
    """Structural description of one automatically detected subproblem."""

    id: int
    variables: tuple[str, ...]
    hard_constraints: tuple[str, ...]
    objective_terms: tuple[LinearTerm, ...]
    relaxed_terms: tuple[RelaxedTerm, ...]


@dataclass(frozen=True, slots=True)
class RelaxedConstraint:
    """Normalized diagnostic description of a relaxed constraint."""

    name: str
    sense: ConstraintSense
    penalty: float
    constant: float
    variables: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Decomposition:
    """Immutable result of structural model decomposition."""

    problem_name: str
    problem_sense: ProblemSense
    objective_constant: float
    subproblems: tuple[Subproblem, ...]
    relaxed_constraints: tuple[RelaxedConstraint, ...]
    constant_hard_constraints: tuple[str, ...]
    _model: ModelIR = field(repr=False, compare=False)

    @property
    def variable_count(self) -> int:
        return len(self._model.variables)

    @property
    def subproblem_count(self) -> int:
        return len(self.subproblems)


def decompose(
    problem: pulp.LpProblem,
    *,
    relax: list[SoftConstraint] | tuple[SoftConstraint, ...],
) -> Decomposition:
    """Return the independent components produced by removing ``relax`` constraints.

    The objective, constraints, bounds, categories, status, and solution values of
    ``problem`` are not changed. PuLP's public ``variables()`` reader may normalize
    its private variable-registration cache and ordering.
    """
    if not isinstance(problem, pulp.LpProblem):
        raise ModelValidationError("problem must be a PuLP LpProblem")
    soft_constraints = tuple(relax)
    model = _read_model(problem)
    penalty_by_constraint = _validate_soft_constraints(model, soft_constraints)
    relaxed_indexes = frozenset(penalty_by_constraint)

    components = _connected_components(model, relaxed_indexes)
    variable_to_component = {
        variable_index: component_index
        for component_index, component in enumerate(components)
        for variable_index in component
    }

    hard_by_component: list[list[str]] = [[] for _ in components]
    constant_hard_constraints: list[str] = []
    for constraint_index, constraint in enumerate(model.constraints):
        if constraint_index in relaxed_indexes:
            continue
        if not constraint.terms:
            if not _constant_constraint_is_feasible(constraint):
                raise ModelValidationError(
                    f"constant hard constraint {constraint.name!r} is infeasible"
                )
            constant_hard_constraints.append(constraint.name)
            continue
        component_index = variable_to_component[constraint.terms[0].variable_index]
        hard_by_component[component_index].append(constraint.name)

    relaxed = tuple(
        RelaxedConstraint(
            name=model.constraints[index].name,
            sense=_constraint_sense(model.constraints[index].sense),
            penalty=penalty_by_constraint[index],
            constant=model.constraints[index].constant,
            variables=tuple(
                model.variables[term.variable_index].name for term in model.constraints[index].terms
            ),
        )
        for index in sorted(relaxed_indexes)
    )

    subproblems = tuple(
        _make_subproblem(
            component_id,
            variable_indexes,
            model,
            hard_by_component[component_id],
            relaxed_indexes,
        )
        for component_id, variable_indexes in enumerate(components)
    )
    return Decomposition(
        problem_name=model.name,
        problem_sense="minimize" if model.sense == pulp.LpMinimize else "maximize",
        objective_constant=model.objective_constant,
        subproblems=subproblems,
        relaxed_constraints=relaxed,
        constant_hard_constraints=tuple(constant_hard_constraints),
        _model=model,
    )


def _read_model(problem: pulp.LpProblem) -> ModelIR:
    if problem.sense not in (pulp.LpMinimize, pulp.LpMaximize):
        raise ModelValidationError(f"unsupported problem sense: {problem.sense!r}")

    source_variables = tuple(problem.variables())
    names = [variable.name for variable in source_variables]
    if any(name is None or name == "" for name in names):
        raise ModelValidationError("all variables must have non-empty names")
    if len(set(names)) != len(names):
        raise ModelValidationError("variable names must be unique")

    variables = tuple(_read_variable(variable) for variable in source_variables)
    variable_index_by_identity = {
        id(variable): index for index, variable in enumerate(source_variables)
    }

    objective = problem.objective
    objective_constant = 0.0 if objective is None else _finite(objective.constant, "objective")
    objective_terms = (
        ()
        if objective is None
        else _read_terms(objective.items(), variable_index_by_identity, "objective")
    )

    constraints = tuple(
        ConstraintIR(
            name=_constraint_name(constraint, index),
            sense=_validated_constraint_sense(constraint),
            constant=_finite(constraint.constant, f"constraint {constraint.name!r}"),
            terms=_read_terms(
                constraint.items(),
                variable_index_by_identity,
                f"constraint {constraint.name!r}",
            ),
            source=constraint,
        )
        for index, constraint in enumerate(problem.constraints())
    )
    return ModelIR(
        name=problem.name,
        sense=problem.sense,
        variables=variables,
        objective_constant=objective_constant,
        objective_terms=objective_terms,
        constraints=constraints,
        source=problem,
    )


def _read_terms(items, variable_indexes: dict[int, int], owner: str) -> tuple[LinearTermIR, ...]:
    terms: list[LinearTermIR] = []
    for variable, coefficient in items:
        try:
            variable_index = variable_indexes[id(variable)]
        except KeyError as error:
            raise ModelValidationError(f"{owner} references an unregistered variable") from error
        terms.append(LinearTermIR(variable_index, _finite(coefficient, owner)))
    return tuple(sorted(terms, key=lambda term: term.variable_index))


def _read_variable(variable: pulp.LpVariable) -> VariableIR:
    low_bound = _finite_bound(variable.lowBound, variable.name, "lower")
    up_bound = _finite_bound(variable.upBound, variable.name, "upper")
    if low_bound is not None and up_bound is not None and low_bound > up_bound:
        raise ModelValidationError(f"variable {variable.name!r} has inconsistent bounds")
    return VariableIR(
        name=variable.name,
        low_bound=low_bound,
        up_bound=up_bound,
        category=variable.cat,
        source=variable,
    )


def _validate_soft_constraints(
    model: ModelIR, soft_constraints: tuple[SoftConstraint, ...]
) -> dict[int, float]:
    constraint_index_by_identity = {
        id(constraint.source): index for index, constraint in enumerate(model.constraints)
    }
    penalties: dict[int, float] = {}
    for soft in soft_constraints:
        try:
            index = constraint_index_by_identity[id(soft.constraint)]
        except KeyError as error:
            raise ModelValidationError(
                "every relaxed constraint must belong to the supplied problem"
            ) from error
        if index in penalties:
            name = model.constraints[index].name
            raise ModelValidationError(f"relaxed constraint {name!r} was specified more than once")
        penalties[index] = soft.penalty
    return penalties


def _connected_components(model: ModelIR, relaxed: frozenset[int]) -> tuple[tuple[int, ...], ...]:
    parent = list(range(len(model.variables)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for constraint_index, constraint in enumerate(model.constraints):
        if constraint_index in relaxed or len(constraint.terms) < 2:
            continue
        first = constraint.terms[0].variable_index
        for term in constraint.terms[1:]:
            union(first, term.variable_index)

    grouped: dict[int, list[int]] = {}
    for variable_index in range(len(model.variables)):
        grouped.setdefault(find(variable_index), []).append(variable_index)
    groups = list(grouped.values())
    groups.sort(key=lambda group: tuple(model.variables[index].name for index in group))
    return tuple(tuple(group) for group in groups)


def _make_subproblem(
    component_id: int,
    variable_indexes: tuple[int, ...],
    model: ModelIR,
    hard_constraints: list[str],
    relaxed_indexes: frozenset[int],
) -> Subproblem:
    owned = frozenset(variable_indexes)
    objective_terms = tuple(
        LinearTerm(model.variables[term.variable_index].name, term.coefficient)
        for term in model.objective_terms
        if term.variable_index in owned
    )
    relaxed_terms = tuple(
        RelaxedTerm(
            constraint=model.constraints[constraint_index].name,
            terms=tuple(
                LinearTerm(model.variables[term.variable_index].name, term.coefficient)
                for term in model.constraints[constraint_index].terms
                if term.variable_index in owned
            ),
        )
        for constraint_index in sorted(relaxed_indexes)
        if any(term.variable_index in owned for term in model.constraints[constraint_index].terms)
    )
    return Subproblem(
        id=component_id,
        variables=tuple(model.variables[index].name for index in variable_indexes),
        hard_constraints=tuple(hard_constraints),
        objective_terms=objective_terms,
        relaxed_terms=relaxed_terms,
    )


def _constant_constraint_is_feasible(constraint: ConstraintIR) -> bool:
    if constraint.sense == pulp.LpConstraintLE:
        return constraint.constant <= 0
    if constraint.sense == pulp.LpConstraintGE:
        return constraint.constant >= 0
    return constraint.constant == 0


def _constraint_name(constraint: pulp.LpConstraint, index: int) -> str:
    if constraint.name is None or constraint.name == "":
        raise ModelValidationError(f"constraint at index {index} must have a non-empty name")
    return constraint.name


def _validated_constraint_sense(constraint: pulp.LpConstraint) -> int:
    if constraint.sense not in (
        pulp.LpConstraintLE,
        pulp.LpConstraintGE,
        pulp.LpConstraintEQ,
    ):
        raise ModelValidationError(
            f"constraint {constraint.name!r} has unsupported sense {constraint.sense!r}"
        )
    return constraint.sense


def _constraint_sense(sense: int) -> ConstraintSense:
    if sense == pulp.LpConstraintLE:
        return "<="
    if sense == pulp.LpConstraintGE:
        return ">="
    return "=="


def _finite(value: float, owner: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ModelValidationError(f"{owner} contains a non-finite value")
    return result


def _finite_bound(value: float | None, variable_name: str, side: str) -> float | None:
    if value is None:
        return None
    try:
        return _finite(value, f"{side} bound of variable {variable_name!r}")
    except (TypeError, ValueError) as error:
        raise ModelValidationError(
            f"{side} bound of variable {variable_name!r} must be finite"
        ) from error

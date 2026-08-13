"""Immutable linear-model representation used by decomposition and solving."""

from dataclasses import dataclass

from pulp import LpConstraint, LpProblem, LpVariable


@dataclass(frozen=True, slots=True)
class VariableIR:
    name: str
    low_bound: float | None
    up_bound: float | None
    category: str
    source: LpVariable


@dataclass(frozen=True, slots=True)
class LinearTermIR:
    variable_index: int
    coefficient: float


@dataclass(frozen=True, slots=True)
class ConstraintIR:
    name: str
    sense: int
    constant: float
    terms: tuple[LinearTermIR, ...]
    source: LpConstraint


@dataclass(frozen=True, slots=True)
class ModelIR:
    name: str
    sense: int
    variables: tuple[VariableIR, ...]
    objective_constant: float
    objective_terms: tuple[LinearTermIR, ...]
    constraints: tuple[ConstraintIR, ...]
    source: LpProblem

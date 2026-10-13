"""Solver-independent immutable representation of a linear model."""

from dataclasses import dataclass
from enum import Enum


class VariableType(Enum):
    """Supported decision-variable domains."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


class ObjectiveSense(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ConstraintSense(Enum):
    LE = "<="
    GE = ">="
    EQ = "=="


@dataclass(frozen=True, slots=True)
class VariableIR:
    index: int
    name: str
    low_bound: float | None
    up_bound: float | None
    vartype: VariableType


@dataclass(frozen=True, slots=True)
class LinearTermIR:
    variable_index: int
    coefficient: float


@dataclass(frozen=True, slots=True)
class ConstraintIR:
    index: int
    name: str
    sense: ConstraintSense
    constant: float
    terms: tuple[LinearTermIR, ...]


@dataclass(frozen=True, slots=True)
class ModelIR:
    name: str
    sense: ObjectiveSense
    variables: tuple[VariableIR, ...]
    objective_constant: float
    objective_terms: tuple[LinearTermIR, ...]
    constraints: tuple[ConstraintIR, ...]


@dataclass(frozen=True, slots=True)
class RelaxationIR:
    constraint_index: int
    penalty: float


@dataclass(frozen=True, slots=True)
class SubproblemIR:
    id: int
    variable_indices: tuple[int, ...]
    hard_constraint_indices: tuple[int, ...]

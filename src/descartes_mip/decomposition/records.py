"""Immutable public records describing an automatic decomposition."""

from dataclasses import dataclass, field
from typing import Literal

from descartes_mip.modeling.ir import ModelIR, SubproblemIR

type ConstraintSenseName = Literal["<=", ">=", "=="]
type ProblemSenseName = Literal["minimize", "maximize"]


@dataclass(frozen=True, slots=True)
class LinearTerm:
    variable: str
    coefficient: float


@dataclass(frozen=True, slots=True)
class RelaxedTerm:
    constraint: str
    terms: tuple[LinearTerm, ...]


@dataclass(frozen=True, slots=True)
class Subproblem:
    id: int
    variables: tuple[str, ...]
    hard_constraints: tuple[str, ...]
    objective_terms: tuple[LinearTerm, ...]
    relaxed_terms: tuple[RelaxedTerm, ...]


@dataclass(frozen=True, slots=True)
class RelaxedConstraint:
    name: str
    sense: ConstraintSenseName
    penalty: float
    constant: float
    variables: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Decomposition:
    problem_name: str
    problem_sense: ProblemSenseName
    objective_constant: float
    subproblems: tuple[Subproblem, ...]
    relaxed_constraints: tuple[RelaxedConstraint, ...]
    constant_hard_constraints: tuple[str, ...]
    _model: ModelIR = field(repr=False, compare=False)
    _subproblem_irs: tuple[SubproblemIR, ...] = field(repr=False, compare=False)
    _relaxation_penalties: dict[int, float] = field(repr=False, compare=False)

    @property
    def variable_count(self) -> int:
        return len(self._model.variables)

    @property
    def subproblem_count(self) -> int:
        return len(self.subproblems)

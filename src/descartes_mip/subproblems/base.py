"""Solver-neutral contracts for persistent subproblem implementations."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from descartes_mip.modeling.ir import ModelIR, SubproblemIR


class SubproblemStatus(Enum):
    OPTIMAL = "optimal"
    TIME_LIMIT = "time_limit"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SubproblemSolution:
    status: SubproblemStatus
    objective_value: float | None
    variable_values: tuple[float, ...]
    detail: str


class CompiledSubproblem(Protocol):
    id: int
    variable_indices: tuple[int, ...]

    def solve(
        self,
        objective_coefficients: Sequence[float],
        remaining_time: float | None,
    ) -> SubproblemSolution: ...


class SubproblemSolverFactory(Protocol):
    def compile(
        self,
        model: ModelIR,
        subproblem: SubproblemIR,
        options: Mapping[str, bool | int | float | str],
    ) -> CompiledSubproblem: ...

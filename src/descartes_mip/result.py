"""Public result records returned by the Lagrangian solver."""

from dataclasses import dataclass
from typing import Literal

from descartes_mip.decomposition import Decomposition

type TerminationReason = Literal["max_iterations", "time_limit", "callback"]


@dataclass(frozen=True, slots=True)
class SubproblemRecord:
    """Outcome of one subproblem solve within an iteration."""

    subproblem_id: int
    status: str
    objective_value: float
    elapsed_time: float


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """Certified values and update data for one completed iteration."""

    iteration: int
    primal_bound: float
    dual_bound: float
    best_primal_bound: float
    best_dual_bound: float
    absolute_gap: float
    step_size: float
    multipliers: dict[str, float]
    residuals: dict[str, float]
    violations: dict[str, float]
    variable_values: dict[str, float]
    subproblems: tuple[SubproblemRecord, ...]
    elapsed_time: float


@dataclass(frozen=True, slots=True)
class LagrangianResult:
    """Best certified bounds, soft-feasible solution, and iteration history."""

    termination_reason: TerminationReason
    iterations: int
    best_primal_bound: float
    best_dual_bound: float
    absolute_gap: float
    relative_gap: float
    variable_values: dict[str, float]
    slacks: dict[str, float]
    hard_feasible: bool
    soft_feasible: bool
    has_certified_bounds: bool
    multipliers: dict[str, float]
    history: tuple[IterationRecord, ...]
    decomposition: Decomposition
    elapsed_time: float
    timed_out_subproblem: int | None

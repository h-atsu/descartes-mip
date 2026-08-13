from collections.abc import Callable
from typing import Any, cast

import pulp
import pytest

from descartes_mip import (
    IterationRecord,
    ModelValidationError,
    SoftConstraint,
    SubproblemSolveError,
    highs_factory,
    solve,
)


def build_binary_pair(
    *,
    sense: int,
    objective: tuple[float, float],
    constraint_sense: str,
) -> tuple[pulp.LpProblem, pulp.LpConstraint]:
    problem = pulp.LpProblem("binary_pair", sense)
    x = problem.add_variable("x", lowBound=0, upBound=1, cat=pulp.LpBinary)
    y = problem.add_variable("y", lowBound=0, upBound=1, cat=pulp.LpBinary)
    problem += objective[0] * x + objective[1] * y
    if constraint_sense == ">=":
        coupling = x + y >= 1
    elif constraint_sense == "<=":
        coupling = x + y <= 1
    else:
        coupling = x + y == 1
    problem += coupling, "coupling"
    return problem, coupling


def test_solve_ge_constraint_finds_matching_certified_bounds() -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )

    result = solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=3)],
        initial_multipliers={"coupling": 1.5},
        max_iterations=1,
    )

    assert result.best_primal_bound == pytest.approx(1)
    assert result.best_dual_bound == pytest.approx(1)
    assert result.absolute_gap == pytest.approx(0)
    assert result.relative_gap == pytest.approx(0)
    assert result.variable_values == {"x": 1.0, "y": 0.0}
    assert result.slacks == {"coupling": 0.0}
    assert result.hard_feasible
    assert result.soft_feasible

    direct = pulp.LpProblem("direct_soft_model", pulp.LpMinimize)
    x = direct.add_variable("x", lowBound=0, upBound=1, cat=pulp.LpBinary)
    y = direct.add_variable("y", lowBound=0, upBound=1, cat=pulp.LpBinary)
    slack = direct.add_variable("slack", lowBound=0)
    direct += x + 2 * y + 3 * slack
    direct += x + y + slack >= 1
    assert direct.solve(highs_factory(None)) == pulp.LpStatusOptimal
    assert result.best_primal_bound == pytest.approx(pulp.value(direct.objective))


def test_solve_le_constraint_uses_excess_violation() -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(-3, -2),
        constraint_sense="<=",
    )

    result = solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=5)],
        initial_multipliers={"coupling": 2.5},
        max_iterations=1,
    )

    assert result.best_primal_bound == pytest.approx(-3)
    assert result.best_dual_bound == pytest.approx(-3)
    assert result.variable_values == {"x": 1.0, "y": 0.0}
    assert result.history[0].residuals == {"coupling": 0.0}


def test_solve_equality_uses_symmetric_multiplier_box() -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense="==",
    )

    result = solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=4)],
        initial_multipliers={"coupling": -1.5},
        max_iterations=1,
    )

    assert result.best_primal_bound == pytest.approx(1)
    assert result.best_dual_bound == pytest.approx(1)
    assert result.multipliers == {"coupling": -1.5}
    assert result.variable_values == {"x": 1.0, "y": 0.0}


def test_solve_maximization_reports_bounds_in_original_sense() -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMaximize,
        objective=(3, 2),
        constraint_sense="<=",
    )

    result = solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=5)],
        initial_multipliers={"coupling": 2.5},
        max_iterations=1,
    )

    assert result.best_primal_bound == pytest.approx(3)
    assert result.best_dual_bound == pytest.approx(3)
    assert result.variable_values == {"x": 1.0, "y": 0.0}


def test_polyak_step_updates_and_projects_multiplier() -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 1),
        constraint_sense=">=",
    )

    result = solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=3)],
        max_iterations=1,
    )

    record = result.history[0]
    assert record.primal_bound == pytest.approx(3)
    assert record.dual_bound == pytest.approx(0)
    assert record.residuals == {"coupling": 1.0}
    assert record.violations == {"coupling": 1.0}
    assert record.step_size == pytest.approx(3)
    assert result.multipliers == {"coupling": 3.0}


def test_callback_stops_after_a_completed_iteration() -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )
    observed: list[IterationRecord] = []

    def stop(record: IterationRecord) -> bool:
        observed.append(record)
        return True

    result = solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=3)],
        max_iterations=10,
        callback=stop,
    )

    assert result.termination_reason == "callback"
    assert result.iterations == 1
    assert observed == [result.history[0]]


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("max_iterations", 0, "positive integer"),
        ("max_iterations", True, "positive integer"),
        ("polyak_factor", 0, "in \\(0, 2]"),
        ("polyak_factor", 3, "in \\(0, 2]"),
    ],
)
def test_solve_rejects_invalid_options(keyword: str, value: Any, message: str) -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )
    options = {keyword: value}

    with pytest.raises(ModelValidationError, match=message):
        solve(problem, relax=[SoftConstraint(coupling, penalty=3)], **options)


def test_solve_rejects_initial_multiplier_outside_box() -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )

    with pytest.raises(ModelValidationError, match="must be in"):
        solve(
            problem,
            relax=[SoftConstraint(coupling, penalty=3)],
            initial_multipliers={"coupling": -1},
            max_iterations=1,
        )


def test_solve_rejects_a_variable_without_explicit_bounds() -> None:
    problem = pulp.LpProblem("unbounded", pulp.LpMinimize)
    x = problem.add_variable("x")
    problem += -x

    with pytest.raises(
        ModelValidationError,
        match=r"all decision variables.*x \(lower, upper\)",
    ):
        solve(problem, relax=[], max_iterations=1)


def test_solver_factory_is_called_once_per_subproblem_and_iteration() -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )
    calls: list[float | None] = []

    def factory(remaining_time: float | None) -> pulp.LpSolver:
        calls.append(remaining_time)
        return highs_factory(remaining_time)

    solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=3)],
        solver_factory=factory,
        max_iterations=2,
    )

    assert calls == [None, None, None, None]


def test_solve_wraps_solver_factory_error() -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )

    def broken_factory(_: float | None) -> pulp.LpSolver:
        raise RuntimeError("broken")

    with pytest.raises(SubproblemSolveError, match="solver raised RuntimeError: broken"):
        solve(
            problem,
            relax=[SoftConstraint(coupling, penalty=3)],
            solver_factory=broken_factory,
            max_iterations=1,
        )


def test_solve_rejects_non_callable_solver_factory() -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )

    with pytest.raises(ModelValidationError, match="solver_factory"):
        solve(
            problem,
            relax=[SoftConstraint(coupling, penalty=3)],
            solver_factory=cast(Callable[[float | None], pulp.LpSolver], None),
            max_iterations=1,
        )


def test_solve_assigns_a_deterministic_value_to_zero_cost_free_variable() -> None:
    problem = pulp.LpProblem("free_zero_cost", pulp.LpMinimize)
    x = problem.add_variable("x", lowBound=2, upBound=5)
    coupling = x <= 4
    problem += coupling, "coupling"

    result = solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=1)],
        max_iterations=1,
    )

    assert result.variable_values == {"x": 2.0}
    assert result.slacks == {"coupling": 0.0}


def test_solve_reconstructs_and_enforces_local_hard_constraints() -> None:
    problem = pulp.LpProblem("local_hard_constraints", pulp.LpMinimize)
    x = problem.add_variable("x", lowBound=0, upBound=10)
    y = problem.add_variable("y", lowBound=0, upBound=10)
    problem += x + y
    problem += x >= 2, "x_minimum"
    problem += y >= 3, "y_minimum"
    coupling = x + y <= 4
    problem += coupling, "coupling"

    result = solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=10)],
        initial_multipliers={"coupling": 5},
        max_iterations=1,
    )

    assert result.variable_values == {"x": 2.0, "y": 3.0}
    assert result.slacks == {"coupling": 1.0}
    assert result.best_primal_bound == pytest.approx(15)
    assert not result.hard_feasible


def test_solve_requires_explicit_bounds_even_when_hard_constraints_imply_them() -> None:
    problem = pulp.LpProblem("implicitly_bounded", pulp.LpMinimize)
    x = problem.add_variable("x", lowBound=0)
    y = problem.add_variable("y", lowBound=0)
    problem += x + y
    problem += x + y <= 10, "shared_upper_bound"

    with pytest.raises(ModelValidationError) as error:
        solve(problem, relax=[], max_iterations=1)

    assert "x (upper)" in str(error.value)
    assert "y (upper)" in str(error.value)


def test_solve_accepts_finite_explicit_bounds_on_every_variable() -> None:
    problem = pulp.LpProblem("explicitly_bounded", pulp.LpMinimize)
    x = problem.add_variable("x", lowBound=-2, upBound=4)
    y = problem.add_variable("y", lowBound=0, upBound=7, cat=pulp.LpInteger)
    problem += x + y

    result = solve(problem, relax=[], max_iterations=1)

    assert result.variable_values == {"x": -2.0, "y": 0.0}


def test_solve_preserves_original_model_solution_state() -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )
    before = (problem.status, tuple(variable.varValue for variable in problem.variables()))

    solve(problem, relax=[SoftConstraint(coupling, penalty=3)], max_iterations=2)

    after = (problem.status, tuple(variable.varValue for variable in problem.variables()))
    assert after == before

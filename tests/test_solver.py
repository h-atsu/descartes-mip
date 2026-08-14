import math
from typing import Any, cast

import pulp
import pytest

from descartes_mip import (
    SUPPORTED_SOLVERS,
    IterationRecord,
    ModelValidationError,
    SoftConstraint,
    SolverClass,
    SubproblemSolveError,
    solve,
)


class NotSolvedSolver(pulp.LpSolver):
    """Test solver that simulates a time-limited solve without an optimum."""

    def available(self) -> bool:
        return True

    def actualSolve(self, lp: pulp.LpProblem) -> int:
        lp.assignStatus(pulp.LpStatusNotSolved, pulp.LpSolutionNoSolutionFound)
        return pulp.LpStatusNotSolved


class FakeTime:
    def __init__(self, current: float = 0.0) -> None:
        self.current = current

    def monotonic(self) -> float:
        return self.current


def bundled_cbc_path() -> str:
    return pulp.PULP_CBC_CMD.pulp_cbc_path


def bundled_cbc_available() -> bool:
    return bool(pulp.COIN_CMD(path=bundled_cbc_path(), msg=False).available())


def test_supported_solver_classes_are_explicit() -> None:
    assert SUPPORTED_SOLVERS == (pulp.HiGHS, pulp.COIN_CMD)


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
    assert direct.solve(pulp.HiGHS(msg=False)) == pulp.LpStatusOptimal
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
        ("time_limit", 0, "positive finite"),
        ("time_limit", float("inf"), "positive finite"),
        ("time_limit", True, "positive finite"),
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


def test_solver_class_is_instantiated_once_per_subproblem_and_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import descartes_mip.solver as solver_module

    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )
    calls: list[tuple[SolverClass, dict[str, object], float | None]] = []
    original_create_solver = solver_module.create_solver

    def create_solver(
        solver: SolverClass,
        options: dict[str, object],
        remaining_time: float | None,
    ) -> pulp.LpSolver:
        calls.append((solver, options, remaining_time))
        return original_create_solver(solver, options, remaining_time)

    monkeypatch.setattr(solver_module, "create_solver", create_solver)

    solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=3)],
        solver_options={"gapRel": 0.0},
        max_iterations=2,
    )

    assert calls == [(pulp.HiGHS, {"gapRel": 0.0}, None)] * 4


def test_time_limit_passes_decreasing_remaining_budget_to_each_subproblem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import descartes_mip.solver as solver_module

    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )
    clock = FakeTime(100)
    monkeypatch.setattr(solver_module, "time", clock)
    remaining_budgets: list[float | None] = []

    original_create_solver = solver_module.create_solver

    def create_solver(
        solver: SolverClass,
        options: dict[str, object],
        remaining_time: float | None,
    ) -> pulp.LpSolver:
        remaining_budgets.append(remaining_time)
        clock.current += 1
        return original_create_solver(solver, options, remaining_time)

    monkeypatch.setattr(solver_module, "create_solver", create_solver)

    result = solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=3)],
        max_iterations=1,
        time_limit=10,
    )

    assert remaining_budgets == pytest.approx([10, 9])
    assert result.elapsed_time == pytest.approx(2)
    assert result.history[0].elapsed_time == pytest.approx(2)
    assert [record.elapsed_time for record in result.history[0].subproblems] == pytest.approx(
        [1, 1]
    )


def test_time_limit_discards_an_incomplete_iteration_and_keeps_prior_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import descartes_mip.solver as solver_module

    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )
    clock = FakeTime()
    monkeypatch.setattr(solver_module, "time", clock)
    calls = 0

    original_create_solver = solver_module.create_solver

    def create_solver(
        solver: SolverClass,
        options: dict[str, object],
        remaining_time: float | None,
    ) -> pulp.LpSolver:
        nonlocal calls
        calls += 1
        clock.current += 1
        if calls == 4:
            return NotSolvedSolver(msg=False, timeLimit=remaining_time)
        return original_create_solver(solver, options, remaining_time)

    monkeypatch.setattr(solver_module, "create_solver", create_solver)

    result = solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=3)],
        max_iterations=10,
        time_limit=100,
    )

    assert result.termination_reason == "time_limit"
    assert result.iterations == 1
    assert len(result.history) == 1
    assert result.timed_out_subproblem == 1
    assert result.has_certified_bounds
    assert result.soft_feasible
    assert result.best_primal_bound == result.history[0].best_primal_bound
    assert result.best_dual_bound == result.history[0].best_dual_bound
    assert result.multipliers == {"coupling": 3.0}


def test_time_limit_before_first_iteration_returns_no_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import descartes_mip.solver as solver_module

    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )

    class ExpiredTime:
        calls = 0

        def monotonic(self) -> float:
            self.calls += 1
            return 0.0 if self.calls == 1 else 2.0

    monkeypatch.setattr(solver_module, "time", ExpiredTime())

    result = solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=3)],
        max_iterations=10,
        time_limit=1,
    )

    assert result.termination_reason == "time_limit"
    assert result.iterations == 0
    assert not result.has_certified_bounds
    assert not result.soft_feasible
    assert not result.hard_feasible
    assert result.variable_values == {}
    assert result.slacks == {}
    assert result.best_primal_bound == math.inf
    assert result.best_dual_bound == -math.inf
    assert result.absolute_gap == math.inf
    assert result.relative_gap == math.inf
    assert result.timed_out_subproblem is None


def test_completed_iteration_is_kept_when_solver_slightly_exceeds_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import descartes_mip.solver as solver_module

    problem = pulp.LpProblem("one_component", pulp.LpMinimize)
    x = problem.add_variable("x", lowBound=0, upBound=1)
    problem += x
    clock = FakeTime()
    monkeypatch.setattr(solver_module, "time", clock)

    original_create_solver = solver_module.create_solver

    def create_solver(
        solver: SolverClass,
        options: dict[str, object],
        remaining_time: float | None,
    ) -> pulp.LpSolver:
        clock.current += 2
        return original_create_solver(solver, options, remaining_time)

    monkeypatch.setattr(solver_module, "create_solver", create_solver)

    result = solve(
        problem,
        relax=[],
        max_iterations=10,
        time_limit=1,
    )

    assert result.termination_reason == "time_limit"
    assert result.iterations == 1
    assert result.has_certified_bounds
    assert result.variable_values == {"x": 0.0}
    assert result.elapsed_time == pytest.approx(2)


def test_not_solved_status_without_time_limit_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import descartes_mip.solver as solver_module

    problem = pulp.LpProblem("not_solved", pulp.LpMinimize)
    x = problem.add_variable("x", lowBound=0, upBound=1)
    problem += x

    monkeypatch.setattr(
        solver_module,
        "create_solver",
        lambda solver, options, remaining: NotSolvedSolver(msg=False),
    )

    with pytest.raises(SubproblemSolveError, match="Not Solved"):
        solve(problem, relax=[], max_iterations=1)


def test_infeasible_subproblem_is_not_misreported_as_time_limit() -> None:
    problem = pulp.LpProblem("infeasible", pulp.LpMinimize)
    x = problem.add_variable("x", lowBound=0, upBound=1)
    problem += x
    problem += x >= 2, "impossible"

    with pytest.raises(SubproblemSolveError, match="subproblem 0.*Infeasible"):
        solve(problem, relax=[], max_iterations=1, time_limit=10)


def test_parallel_solve_matches_sequential_certified_results() -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )

    sequential = solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=3)],
        max_iterations=4,
    )
    parallel = solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=3)],
        max_iterations=4,
        parallel=True,
        workers=2,
    )

    assert parallel.best_primal_bound == pytest.approx(sequential.best_primal_bound)
    assert parallel.best_dual_bound == pytest.approx(sequential.best_dual_bound)
    assert parallel.absolute_gap == pytest.approx(sequential.absolute_gap)
    assert parallel.variable_values == sequential.variable_values
    assert parallel.slacks == sequential.slacks
    assert parallel.multipliers == sequential.multipliers
    assert [
        [subproblem.subproblem_id for subproblem in record.subproblems]
        for record in parallel.history
    ] == [[0, 1]] * 4
    assert [record.dual_bound for record in parallel.history] == pytest.approx(
        [record.dual_bound for record in sequential.history]
    )


def test_parallel_solve_creates_an_independent_solver_for_each_subproblem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import descartes_mip.solver as solver_module

    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )
    solvers: list[pulp.LpSolver] = []
    original_create_solver = solver_module.create_solver

    def create_solver(
        solver: SolverClass,
        options: dict[str, object],
        remaining_time: float | None,
    ) -> pulp.LpSolver:
        instance = original_create_solver(solver, options, remaining_time)
        solvers.append(instance)
        return instance

    monkeypatch.setattr(solver_module, "create_solver", create_solver)

    solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=3)],
        max_iterations=1,
        parallel=True,
        workers=2,
    )

    assert len(solvers) == 2
    assert solvers[0] is not solvers[1]


def test_parallel_time_limit_discards_the_whole_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import descartes_mip.solver as solver_module

    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )

    monkeypatch.setattr(
        solver_module,
        "create_solver",
        lambda solver, options, remaining: NotSolvedSolver(
            msg=False,
            timeLimit=remaining,
        ),
    )

    result = solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=3)],
        max_iterations=2,
        time_limit=10,
        parallel=True,
        workers=2,
    )

    assert result.termination_reason == "time_limit"
    assert result.iterations == 0
    assert result.timed_out_subproblem == 0
    assert not result.has_certified_bounds
    assert result.history == ()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"parallel": "yes"}, "parallel must be a boolean"),
        ({"parallel": True, "workers": 0}, "positive integer"),
        ({"parallel": True, "workers": True}, "positive integer"),
        ({"workers": 2}, "requires parallel"),
    ],
)
def test_solve_rejects_invalid_parallel_options(options: dict[str, Any], message: str) -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )

    with pytest.raises(ModelValidationError, match=message):
        solve(problem, relax=[SoftConstraint(coupling, penalty=3)], **options)


@pytest.mark.skipif(not bundled_cbc_available(), reason="CBC is unavailable")
def test_cbc_solver_class_is_compatible() -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )

    result = solve(
        problem,
        relax=[SoftConstraint(coupling, penalty=3)],
        solver=pulp.COIN_CMD,
        solver_options={"path": bundled_cbc_path()},
        initial_multipliers={"coupling": 1.5},
        max_iterations=1,
        parallel=True,
        workers=2,
    )

    assert result.best_primal_bound == pytest.approx(1)
    assert result.best_dual_bound == pytest.approx(1)
    assert result.has_certified_bounds


def test_solve_wraps_solver_construction_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import descartes_mip.solver as solver_module

    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )

    def broken_create_solver(
        solver: SolverClass,
        options: dict[str, object],
        remaining_time: float | None,
    ) -> pulp.LpSolver:
        raise RuntimeError("broken")

    monkeypatch.setattr(solver_module, "create_solver", broken_create_solver)

    with pytest.raises(SubproblemSolveError, match="solver raised RuntimeError: broken"):
        solve(problem, relax=[SoftConstraint(coupling, penalty=3)], max_iterations=1)


@pytest.mark.parametrize("unsupported", [pulp.GLPK_CMD, NotSolvedSolver])
def test_solve_rejects_unsupported_solver_class(unsupported: SolverClass) -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )

    with pytest.raises(ModelValidationError, match="unsupported solver.*HiGHS.*COIN_CMD"):
        solve(
            problem,
            relax=[SoftConstraint(coupling, penalty=3)],
            solver=unsupported,
            max_iterations=1,
        )


def test_solve_rejects_a_solver_instance() -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )

    with pytest.raises(ModelValidationError, match="unsupported solver"):
        solve(
            problem,
            relax=[SoftConstraint(coupling, penalty=3)],
            solver=cast(SolverClass, pulp.HiGHS(msg=False)),
            max_iterations=1,
        )


@pytest.mark.parametrize(
    ("solver_options", "message"),
    [
        ([], "mapping or None"),
        ({1: "invalid"}, "keys must be strings"),
        ({"timeLimit": 1}, "reserved option 'timeLimit'"),
    ],
)
def test_solve_rejects_invalid_solver_options(
    solver_options: Any,
    message: str,
) -> None:
    problem, coupling = build_binary_pair(
        sense=pulp.LpMinimize,
        objective=(1, 2),
        constraint_sense=">=",
    )

    with pytest.raises(ModelValidationError, match=message):
        solve(
            problem,
            relax=[SoftConstraint(coupling, penalty=3)],
            solver_options=solver_options,
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

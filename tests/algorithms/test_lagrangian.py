import math
from typing import Any

import pytest

from descartes_mip import (
    IterationRecord,
    Model,
    ModelValidationError,
    SubproblemSolveError,
    VariableType,
)


def build_binary_pair(
    objective: tuple[float, float] = (1, 2),
    sense: str = ">=",
    maximize: bool = False,
):
    model = Model("binary_pair")
    x = model.add_var("x", vartype=VariableType.BINARY)
    y = model.add_var("y", vartype=VariableType.BINARY)
    (model.maximize if maximize else model.minimize)(objective[0] * x + objective[1] * y)
    expression = x + y
    draft = (
        expression >= 1 if sense == ">=" else expression <= 1 if sense == "<=" else expression == 1
    )
    coupling = model.add_constraint(draft, "coupling")
    return model, x, y, coupling


def test_ge_constraint_finds_matching_certified_bounds_and_handle_values() -> None:
    model, x, y, coupling = build_binary_pair()
    model.relax(coupling, penalty=3)
    result = model.solve(initial_multipliers={coupling: 1.5}, max_iterations=1)

    assert result.best_primal_bound == pytest.approx(1)
    assert result.best_dual_bound == pytest.approx(1)
    assert result.value(x) == 1
    assert result.value(y) == 0
    assert result.violation(coupling) == 0
    assert result.multiplier(coupling) == 1.5
    assert result.variable_values == {"x": 1.0, "y": 0.0}
    assert result.violations == {"coupling": 0.0}
    assert result.hard_feasible


def test_le_and_equality_violation_and_multiplier_boxes() -> None:
    le_model, _, _, le = build_binary_pair(objective=(-3, -2), sense="<=")
    le_model.relax(le, penalty=5)
    le_result = le_model.solve(initial_multipliers={le: 2.5}, max_iterations=1)
    assert le_result.best_primal_bound == pytest.approx(-3)
    assert le_result.violation(le) == 0

    eq_model, _, _, eq = build_binary_pair(sense="==")
    eq_model.relax(eq, penalty=4)
    eq_result = eq_model.solve(initial_multipliers={eq: -1.5}, max_iterations=1)
    assert eq_result.best_primal_bound == pytest.approx(1)
    assert eq_result.multiplier(eq) == -1.5


def test_maximization_reports_bounds_in_original_sense() -> None:
    model, x, _, coupling = build_binary_pair(objective=(3, 2), sense="<=", maximize=True)
    model.relax(coupling, penalty=5)
    result = model.solve(initial_multipliers={coupling: 2.5}, max_iterations=1)
    assert result.best_primal_bound == pytest.approx(3)
    assert result.best_dual_bound == pytest.approx(3)
    assert result.value(x) == 1


def test_polyak_step_updates_and_projects_multiplier() -> None:
    model, _, _, coupling = build_binary_pair(objective=(1, 1))
    model.relax(coupling, penalty=3)
    result = model.solve(max_iterations=1)
    record = result.history[0]
    assert record.primal_bound == pytest.approx(3)
    assert record.dual_bound == pytest.approx(0)
    assert record.residuals == {"coupling": 1.0}
    assert record.violations == {"coupling": 1.0}
    assert record.step_size == pytest.approx(3)
    assert result.multiplier(coupling) == 3


def test_callback_stops_after_completed_iteration() -> None:
    model, _, _, coupling = build_binary_pair()
    model.relax(coupling, penalty=3)
    observed: list[IterationRecord] = []

    def stop(record: IterationRecord) -> bool:
        observed.append(record)
        return True

    result = model.solve(max_iterations=10, callback=stop)
    assert result.termination_reason == "callback"
    assert observed == [result.history[0]]


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("max_iterations", 0, "positive integer"),
        ("time_limit", 0, "positive finite"),
        ("polyak_factor", 0, "in \\(0, 2]"),
        ("workers", 2, "parallel=True"),
    ],
)
def test_invalid_solve_options(keyword: str, value: Any, message: str) -> None:
    model, _, _, coupling = build_binary_pair()
    model.relax(coupling, penalty=3)
    with pytest.raises(ModelValidationError, match=message):
        model.solve(**{keyword: value})


def test_missing_explicit_bounds_are_reported() -> None:
    model = Model()
    x = model.add_var("x")
    model.minimize(-x)
    with pytest.raises(ModelValidationError, match=r"x \(lower, upper\)"):
        model.solve(max_iterations=1)


def test_initial_multiplier_uses_constraint_handles_and_validates_box() -> None:
    model, _, _, coupling = build_binary_pair()
    model.relax(coupling, penalty=3)
    with pytest.raises(ModelValidationError, match="must be in"):
        model.solve(initial_multipliers={coupling: -1}, max_iterations=1)

    foreign, _, _, foreign_constraint = build_binary_pair()
    foreign.relax(foreign_constraint, penalty=3)
    with pytest.raises(ModelValidationError, match="another model"):
        model.solve(initial_multipliers={foreign_constraint: 1}, max_iterations=1)


def test_result_rejects_foreign_new_and_non_relaxed_handles() -> None:
    model, x, _, coupling = build_binary_pair()
    hard = model.add_constraint(x <= 1, "hard")
    model.relax(coupling, penalty=3)
    result = model.solve(max_iterations=1)
    new_variable = model.add_var("later", lb=0, ub=1)
    foreign = Model().add_var(lb=0, ub=1)

    with pytest.raises(ModelValidationError, match="after"):
        result.value(new_variable)
    with pytest.raises(ModelValidationError, match="solved model"):
        result.value(foreign)
    with pytest.raises(ModelValidationError, match="not relaxed"):
        result.violation(hard)


def test_same_variable_handle_can_compare_multiple_results() -> None:
    model, x, _, coupling = build_binary_pair()
    model.relax(coupling, penalty=3)
    first = model.solve(initial_multipliers={coupling: 0}, max_iterations=1)
    second = model.solve(initial_multipliers={coupling: 2}, max_iterations=1)

    assert first.value(x) == 0
    assert second.value(x) == 1
    assert not hasattr(first, "slacks")


def test_parallel_and_sequential_results_match() -> None:
    first, _, _, coupling = build_binary_pair()
    first.relax(coupling, penalty=3)
    sequential = first.solve(max_iterations=3)

    second, _, _, second_coupling = build_binary_pair()
    second.relax(second_coupling, penalty=3)
    parallel = second.solve(max_iterations=3, parallel=True, workers=2)

    assert parallel.best_primal_bound == pytest.approx(sequential.best_primal_bound)
    assert parallel.best_dual_bound == pytest.approx(sequential.best_dual_bound)
    assert parallel.variable_values == sequential.variable_values


def test_each_subproblem_is_compiled_only_once_across_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from descartes_mip.algorithms import lagrangian

    model, _, _, coupling = build_binary_pair()
    model.relax(coupling, penalty=3)
    calls: list[int] = []
    original = lagrangian.HighsSubproblemFactory.compile

    def compile_once(self, model_ir, subproblem, options):
        calls.append(subproblem.id)
        return original(self, model_ir, subproblem, options)

    monkeypatch.setattr(lagrangian.HighsSubproblemFactory, "compile", compile_once)
    model.solve(max_iterations=3)

    assert calls == [0, 1]


def test_timed_out_subproblem_discards_incomplete_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from descartes_mip.algorithms import lagrangian
    from descartes_mip.subproblems.base import SubproblemSolution, SubproblemStatus

    model, _, _, coupling = build_binary_pair()
    model.relax(coupling, penalty=3)

    class TimedOutSubproblem:
        def __init__(self, subproblem) -> None:
            self.id = subproblem.id
            self.variable_indices = subproblem.variable_indices

        def solve(self, objective_coefficients, remaining_time):
            return SubproblemSolution(SubproblemStatus.TIME_LIMIT, None, (), "Time limit")

    def compile_timeout(self, model_ir, subproblem, options):
        return TimedOutSubproblem(subproblem)

    monkeypatch.setattr(lagrangian.HighsSubproblemFactory, "compile", compile_timeout)
    result = model.solve(max_iterations=2, time_limit=10)

    assert result.termination_reason == "time_limit"
    assert result.timed_out_subproblem == 0
    assert result.iterations == 0
    assert not result.has_certified_bounds
    assert result.variable_values == {}


def test_infeasible_hard_subproblem_is_reported() -> None:
    model = Model()
    x = model.add_var("x", lb=0, ub=1)
    model.minimize(x)
    model.add_constraint(x >= 2)
    with pytest.raises(SubproblemSolveError, match="did not reach optimality"):
        model.solve(max_iterations=1)


def test_zero_variable_model_can_complete_iterations() -> None:
    result = Model("empty").solve(max_iterations=1)
    assert result.has_certified_bounds
    assert result.best_primal_bound == 0
    assert result.best_dual_bound == 0
    assert result.relative_gap == 0
    assert result.variable_values == {}
    assert math.isfinite(result.elapsed_time)

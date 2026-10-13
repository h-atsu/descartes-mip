import pytest

import descartes_mip
from descartes_mip import Model, VariableType, quicksum


def test_public_api_solves_integer_model_with_native_highs() -> None:
    model = Model("production")
    plants = [
        model.add_var(f"plant_{index}", lb=0, ub=8, vartype=VariableType.INTEGER)
        for index in range(2)
    ]
    model.minimize(quicksum(cost * variable for cost, variable in zip((3, 5), plants)))
    demand = model.add_constraint(quicksum(plants) >= 10, "demand")
    model.relax(demand, penalty=20)

    result = model.solve(max_iterations=20, parallel=True, workers=2)

    assert result.has_certified_bounds
    assert result.best_primal_bound >= result.best_dual_bound
    assert result.value(plants[0]) == pytest.approx(result.variable_values["plant_0"])
    assert result.violation(demand) == pytest.approx(result.violations["demand"])


def test_removed_pulp_api_is_not_exported() -> None:
    for name in ("SoftConstraint", "SolverClass", "SUPPORTED_SOLVERS", "solve", "decompose"):
        assert not hasattr(descartes_mip, name)

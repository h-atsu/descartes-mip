import pytest

from descartes_mip import Model, ModelValidationError, VariableType
from descartes_mip.subproblems.highs import HighsSubproblemFactory


def test_compiled_highs_model_can_update_objective_without_rebuilding() -> None:
    model = Model()
    x = model.add_var("x", vartype=VariableType.BINARY)
    model.minimize(x)
    ir, _ = model._snapshot()
    decomposition = model.decompose()
    compiled = HighsSubproblemFactory().compile(
        ir, decomposition._subproblem_irs[0], {"output_flag": False}
    )

    first = compiled.solve([1.0], None)
    second = compiled.solve([-1.0], None)

    assert first.variable_values == (0.0,)
    assert second.variable_values == (1.0,)


def test_invalid_and_reserved_highs_options_are_rejected() -> None:
    model = Model()
    model.add_var(lb=0, ub=1)
    with pytest.raises(ModelValidationError, match="invalid HiGHS option"):
        model.solve(highs_options={"not_an_option": 1}, max_iterations=1)
    with pytest.raises(ModelValidationError, match="reserved"):
        model.solve(highs_options={"time_limit": 1}, max_iterations=1)

    with pytest.raises(ModelValidationError, match="invalid HiGHS option"):
        Model("empty").solve(highs_options={"not_an_option": 1}, max_iterations=1)

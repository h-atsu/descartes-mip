import math
from typing import Any, cast

import pytest

from descartes_mip import (
    Constraint,
    LinearExpr,
    Model,
    ModelValidationError,
    VariableType,
    quicksum,
)


def test_model_builds_sparse_linear_ir_with_stable_generated_names() -> None:
    model = Model("example")
    x = model.add_var(lb=0, ub=4)
    y = model.add_var(name="y", lb=-2, ub=3, vartype=VariableType.INTEGER)
    model.minimize(2 * x - y + 7)
    constraint = model.add_constraint(x + 3 * y >= 5)

    ir, relaxations = model._snapshot()

    assert (x.name, y.name, constraint.name) == ("var_0", "y", "constraint_0")
    assert ir.objective_constant == 7
    assert [(term.variable_index, term.coefficient) for term in ir.objective_terms] == [
        (0, 2),
        (1, -1),
    ]
    assert ir.constraints[0].constant == -5
    assert relaxations == ()


def test_quicksum_and_scalar_arithmetic_cancel_zero_terms() -> None:
    model = Model()
    x = model.add_var(lb=0, ub=1)
    y = model.add_var(lb=0, ub=1)
    expression = quicksum([x, 2 * y, -x, 3]) / 2
    model.minimize(expression)

    ir, _ = model._snapshot()

    assert ir.objective_constant == pytest.approx(1.5)
    assert [(term.variable_index, term.coefficient) for term in ir.objective_terms] == [(1, 1.0)]


def test_binary_defaults_to_zero_one_and_accepts_fixed_binary() -> None:
    model = Model()
    default = model.add_var(vartype=VariableType.BINARY)
    fixed = model.add_var(lb=1, ub=1, vartype=VariableType.BINARY)

    assert (default.low_bound, default.up_bound) == (0, 1)
    assert (fixed.low_bound, fixed.up_bound) == (1, 1)


@pytest.mark.parametrize("bounds", [(-1, 1), (0, 2), (1, 0)])
def test_invalid_binary_or_inconsistent_bounds_are_rejected(
    bounds: tuple[float, float],
) -> None:
    model = Model()
    with pytest.raises(ModelValidationError):
        model.add_var(lb=bounds[0], ub=bounds[1], vartype=VariableType.BINARY)


def test_explicit_and_generated_variable_names_never_collide() -> None:
    model = Model()
    model.add_var("var_0", lb=0, ub=1)
    model.add_var("var_2", lb=0, ub=1)

    generated = [model.add_var(lb=0, ub=1).name for _ in range(3)]

    assert generated == ["var_1", "var_3", "var_4"]
    with pytest.raises(ModelValidationError, match="duplicate"):
        model.add_var("var_0")


def test_explicit_and_generated_constraint_names_never_collide() -> None:
    model = Model()
    x = model.add_var(lb=0, ub=1)
    model.add_constraint(x <= 1, "constraint_0")
    model.add_constraint(x <= 1, "constraint_2")

    generated = [model.add_constraint(x <= 1).name for _ in range(3)]

    assert generated == ["constraint_1", "constraint_3", "constraint_4"]


def test_failed_variable_validation_does_not_consume_generated_name() -> None:
    model = Model()

    with pytest.raises(ModelValidationError, match="inconsistent bounds"):
        model.add_var(lb=1, ub=0)

    assert model.add_var().name == "var_0"


def test_cross_model_and_nonlinear_expressions_are_rejected() -> None:
    left = Model().add_var(lb=0, ub=1)
    right = Model().add_var(lb=0, ub=1)
    with pytest.raises(ModelValidationError, match="different models"):
        _ = left + right
    with pytest.raises(TypeError, match="real number"):
        _ = left * cast(Any, right)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_non_finite_model_numbers_are_rejected(value: float) -> None:
    model = Model()
    with pytest.raises(ModelValidationError, match="finite"):
        model.add_var(lb=value)
    x = model.add_var(lb=0, ub=1)
    with pytest.raises(ModelValidationError, match="finite"):
        _ = value * x


def test_constraint_must_be_registered_once_with_its_owner() -> None:
    model = Model()
    x = model.add_var(lb=0, ub=1)
    constraint = x <= 1
    assert isinstance(constraint, Constraint)
    model.add_constraint(constraint)
    with pytest.raises(ModelValidationError, match="already"):
        model.add_constraint(constraint)
    with pytest.raises(ModelValidationError, match="another model"):
        Model().add_constraint(x >= 0)


def test_constant_constraints_can_be_constructed_explicitly() -> None:
    model = Model()
    constraint = model.add_constraint(LinearExpr() <= 1)
    assert constraint.name == "constraint_0"


def test_relax_accepts_a_batch_and_rejects_duplicates() -> None:
    model = Model()
    x = model.add_var(lb=0, ub=1)
    first = model.add_constraint(x <= 1)
    second = model.add_constraint(x >= 0)
    model.relax([first, second], penalty=4)
    with pytest.raises(ModelValidationError, match="more than once"):
        model.relax(first, penalty=4)

    another = Model()
    variable = another.add_var(lb=0, ub=1)
    repeated = another.add_constraint(variable <= 1)
    with pytest.raises(ModelValidationError, match="more than once"):
        another.relax([repeated, repeated], penalty=4)


@pytest.mark.parametrize("penalty", [0, -1, math.inf, math.nan, True])
def test_relax_rejects_invalid_penalty(penalty: Any) -> None:
    model = Model()
    x = model.add_var(lb=0, ub=1)
    constraint = model.add_constraint(x <= 1)
    with pytest.raises(ModelValidationError, match="positive finite"):
        model.relax(constraint, penalty=penalty)

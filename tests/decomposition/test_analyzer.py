import pytest

from descartes_mip import LinearExpr, Model, ModelValidationError, VariableType


def build_two_component_model() -> tuple[Model, object]:
    model = Model("two_components")
    x1 = model.add_var("x1", lb=0, ub=10)
    x2 = model.add_var("x2", lb=0, ub=10, vartype=VariableType.INTEGER)
    y = model.add_var("y", lb=0, ub=8)
    z = model.add_var("z", lb=0, ub=2)
    model.minimize(2 * x1 + 3 * x2 + 4 * y + 5 * z + 7)
    model.add_constraint(x1 + x2 <= 10, "x_capacity")
    model.add_constraint(y <= 8, "y_capacity")
    coupling = model.add_constraint(x2 + 2 * y >= 6, "coupling")
    model.relax(coupling, penalty=20)
    return model, coupling


def test_decompose_finds_deterministic_components_and_terms() -> None:
    model, _ = build_two_component_model()
    decomposition = model.decompose()

    assert decomposition.problem_name == "two_components"
    assert decomposition.problem_sense == "minimize"
    assert decomposition.objective_constant == 7
    assert [part.variables for part in decomposition.subproblems] == [
        ("x1", "x2"),
        ("y",),
        ("z",),
    ]
    assert [part.hard_constraints for part in decomposition.subproblems] == [
        ("x_capacity",),
        ("y_capacity",),
        (),
    ]
    assert decomposition.relaxed_constraints[0].variables == ("x2", "y")
    assert decomposition.relaxed_constraints[0].sense == ">="


def test_decompose_does_not_require_finite_bounds() -> None:
    model = Model()
    x = model.add_var()
    model.minimize(x)
    assert model.decompose().variable_count == 1


def test_constant_hard_constraints_are_kept_or_rejected() -> None:
    feasible = Model()
    feasible.add_constraint(LinearExpr() <= 1, "always_true")
    assert feasible.decompose().constant_hard_constraints == ("always_true",)

    infeasible = Model()
    infeasible.add_constraint(LinearExpr() >= 1, "always_false")
    with pytest.raises(ModelValidationError, match="constant hard constraint"):
        infeasible.decompose()


def test_empty_model_has_no_subproblems() -> None:
    decomposition = Model("empty").decompose()
    assert decomposition.variable_count == 0
    assert decomposition.subproblems == ()

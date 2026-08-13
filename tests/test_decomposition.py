from typing import Any, cast

import pulp
import pytest

from descartes_mip import ModelValidationError, SoftConstraint, decompose


def build_two_component_problem() -> tuple[pulp.LpProblem, pulp.LpConstraint]:
    problem = pulp.LpProblem("two_components", pulp.LpMinimize)
    x1 = problem.add_variable("x1", lowBound=0)
    x2 = problem.add_variable("x2", lowBound=0, cat=pulp.LpInteger)
    y = problem.add_variable("y", lowBound=0)
    z = problem.add_variable("z", lowBound=0)

    problem += 2 * x1 + 3 * x2 + 4 * y + 5 * z + 7
    problem += x1 + x2 <= 10, "x_capacity"
    problem += y <= 8, "y_capacity"
    coupling = x2 + 2 * y >= 6
    problem += coupling, "coupling"
    return problem, coupling


def test_decompose_removes_soft_constraint_and_finds_components() -> None:
    problem, coupling = build_two_component_problem()

    decomposition = decompose(problem, relax=[SoftConstraint(coupling, penalty=20)])

    assert decomposition.problem_name == "two_components"
    assert decomposition.problem_sense == "minimize"
    assert decomposition.objective_constant == 7
    assert decomposition.variable_count == 4
    assert decomposition.subproblem_count == 3
    assert [subproblem.variables for subproblem in decomposition.subproblems] == [
        ("x1", "x2"),
        ("y",),
        ("z",),
    ]
    assert [subproblem.hard_constraints for subproblem in decomposition.subproblems] == [
        ("x_capacity",),
        ("y_capacity",),
        (),
    ]


def test_decompose_distributes_objective_and_relaxed_terms() -> None:
    problem, coupling = build_two_component_problem()

    decomposition = decompose(problem, relax=[SoftConstraint(coupling, penalty=20)])

    x_component, y_component, z_component = decomposition.subproblems
    assert [(term.variable, term.coefficient) for term in x_component.objective_terms] == [
        ("x1", 2),
        ("x2", 3),
    ]
    assert [(term.variable, term.coefficient) for term in y_component.objective_terms] == [("y", 4)]
    assert [(term.variable, term.coefficient) for term in z_component.objective_terms] == [("z", 5)]
    assert [
        (
            term.constraint,
            [(coefficient.variable, coefficient.coefficient) for coefficient in term.terms],
        )
        for term in x_component.relaxed_terms
    ] == [("coupling", [("x2", 1)])]
    assert [
        (
            term.constraint,
            [(coefficient.variable, coefficient.coefficient) for coefficient in term.terms],
        )
        for term in y_component.relaxed_terms
    ] == [("coupling", [("y", 2)])]
    assert z_component.relaxed_terms == ()
    assert decomposition.relaxed_constraints[0].name == "coupling"
    assert decomposition.relaxed_constraints[0].sense == ">="
    assert decomposition.relaxed_constraints[0].penalty == 20.0
    assert decomposition.relaxed_constraints[0].constant == -6
    assert decomposition.relaxed_constraints[0].variables == ("x2", "y")


def test_decompose_does_not_change_the_model_semantics_or_solution_state() -> None:
    problem, coupling = build_two_component_problem()
    before = (
        str(problem.objective),
        tuple(
            (id(constraint), constraint.name, str(constraint))
            for constraint in problem.constraints()
        ),
        problem.status,
        tuple(variable.varValue for variable in problem.variables()),
    )

    decompose(problem, relax=[SoftConstraint(coupling, penalty=20)])

    after = (
        str(problem.objective),
        tuple(
            (id(constraint), constraint.name, str(constraint))
            for constraint in problem.constraints()
        ),
        problem.status,
        tuple(variable.varValue for variable in problem.variables()),
    )
    assert after == before


@pytest.mark.parametrize("penalty", [0, -1, float("inf"), float("nan"), True])
def test_soft_constraint_rejects_invalid_penalty(penalty: Any) -> None:
    _, coupling = build_two_component_problem()

    with pytest.raises(ModelValidationError, match="positive finite"):
        SoftConstraint(coupling, penalty=penalty)


def test_decompose_rejects_constraint_from_another_problem() -> None:
    problem, _ = build_two_component_problem()
    other = pulp.LpProblem("other")
    x = other.add_variable("other_x")
    foreign_constraint = x <= 1
    other += foreign_constraint, "foreign"

    with pytest.raises(ModelValidationError, match="belong"):
        decompose(problem, relax=[SoftConstraint(foreign_constraint, penalty=1)])


def test_decompose_rejects_duplicate_soft_constraint() -> None:
    problem, coupling = build_two_component_problem()

    with pytest.raises(ModelValidationError, match="more than once"):
        decompose(
            problem,
            relax=[
                SoftConstraint(coupling, penalty=1),
                SoftConstraint(coupling, penalty=2),
            ],
        )


def test_soft_constraint_rejects_non_constraint() -> None:
    with pytest.raises(ModelValidationError, match="LpConstraint"):
        SoftConstraint(cast(Any, "not a constraint"), penalty=1)


def test_decompose_supports_no_objective_and_no_variables() -> None:
    problem = pulp.LpProblem("empty", pulp.LpMaximize)

    decomposition = decompose(problem, relax=[])

    assert decomposition.problem_sense == "maximize"
    assert decomposition.objective_constant == 0
    assert decomposition.variable_count == 0
    assert decomposition.subproblems == ()


def test_decompose_keeps_a_feasible_constant_hard_constraint() -> None:
    problem = pulp.LpProblem("constant")
    problem += pulp.LpConstraint(e=0, sense=pulp.LpConstraintLE, rhs=1), "always_true"

    decomposition = decompose(problem, relax=[])

    assert decomposition.constant_hard_constraints == ("always_true",)


def test_decompose_rejects_an_infeasible_constant_hard_constraint() -> None:
    problem = pulp.LpProblem("constant")
    problem += pulp.LpConstraint(e=0, sense=pulp.LpConstraintGE, rhs=1), "always_false"

    with pytest.raises(ModelValidationError, match="constant hard constraint"):
        decompose(problem, relax=[])

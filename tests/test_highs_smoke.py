import pulp

from descartes_mip import highs_factory


def test_highs_solves_a_minimal_milp_to_optimality() -> None:
    problem = pulp.LpProblem("smoke", pulp.LpMinimize)
    x = problem.add_variable("x", lowBound=0, cat=pulp.LpInteger)
    problem += x
    problem += x >= 3

    status = problem.solve(highs_factory(None))

    assert status == pulp.LpStatusOptimal
    assert pulp.value(x) == 3
    assert pulp.value(problem.objective) == 3

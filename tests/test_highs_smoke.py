import pulp


def test_highs_solves_a_minimal_milp_to_optimality() -> None:
    problem = pulp.LpProblem("smoke", pulp.LpMinimize)
    x = problem.add_variable("x", lowBound=0, cat=pulp.LpInteger)
    problem += x
    problem += x >= 3

    status = problem.solve(pulp.HiGHS(msg=False))

    assert status == pulp.LpStatusOptimal
    assert pulp.value(x) == 3
    assert pulp.value(problem.objective) == 3

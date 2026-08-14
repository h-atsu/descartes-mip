"""Complete example: decompose and solve a soft production target."""

import pulp

from descartes_mip import SoftConstraint, decompose, solve


def main() -> None:
    problem = pulp.LpProblem("production", pulp.LpMinimize)
    plant_a = problem.add_variable("plant_a", lowBound=0, upBound=8, cat=pulp.LpInteger)
    plant_b = problem.add_variable("plant_b", lowBound=0, upBound=8, cat=pulp.LpInteger)

    problem += 3 * plant_a + 5 * plant_b
    demand = plant_a + plant_b >= 10
    problem += demand, "demand"
    relaxed = [SoftConstraint(demand, penalty=20)]

    structure = decompose(problem, relax=relaxed)
    print("subproblems:", [part.variables for part in structure.subproblems])

    result = solve(
        problem,
        relax=relaxed,
        max_iterations=50,
        time_limit=30,
        parallel=True,
        workers=2,
    )
    print("termination:", result.termination_reason)
    print("certified bounds:", result.best_dual_bound, result.best_primal_bound)
    print("solution:", result.variable_values)
    print("soft-constraint slacks:", result.slacks)


if __name__ == "__main__":
    main()

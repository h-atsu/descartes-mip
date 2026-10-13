"""Build, decompose, and solve a soft production target."""

from descartes_mip import Model, VariableType


def main() -> None:
    model = Model("production")
    plant_a = model.add_var("plant_a", lb=0, ub=8, vartype=VariableType.INTEGER)
    plant_b = model.add_var("plant_b", lb=0, ub=8, vartype=VariableType.INTEGER)
    model.minimize(3 * plant_a + 5 * plant_b)
    demand = model.add_constraint(plant_a + plant_b >= 10, "demand")
    model.relax(demand, penalty=20)

    structure = model.decompose()
    print("subproblems:", [part.variables for part in structure.subproblems])

    result = model.solve(
        highs_options={"mip_rel_gap": 0.0},
        max_iterations=50,
        time_limit=30,
        parallel=True,
        workers=2,
    )
    print("termination:", result.termination_reason)
    print("certified bounds:", result.best_dual_bound, result.best_primal_bound)
    print("solution:", result.variable_values)
    print("demand violation:", result.violation(demand))
    print("plant A:", result.value(plant_a))
    print("plant B:", result.value(plant_b))


if __name__ == "__main__":
    main()

# %%
import pulp

from descartes_mip.decomposition import SoftConstraint, decompose

# %%
problem = pulp.LpProblem("two_components", pulp.LpMinimize)
x1 = problem.add_variable("x1", lowBound=0)
x2 = problem.add_variable("x2", lowBound=0, cat=pulp.LpInteger)
y = problem.add_variable("y", lowBound=0)
z = problem.add_variable("z", lowBound=0)

problem += 2 * x1 + 3 * x2 + 4 * y + 5 * z + 7
x_capacity_constraint = x1 + x2 <= 10
problem += x_capacity_constraint, "x_capacity"
problem += y <= 8, "y_capacity"
coupling = x2 + 2 * y >= 6
problem += coupling, "coupling"


# %%

decomposition = decompose(
    problem,
    relax=[
        SoftConstraint(coupling, penalty=20),
        SoftConstraint(x_capacity_constraint, penalty=20),
    ],
)


# %%
for subproblem in decomposition.subproblems:
    print(subproblem)
# %%
status = problem.solve()
# %%
assert status == pulp.LpStatusOptimal


# %%

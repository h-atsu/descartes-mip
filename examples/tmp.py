# %%
import pulp

# %%
problem = pulp.LpProblem("sample", pulp.LpMinimize)
x = problem.add_variable("x", lowBound=0, cat=pulp.LpInteger)
problem += x
problem += x >= 3

status = problem.solve()


# %%
assert status == pulp.LpStatusOptimal
assert pulp.value(x) == 3
assert pulp.value(problem.objective) == 3

# %%

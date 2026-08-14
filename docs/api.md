# API reference

## `SoftConstraint`

```python
SoftConstraint(constraint: pulp.LpConstraint, penalty: float)
```

Marks a constraint for Lagrangian relaxation with a positive finite L1 violation penalty.
The constraint must already belong to the supplied `LpProblem`.

## `decompose`

```python
decompose(
    problem: pulp.LpProblem,
    *,
    relax: list[SoftConstraint] | tuple[SoftConstraint, ...],
) -> Decomposition
```

Removes the selected constraints from the variable–constraint graph and returns its connected
components. It does not solve or modify the semantic state of the source model. The result exposes
the variables, hard constraints, objective terms, and relaxed terms assigned to each subproblem.

## `solve`

```python
solve(
    problem: pulp.LpProblem,
    *,
    relax: Sequence[SoftConstraint],
    solver: SolverClass = pulp.HiGHS,
    solver_options: Mapping[str, object] | None = None,
    max_iterations: int = 100,
    time_limit: float | None = None,
    parallel: bool = False,
    workers: int | None = None,
    polyak_factor: float = 1.0,
    initial_multipliers: Mapping[str, float] | None = None,
    callback: IterationCallback | None = None,
) -> LagrangianResult
```

Solves every generated MILP to proven optimality and updates the boxed Lagrangian multipliers with
projected Polyak subgradient steps.

- Every decision variable must have finite explicit lower and upper bounds.
- `time_limit` is a global monotonic wall-clock budget. An incomplete iteration is discarded.
- `parallel=True` solves independent subproblems in a thread pool. `workers` controls its maximum
  concurrency and is only valid in parallel mode.
- A fresh solver instance is constructed for every subproblem solve. Instance creation is
  serialized; the resulting solvers execute concurrently.
- Supported classes are `pulp.HiGHS` and `pulp.COIN_CMD`. Passing any other class or an existing
  solver instance raises `ModelValidationError` before decomposition or solving starts.
- `solver_options` are forwarded as keyword arguments to every new instance. `msg=False` is the
  default and can be overridden. `timeLimit` is reserved because the library derives it from the
  global `time_limit` budget.

For minimization, `best_primal_bound` is a soft-feasible upper bound and `best_dual_bound` is a
certified lower bound. For maximization they are respectively a lower and upper bound.

## Result records

`LagrangianResult` contains the termination reason, best bounds, gap, best variable values, implicit
L1 slacks, final multipliers, decomposition, elapsed time, and complete iteration history.

- `has_certified_bounds` is false when no iteration completed.
- `soft_feasible` indicates that a complete soft-feasible solution is available.
- `hard_feasible` indicates that the best solution also has zero relaxed-constraint violation within
  tolerance.
- `timed_out_subproblem` identifies the subproblem that caused an incomplete timed-out iteration.

Each `IterationRecord` stores the current and incumbent bounds, step size, multipliers, residuals,
violations, variable values, subproblem records, and cumulative elapsed time. Each
`SubproblemRecord` stores its ID, PuLP status, Lagrangian objective contribution, and elapsed time.

## Solvers

```python
solve(
    problem,
    relax=relaxed,
    solver=pulp.HiGHS,
    solver_options={"msg": False, "gapRel": 1e-9},
)
```

`SUPPORTED_SOLVERS` exposes the exact accepted class tuple. The class itself—not an instantiated
solver—is supplied so sequential and parallel modes can both create one independent backend per
subproblem. CBC can be selected with `solver=pulp.COIN_CMD` when its executable is available; a
nonstandard executable location can be supplied as `solver_options={"path": "..."}`.

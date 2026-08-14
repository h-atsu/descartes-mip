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
    solver_factory: SolverFactory = highs_factory,
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
- The factory is called once per subproblem solve and must return a new solver instance. Factory
  creation is serialized; the returned solvers execute concurrently.
- The default and CI reference backend is HiGHS. Other PuLP solvers are accepted when they report
  statuses through the standard PuLP interface.

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

## Solver factories

```python
highs_factory(remaining_time: float | None) -> pulp.LpSolver
```

`highs_factory` creates a quiet independent HiGHS instance and applies the supplied remaining-time
limit. A custom factory may construct any compatible PuLP solver in the same way.

"""Find independent components after removing selected coupling constraints."""

from descartes_mip.exceptions import ModelValidationError
from descartes_mip.modeling.ir import (
    ConstraintIR,
    ConstraintSense,
    ModelIR,
    ObjectiveSense,
    RelaxationIR,
    SubproblemIR,
)

from .records import (
    Decomposition,
    LinearTerm,
    RelaxedConstraint,
    RelaxedTerm,
    Subproblem,
)


def analyze(model: ModelIR, relaxations: tuple[RelaxationIR, ...]) -> Decomposition:
    """Return the deterministic variable components induced by the relaxed constraints."""

    penalty_by_constraint = {
        relaxation.constraint_index: relaxation.penalty for relaxation in relaxations
    }
    relaxed_indexes = frozenset(penalty_by_constraint)
    components = _connected_components(model, relaxed_indexes)
    variable_to_component = {
        variable_index: component_id
        for component_id, component in enumerate(components)
        for variable_index in component
    }
    hard_by_component: list[list[int]] = [[] for _ in components]
    constant_hard_constraints: list[str] = []
    for constraint in model.constraints:
        if constraint.index in relaxed_indexes:
            continue
        if not constraint.terms:
            if not _constant_constraint_is_feasible(constraint):
                raise ModelValidationError(
                    f"constant hard constraint {constraint.name!r} is infeasible"
                )
            constant_hard_constraints.append(constraint.name)
            continue
        component_id = variable_to_component[constraint.terms[0].variable_index]
        hard_by_component[component_id].append(constraint.index)

    subproblem_irs = tuple(
        SubproblemIR(component_id, variable_indexes, tuple(hard_by_component[component_id]))
        for component_id, variable_indexes in enumerate(components)
    )
    public_subproblems = tuple(
        _public_subproblem(item, model, relaxed_indexes) for item in subproblem_irs
    )
    relaxed_constraints = tuple(
        RelaxedConstraint(
            name=model.constraints[index].name,
            sense=model.constraints[index].sense.value,
            penalty=penalty_by_constraint[index],
            constant=model.constraints[index].constant,
            variables=tuple(
                model.variables[term.variable_index].name for term in model.constraints[index].terms
            ),
        )
        for index in sorted(relaxed_indexes)
    )
    return Decomposition(
        problem_name=model.name,
        problem_sense=("minimize" if model.sense is ObjectiveSense.MINIMIZE else "maximize"),
        objective_constant=model.objective_constant,
        subproblems=public_subproblems,
        relaxed_constraints=relaxed_constraints,
        constant_hard_constraints=tuple(constant_hard_constraints),
        _model=model,
        _subproblem_irs=subproblem_irs,
        _relaxation_penalties=penalty_by_constraint,
    )


def _connected_components(model: ModelIR, relaxed: frozenset[int]) -> tuple[tuple[int, ...], ...]:
    parent = list(range(len(model.variables)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for constraint in model.constraints:
        if constraint.index in relaxed or len(constraint.terms) < 2:
            continue
        first = constraint.terms[0].variable_index
        for term in constraint.terms[1:]:
            union(first, term.variable_index)

    grouped: dict[int, list[int]] = {}
    for variable_index in range(len(model.variables)):
        grouped.setdefault(find(variable_index), []).append(variable_index)
    groups = list(grouped.values())
    groups.sort(key=lambda group: tuple(model.variables[index].name for index in group))
    return tuple(tuple(group) for group in groups)


def _public_subproblem(
    subproblem: SubproblemIR,
    model: ModelIR,
    relaxed_indexes: frozenset[int],
) -> Subproblem:
    owned = frozenset(subproblem.variable_indices)
    return Subproblem(
        id=subproblem.id,
        variables=tuple(model.variables[index].name for index in subproblem.variable_indices),
        hard_constraints=tuple(
            model.constraints[index].name for index in subproblem.hard_constraint_indices
        ),
        objective_terms=tuple(
            LinearTerm(model.variables[term.variable_index].name, term.coefficient)
            for term in model.objective_terms
            if term.variable_index in owned
        ),
        relaxed_terms=tuple(
            RelaxedTerm(
                constraint=model.constraints[index].name,
                terms=tuple(
                    LinearTerm(model.variables[term.variable_index].name, term.coefficient)
                    for term in model.constraints[index].terms
                    if term.variable_index in owned
                ),
            )
            for index in sorted(relaxed_indexes)
            if any(term.variable_index in owned for term in model.constraints[index].terms)
        ),
    )


def _constant_constraint_is_feasible(constraint: ConstraintIR) -> bool:
    if constraint.sense is ConstraintSense.LE:
        return constraint.constant <= 0
    if constraint.sense is ConstraintSense.GE:
        return constraint.constant >= 0
    return constraint.constant == 0

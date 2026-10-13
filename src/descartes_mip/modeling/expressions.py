"""Linear expressions and modeling handles."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

from descartes_mip.exceptions import ModelValidationError
from descartes_mip.modeling.ir import ConstraintSense, VariableType

if TYPE_CHECKING:
    from descartes_mip.modeling.model import Model

type Number = int | float
type LinearOperand = Variable | LinearExpr | Number


def _finite_number(value: object, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{owner} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ModelValidationError(f"{owner} must be finite")
    return result


class Variable:
    """A decision-variable handle owned by one :class:`Model`."""

    __slots__ = ("_index", "_low_bound", "_name", "_owner", "_up_bound", "_vartype")
    __hash__ = None

    def __init__(
        self,
        owner: Model,
        index: int,
        name: str,
        low_bound: float | None,
        up_bound: float | None,
        vartype: VariableType,
    ) -> None:
        self._owner = owner
        self._index = index
        self._name = name
        self._low_bound = low_bound
        self._up_bound = up_bound
        self._vartype = vartype

    @property
    def name(self) -> str:
        return self._name

    @property
    def low_bound(self) -> float | None:
        return self._low_bound

    @property
    def up_bound(self) -> float | None:
        return self._up_bound

    @property
    def vartype(self) -> VariableType:
        return self._vartype

    def _expr(self) -> LinearExpr:
        return LinearExpr(self._owner, {self._index: 1.0})

    def __add__(self, other: LinearOperand) -> LinearExpr:
        return self._expr() + other

    def __radd__(self, other: LinearOperand) -> LinearExpr:
        return self._expr() + other

    def __sub__(self, other: LinearOperand) -> LinearExpr:
        return self._expr() - other

    def __rsub__(self, other: LinearOperand) -> LinearExpr:
        return _as_expression(other) - self

    def __mul__(self, scalar: Number) -> LinearExpr:
        return self._expr() * scalar

    def __rmul__(self, scalar: Number) -> LinearExpr:
        return self * scalar

    def __truediv__(self, scalar: Number) -> LinearExpr:
        return self._expr() / scalar

    def __neg__(self) -> LinearExpr:
        return -self._expr()

    def __le__(self, other: LinearOperand) -> Constraint:
        return self._expr() <= other

    def __ge__(self, other: LinearOperand) -> Constraint:
        return self._expr() >= other

    def __eq__(self, other: object) -> Constraint:  # ty: ignore[invalid-method-override]
        return self._expr() == other

    def __repr__(self) -> str:
        return self._name


class LinearExpr:
    """A sparse affine expression used while constructing a model."""

    __slots__ = ("_constant", "_owner", "_terms")

    def __init__(
        self,
        owner: Model | None = None,
        terms: dict[int, float] | None = None,
        constant: float = 0.0,
    ) -> None:
        self._owner = owner
        self._terms = {} if terms is None else terms
        self._constant = constant

    def _combine(self, other: LinearOperand, factor: float) -> LinearExpr:
        right = _as_expression(other)
        if self._owner is not None and right._owner is not None and self._owner is not right._owner:
            raise ModelValidationError("cannot combine expressions from different models")
        owner = self._owner if self._owner is not None else right._owner
        terms = dict(self._terms)
        for index, coefficient in right._terms.items():
            updated = terms.get(index, 0.0) + factor * coefficient
            if updated == 0.0:
                terms.pop(index, None)
            else:
                terms[index] = updated
        return LinearExpr(owner, terms, self._constant + factor * right._constant)

    def __add__(self, other: LinearOperand) -> LinearExpr:
        return self._combine(other, 1.0)

    def __radd__(self, other: LinearOperand) -> LinearExpr:
        return self + other

    def __sub__(self, other: LinearOperand) -> LinearExpr:
        return self._combine(other, -1.0)

    def __rsub__(self, other: LinearOperand) -> LinearExpr:
        return _as_expression(other)._combine(self, -1.0)

    def __mul__(self, scalar: Number) -> LinearExpr:
        value = _finite_number(scalar, "scalar")
        return LinearExpr(
            self._owner,
            {index: value * coefficient for index, coefficient in self._terms.items()},
            value * self._constant,
        )

    def __rmul__(self, scalar: Number) -> LinearExpr:
        return self * scalar

    def __truediv__(self, scalar: Number) -> LinearExpr:
        value = _finite_number(scalar, "divisor")
        if value == 0.0:
            raise ZeroDivisionError("cannot divide a linear expression by zero")
        return self * (1.0 / value)

    def __neg__(self) -> LinearExpr:
        return self * -1.0

    def _constraint(self, other: object, sense: ConstraintSense) -> Constraint:
        try:
            expression = self - _as_expression(other)
        except TypeError:
            return NotImplemented  # type: ignore[return-value]
        return Constraint(expression, sense)

    def __le__(self, other: LinearOperand) -> Constraint:
        return self._constraint(other, ConstraintSense.LE)

    def __ge__(self, other: LinearOperand) -> Constraint:
        return self._constraint(other, ConstraintSense.GE)

    def __eq__(self, other: object) -> Constraint:  # ty: ignore[invalid-method-override]
        return self._constraint(other, ConstraintSense.EQ)

    def __bool__(self) -> bool:
        raise TypeError("a linear expression has no truth value")


class Constraint:
    """A linear constraint, registered by :meth:`Model.add_constraint`."""

    __slots__ = ("_expression", "_index", "_name", "_owner", "_sense")

    def __init__(self, expression: LinearExpr, sense: ConstraintSense) -> None:
        self._expression = expression
        self._sense = sense
        self._owner = expression._owner
        self._index: int | None = None
        self._name: str | None = None

    __hash__ = object.__hash__

    @property
    def name(self) -> str:
        if self._name is None:
            raise ModelValidationError("constraint has not been added to a model")
        return self._name

    @property
    def sense(self) -> ConstraintSense:
        return self._sense

    def __bool__(self) -> bool:
        raise TypeError("a constraint has no truth value; chained comparisons are unsupported")

    def __repr__(self) -> str:
        return self._name or "<unregistered constraint>"


def _as_expression(value: object) -> LinearExpr:
    if isinstance(value, LinearExpr):
        return value
    if isinstance(value, Variable):
        return value._expr()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("linear expressions support only variables, expressions, and real scalars")
    return LinearExpr(constant=_finite_number(value, "constant"))


def quicksum(values: Iterable[LinearOperand]) -> LinearExpr:
    """Return the linear sum of an iterable without creating a solver model."""

    owner: Model | None = None
    terms: dict[int, float] = {}
    constant = 0.0
    for value in values:
        expression = _as_expression(value)
        if owner is not None and expression._owner is not None and owner is not expression._owner:
            raise ModelValidationError("cannot combine expressions from different models")
        if owner is None:
            owner = expression._owner
        for index, coefficient in expression._terms.items():
            updated = terms.get(index, 0.0) + coefficient
            if updated == 0.0:
                terms.pop(index, None)
            else:
                terms[index] = updated
        constant += expression._constant
    return LinearExpr(owner, terms, constant)


def as_expression(value: LinearOperand) -> LinearExpr:
    """Validate and convert a public linear operand."""

    return _as_expression(value)

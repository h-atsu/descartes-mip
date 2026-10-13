"""Public exception hierarchy for descartes-mip."""


class DescartesMipError(Exception):
    """Base class for errors raised by descartes-mip."""


class ModelValidationError(DescartesMipError, ValueError):
    """Raised when a model or solver option cannot be processed safely."""


class SubproblemSolveError(DescartesMipError, RuntimeError):
    """Raised when a generated subproblem cannot be solved as required."""

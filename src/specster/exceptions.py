"""
Specster specific exceptions.
"""


class SpecsterError(Exception):
    """Specster exception parent."""


class UnhandledParFileLine(SpecsterError, ValueError):
    """Raised when a non-comment line isn't parsable."""


class SpecFEMError(SpecsterError, ValueError):
    """Raised when calling specfem gives an error."""

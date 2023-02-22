"""
Specster specific exceptions.
"""


class SpecsterError(Exception):
    """Specster exception parent."""


class MissingSpecFEMError(SpecsterError, FileNotFoundError):
    """Raised when no binaries are found."""


class UnhandledParFileLine(SpecsterError, ValueError):
    """Raised when a non-comment line isn't parsable."""

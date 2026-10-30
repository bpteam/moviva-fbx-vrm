from __future__ import annotations


class ConversionError(Exception):
    """A fatal conversion problem. ``code`` is stable and appears in the report."""

    def __init__(self, code: str, message: str, details: list[str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or []

    def __str__(self) -> str:
        return "\n".join([f"[{self.code}] {self.message}", *(f"  - {d}" for d in self.details)])

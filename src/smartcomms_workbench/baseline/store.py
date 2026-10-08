"""Baseline storage contract."""

from typing import Protocol


class BaselineStore(Protocol):
    def resolve(self, name: str) -> bytes | None:
        """Return the approved PDF, or None when no baseline exists."""
        ...

    def approve(self, name: str, pdf: bytes) -> None:
        """Persist an approved PDF; invalid PDF input raises PdfInputError."""
        ...

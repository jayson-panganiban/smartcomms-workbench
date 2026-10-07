"""Immutable execution snapshots and pure state evolution."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Self


@dataclass(frozen=True)
class ExecutionContext:
    """Freeze state and mapping entries, but share artifact values and PDF buffers."""

    run_id: str
    environment: str
    template_id: str
    payload_xml: str | None = None
    rendered_pdf: bytes | None = None
    baseline_pdf: bytes | None = None
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    def evolve(self, **changes: Any) -> Self:
        """Return a new snapshot with the requested fields replaced."""
        return replace(self, **changes)

    def get_artifact(self, name: str) -> Any:
        """Retrieve a named artifact, raising KeyError if it is missing."""
        return self.artifacts[name]

    def with_artifact(self, name: str, value: Any) -> Self:
        """Return a new snapshot with an artifact added or replaced."""
        return self.evolve(artifacts={**self.artifacts, name: value})

"""Explicit domain outcomes and the pipeline step interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from smartcomms_workbench.pipeline.context import ExecutionContext


@dataclass(frozen=True)
class StepSuccess:
    """A successful step carrying the next execution snapshot."""

    step_name: str
    context: ExecutionContext
    message: str = ""


@dataclass(frozen=True)
class StepFailure:
    """An expected domain failure, not an unexpected programming exception."""

    step_name: str
    error: str
    details: dict[str, Any] = field(default_factory=dict)


type StepResult = StepSuccess | StepFailure


class Step(ABC):
    """Transform an execution snapshot into an explicit domain result."""

    @abstractmethod
    def execute(self, context: ExecutionContext) -> StepResult:
        """Return success or expected failure; let programming exceptions propagate."""
        ...

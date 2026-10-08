"""Explicit step factories and optional decorator-based consumer registration."""

from collections.abc import Callable, Mapping
from typing import TypeVar

from smartcomms_workbench.pipeline.step import Step

type StepFactory = Callable[[Mapping[str, object]], Step]
StepType = TypeVar("StepType", bound=type[Step])


class StepRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, StepFactory] = {}

    def register(self, name: str, factory: StepFactory) -> None:
        if not name or name in self._factories:
            raise ValueError(f"Empty or duplicate step name: {name!r}")
        self._factories[name] = factory

    def create(self, name: str, options: Mapping[str, object]) -> Step:
        try:
            factory = self._factories[name]
        except KeyError:
            raise ValueError(f"Unknown pipeline step: {name}") from None
        step = factory(options)
        if not isinstance(step, Step):
            raise TypeError(f"Factory for {name} did not return a Step")
        return step

    def copy(self) -> "StepRegistry":
        registry = StepRegistry()
        registry._factories.update(self._factories)
        return registry


default_registry = StepRegistry()


def register_step(name: str, *, registry: StepRegistry = default_registry) -> Callable[[StepType], StepType]:
    def decorate(step_type: StepType) -> StepType:
        registry.register(name, lambda options: step_type(**options))
        return step_type

    return decorate

"""Validated recipe interpretation and fail-fast snapshot execution."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from smartcomms_workbench.pipeline.context import ExecutionContext
from smartcomms_workbench.pipeline.registry import StepRegistry
from smartcomms_workbench.pipeline.step import Step, StepFailure, StepResult, StepSuccess


class StepSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1)
    options: dict[str, object] = Field(default_factory=dict)


class RecipeContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    environment: str
    template_id: str
    run_id: str | None = None


class PipelineRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    version: Literal[1] = 1
    context: RecipeContext
    steps: list[StepSpec] = Field(min_length=1)
    payload: str | None = None
    candidate: str | None = None
    baseline: str | None = None
    baseline_store: str | None = None

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int) or value != 1:
            raise ValueError("Recipe version must be the integer 1")
        return value


@dataclass(frozen=True)
class PipelineResult:
    """Always retain the last successful snapshot, including on expected failure."""

    context: ExecutionContext
    results: tuple[StepResult, ...]

    @property
    def success(self) -> bool:
        return not self.results or isinstance(self.results[-1], StepSuccess)

    @property
    def failure(self) -> StepFailure | None:
        last = self.results[-1] if self.results else None
        return last if isinstance(last, StepFailure) else None


class Pipeline:
    def __init__(self, steps: Sequence[Step]) -> None:
        self.steps = tuple(steps)

    @classmethod
    def from_recipe(cls, recipe: PipelineRecipe, registry: StepRegistry) -> "Pipeline":
        # Construct every step before running any effects.
        return cls([registry.create(spec.name, spec.options) for spec in recipe.steps])

    @staticmethod
    def parse_recipe(text: str) -> PipelineRecipe:
        return PipelineRecipe.model_validate(yaml.safe_load(text))

    @classmethod
    def load(cls, path: Path, registry: StepRegistry) -> "Pipeline":
        return cls.from_recipe(cls.parse_recipe(path.read_text(encoding="utf-8")), registry)

    def run(self, context: ExecutionContext) -> PipelineResult:
        results: list[StepResult] = []
        for step in self.steps:
            result = step.execute(context)
            match result:
                case StepSuccess(context=next_context):
                    context = next_context
                case StepFailure():
                    results.append(result)
                    break
                case _:
                    raise TypeError("Step.execute must return StepSuccess or StepFailure")
            results.append(result)
        return PipelineResult(context, tuple(results))

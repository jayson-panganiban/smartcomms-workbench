from dataclasses import FrozenInstanceError
from operator import setitem
from typing import Any

import pytest

from smartcomms_workbench.pipeline.context import ExecutionContext
from smartcomms_workbench.pipeline.step import Step, StepFailure, StepResult, StepSuccess


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext("run-1", "test", "template-1")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("run_id", "run-2"),
        ("environment", "production"),
        ("template_id", "template-2"),
        ("payload_xml", "<payload/>"),
        ("rendered_pdf", b"rendered"),
        ("baseline_pdf", b"baseline"),
        ("artifacts", {}),
        ("metrics", {}),
    ],
)
def test_context_fields_are_frozen(context: ExecutionContext, field_name: str, value: Any) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(context, field_name, value)


def test_context_mapping_entries_are_immutable(context: ExecutionContext) -> None:
    with pytest.raises(TypeError):
        setitem(context.artifacts, "report", "report.html")  # ty: ignore[no-matching-overload] - test runtime immutability
    with pytest.raises(TypeError):
        setitem(context.metrics, "duration", 1.0)  # ty: ignore[no-matching-overload] - test runtime immutability


def test_context_takes_owned_shallow_mapping_snapshots() -> None:
    artifact = bytearray(b"large binary artifact")
    artifacts = {"report": artifact}
    metrics = {"duration": 1.0}
    context = ExecutionContext("run-1", "test", "template-1", artifacts=artifacts, metrics=metrics)

    artifacts.clear()
    metrics["duration"] = 2.0

    assert context.get_artifact("report") is artifact
    assert context.metrics == {"duration": 1.0}


def test_context_defaults_are_empty_and_optional(context: ExecutionContext) -> None:
    assert context.payload_xml is None
    assert context.rendered_pdf is None
    assert context.baseline_pdf is None
    assert context.artifacts == {}
    assert context.metrics == {}
    assert context.artifacts is not ExecutionContext("run-2", "test", "template-1").artifacts


def test_evolve_preserves_original_and_unchanged_fields(context: ExecutionContext) -> None:
    evolved = context.evolve(payload_xml="<payload/>", metrics={"duration": 0.5})

    assert evolved is not context
    assert evolved.run_id == context.run_id
    assert evolved.environment == context.environment
    assert evolved.template_id == context.template_id
    assert evolved.payload_xml == "<payload/>"
    assert evolved.metrics == {"duration": 0.5}
    assert context.payload_xml is None
    assert context.metrics == {}
    with pytest.raises(TypeError):
        setitem(evolved.metrics, "duration", 1.0)  # ty: ignore[no-matching-overload] - test runtime immutability


def test_evolve_shares_pdf_buffers_and_artifact_values(context: ExecutionContext) -> None:
    rendered = b"%PDF-rendered"
    baseline = b"%PDF-baseline"
    artifact = bytearray(b"binary artifact")
    original = context.evolve(rendered_pdf=rendered, baseline_pdf=baseline, artifacts={"binary": artifact})

    evolved = original.evolve(payload_xml="<payload/>")

    assert evolved.rendered_pdf is rendered
    assert evolved.baseline_pdf is baseline
    assert evolved.get_artifact("binary") is artifact


def test_evolve_without_changes_returns_new_equal_snapshot(context: ExecutionContext) -> None:
    evolved = context.evolve()

    assert evolved == context
    assert evolved is not context


def test_evolve_unknown_field_raises(context: ExecutionContext) -> None:
    with pytest.raises(TypeError):
        context.evolve(unknown_field="invalid")


def test_artifact_helpers_add_and_replace_without_mutation(context: ExecutionContext) -> None:
    first = context.with_artifact("report", "first.html")
    second = first.with_artifact("pdf", b"%PDF")
    third = second.with_artifact("report", "updated.html")

    assert context.artifacts == {}
    assert first.artifacts == {"report": "first.html"}
    assert second.artifacts == {"report": "first.html", "pdf": b"%PDF"}
    assert third.get_artifact("report") == "updated.html"
    assert third.get_artifact("pdf") == b"%PDF"
    with pytest.raises(KeyError, match="missing"):
        context.get_artifact("missing")


def test_step_requires_execute_implementation() -> None:
    with pytest.raises(TypeError, match="abstract"):
        Step()  # ty: ignore[call-non-callable] - test runtime ABC enforcement


def test_successful_step_returns_evolved_context(context: ExecutionContext) -> None:
    class SetPayloadStep(Step):
        def execute(self, context: ExecutionContext) -> StepResult:
            return StepSuccess("set_payload", context.evolve(payload_xml="<payload/>"), "Payload prepared")

    result = SetPayloadStep().execute(context)

    assert isinstance(result, StepSuccess)
    assert result.step_name == "set_payload"
    assert result.message == "Payload prepared"
    assert result.context.payload_xml == "<payload/>"
    assert context.payload_xml is None


def test_expected_failure_is_explicit(context: ExecutionContext) -> None:
    class ValidatePayloadStep(Step):
        def execute(self, context: ExecutionContext) -> StepResult:
            if context.payload_xml is None:
                return StepFailure("validate_payload", "Payload is required", {"field": "payload_xml"})
            return StepSuccess("validate_payload", context)

    result = ValidatePayloadStep().execute(context)

    match result:
        case StepFailure(step_name=step_name, error=error, details=details):
            assert step_name == "validate_payload"
            assert error == "Payload is required"
            assert details == {"field": "payload_xml"}
        case StepSuccess():
            pytest.fail("Missing payload must return an expected failure")


@pytest.mark.parametrize("exception_type", [TypeError, KeyError])
def test_programming_exceptions_propagate(context: ExecutionContext, exception_type: type[Exception]) -> None:
    class BrokenStep(Step):
        def execute(self, context: ExecutionContext) -> StepResult:
            raise exception_type(context.run_id)

    with pytest.raises(exception_type, match="run-1"):
        BrokenStep().execute(context)


@pytest.mark.parametrize("field_name", ["context", "error"])
def test_step_results_are_frozen(context: ExecutionContext, field_name: str) -> None:
    success = StepSuccess("success", context)
    failure = StepFailure("failure", "Expected domain failure")

    assert success.message == ""
    assert success.context is context
    assert failure.details == {}
    result = success if field_name == "context" else failure
    value = context.evolve(run_id="run-2") if field_name == "context" else "Changed error"
    with pytest.raises(FrozenInstanceError):
        setattr(result, field_name, value)


def test_failure_details_defaults_are_independent() -> None:
    first = StepFailure("first", "Expected failure")
    second = StepFailure("second", "Expected failure")

    first.details["field"] = "payload_xml"

    assert second.details == {}

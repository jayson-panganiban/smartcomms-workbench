import base64
from collections.abc import Callable

import httpx
import pymupdf
import pytest
from pydantic import ValidationError

from smartcomms_workbench.client.gateway import SmartCommsClient
from smartcomms_workbench.client.models import DraftResult, ReviewCaseOperation, ReviewCaseResult
from smartcomms_workbench.client.rest import CloudRestClient, RestContract
from smartcomms_workbench.client.soap import APP_NS, SOAP_NS
from smartcomms_workbench.client.transport import ReviewCaseSoapClient
from smartcomms_workbench.pipeline.builtins import PipelineServices, create_registry
from smartcomms_workbench.pipeline.context import ExecutionContext
from smartcomms_workbench.pipeline.registry import StepRegistry, register_step
from smartcomms_workbench.pipeline.runner import Pipeline, PipelineRecipe, StepSpec
from smartcomms_workbench.pipeline.step import Step, StepFailure, StepResult, StepSuccess


def test_registration_and_fail_fast_keep_last_snapshot() -> None:
    registry = StepRegistry()

    @register_step("prepare", registry=registry)
    class Prepare(Step):
        def __init__(self, xml: str) -> None:
            self.xml = xml

        def execute(self, context: ExecutionContext) -> StepResult:
            return StepSuccess("prepare", context.evolve(payload_xml=self.xml))

    @register_step("fail", registry=registry)
    class Fail(Step):
        def execute(self, context: ExecutionContext) -> StepResult:
            return StepFailure("fail", context.payload_xml or "missing")

    @register_step("unreachable", registry=registry)
    class Unreachable(Step):
        def execute(self, context: ExecutionContext) -> StepResult:
            raise AssertionError(context)

    recipe = Pipeline.parse_recipe("""
context: {environment: test, template_id: t}
steps:
  - name: prepare
    options: {xml: "<r/>"}
  - name: fail
  - name: unreachable
""")
    initial = ExecutionContext("run", "test", "t")
    result = Pipeline.from_recipe(recipe, registry).run(initial)
    assert not result.success and result.failure == StepFailure("fail", "<r/>")
    assert result.context.payload_xml == "<r/>" and initial.payload_xml is None
    assert len(result.results) == 2
    with pytest.raises(ValueError, match="duplicate"):
        registry.register("prepare", lambda _: Prepare("<r/>"))
    with pytest.raises(ValueError, match="Unknown"):
        registry.create("unknown", {})


@pytest.mark.parametrize("exception_type", [TypeError, KeyError])
def test_runner_programming_exceptions_escape(exception_type: type[Exception]) -> None:
    class Broken(Step):
        def execute(self, context: ExecutionContext) -> StepResult:
            raise exception_type(context.run_id)

    with pytest.raises(exception_type):
        Pipeline([Broken()]).run(ExecutionContext("r", "e", "t"))


@pytest.mark.parametrize(
    "yaml",
    [
        "context: {environment: test, template_id: t}\nsteps: []",
        "context: {environment: test, template_id: t}\nsteps: [{name: hydrate, unexpected: 1}]",
        "context: {environment: test, template_id: t}\nsteps: [{name: diff}]\nversion: 2",
        "context: {environment: test, template_id: t}\nsteps: [{name: diff}]\nunknown: true",
    ],
)
def test_recipes_reject_invalid_configuration(yaml: str) -> None:
    with pytest.raises(ValidationError):
        Pipeline.parse_recipe(yaml)


def test_builtin_options_validated_before_execution() -> None:
    recipe = PipelineRecipe.model_validate(
        {
            "context": {"environment": "test", "template_id": "t"},
            "steps": [
                {"name": "hydrate", "options": {"template": "<r/>", "data": {}}},
                {"name": "diff", "options": {"typo": 1}},
            ],
        }
    )
    with pytest.raises(ValidationError):
        Pipeline.from_recipe(recipe, create_registry())
    with pytest.raises(ValueError, match="client"):
        create_registry().create("sc_submit", {})
    with pytest.raises(ValueError, match="regex masks"):
        create_registry().create(
            "diff", {"mode": "text", "masks": [{"rule_type": "bbox", "bbox": [0.0, 0.0, 1.0, 1.0]}]}
        )


def test_payload_pipeline_hydrate_mutate_validate() -> None:
    schema = '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"><xs:element name="amount" type="xs:integer"/></xs:schema>'
    recipe = PipelineRecipe.model_validate(
        {
            "context": {"environment": "test", "template_id": "t"},
            "steps": [
                {"name": "hydrate", "options": {"template": "<amount>{{ amount }}</amount>", "data": {"amount": 1}}},
                {"name": "mutate", "options": {"mutations": [{"xpath": "/amount", "value": "2"}]}},
                {"name": "validate", "options": {"schema": schema}},
            ],
        }
    )
    result = Pipeline.from_recipe(recipe, create_registry()).run(ExecutionContext("r", "test", "t"))
    assert result.success and result.context.payload_xml == "<amount>2</amount>"


def test_rest_pipeline_submit_poll_extract_diff_report(pdf_factory: Callable[[str], bytes]) -> None:
    pdf = pdf_factory("integration")
    paths: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.method == "POST":
            return httpx.Response(200, json={"job_id": "job-1"})
        return httpx.Response(200, json={"status": "completed", "pdf": base64.b64encode(pdf).decode()})

    recipe = PipelineRecipe.model_validate(
        {
            "context": {"environment": "test", "template_id": "t"},
            "steps": [
                {"name": "sc_submit"},
                {"name": "sc_job"},
                {"name": "pdf_extract", "options": {"source": "job"}},
                {"name": "diff", "options": {"report": True}},
            ],
        }
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = SmartCommsClient(
            rest=CloudRestClient(
                RestContract(
                    "https://example.test/submit",
                    job_url="https://example.test/jobs/{job_id}",
                ),
                http,
            )
        )
        pipeline = Pipeline.from_recipe(recipe, create_registry(PipelineServices(client)))
        result = pipeline.run(ExecutionContext("r", "test", "t", payload_xml="<r/>", baseline_pdf=pdf))
    assert result.success and result.context.rendered_pdf == pdf
    assert result.context.metrics["max_changed_pixel_ratio"] == 0
    assert "data:image/png;base64," in result.context.artifacts["report_html"]
    assert paths == ["/submit", "/jobs/job-1"]


def test_diff_mismatch_contains_report_and_diagnostics(pdf_factory: Callable[[str], bytes]) -> None:
    step = create_registry().create("diff", {"report": True})
    result = step.execute(
        ExecutionContext("r", "e", "t", baseline_pdf=pdf_factory("one"), rendered_pdf=pdf_factory("two"))
    )
    assert isinstance(result, StepFailure)
    assert result.error == "visual_mismatch"
    assert "report_html" in result.details and "comparison" in result.details
    assert isinstance(create_registry().create("pdf_extract", {}).execute(ExecutionContext("r", "e", "t")), StepFailure)


def test_registry_copy_is_independent() -> None:
    registry = StepRegistry()
    copied = registry.copy()
    copied.register("hydrate", lambda options: create_registry().create("hydrate", options))
    with pytest.raises(ValueError):
        registry.create("hydrate", {})
    assert copied.create("hydrate", {"template": "<r/>", "data": {}})
    assert StepSpec(name="custom").options == {}


@pytest.mark.parametrize("operation", ["retrieve", "flatten", "expand"])
def test_draft_builtin_preserves_rest_and_soap_contracts(operation: str) -> None:
    xml = "<draft>example</draft>"
    if operation == "flatten":
        body = (
            f'<s:Envelope xmlns:s="{SOAP_NS}" xmlns:a="{APP_NS}"><s:Body>'
            f"<a:FlattenReviewCaseResponse><bytes>{base64.b64encode(xml.encode()).decode()}</bytes>"
            "</a:FlattenReviewCaseResponse></s:Body></s:Envelope>"
        ).encode()
    elif operation == "expand":
        body = (
            f'<s:Envelope xmlns:s="{SOAP_NS}" xmlns:a="{APP_NS}"><s:Body>'
            "<a:ExpandReviewCaseResponse><bytes>expanded</bytes></a:ExpandReviewCaseResponse>"
            "</s:Body></s:Envelope>"
        ).encode()
    else:
        body = b""

    def respond(request: httpx.Request) -> httpx.Response:
        if operation == "retrieve":
            assert request.method == "GET"
            return httpx.Response(200, json={"data": xml})
        assert request.method == "POST"
        return httpx.Response(200, content=body)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = SmartCommsClient(
            rest=CloudRestClient(
                RestContract("https://example.test/submit", "https://example.test/draft/{job_id}"), http
            ),
            soap=ReviewCaseSoapClient(
                flatten_endpoint="https://example.test/flatten",
                expand_endpoint="https://example.test/expand",
                http=http,
            ),
        )
        step = create_registry(PipelineServices(client)).create("sc_draft", {"operation": operation})
        result = step.execute(ExecutionContext("r", "e", "t", payload_xml="<review/>", artifacts={"job_id": "j"}))
    assert isinstance(result, StepSuccess)
    if operation == "retrieve":
        assert result.context.artifacts["draft"] == DraftResult(xml)
    else:
        expected = xml if operation == "flatten" else "expanded"
        assert result.context.artifacts["draft"] == ReviewCaseResult(ReviewCaseOperation(operation), expected)
    assert result.context.payload_xml == (xml if operation != "expand" else "expanded")


@pytest.mark.parametrize("name", ["hydrate", "mutate", "validate"])
def test_payload_builtin_expected_failures(name: str) -> None:
    options = {
        "hydrate": {"template": "<r>{{ missing }}</r>", "data": {}},
        "mutate": {"mutations": [{"xpath": "/r", "value": "x"}]},
        "validate": {"schema": '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"/>'},
    }
    result = create_registry().create(name, options[name]).execute(ExecutionContext("r", "e", "t"))
    assert isinstance(result, StepFailure)


def test_text_pipeline_detects_page_count_mismatch(pdf_factory: Callable[[str], bytes]) -> None:
    one = pdf_factory("one")
    with pymupdf.open(stream=one, filetype="pdf") as document:
        document.new_page()
        two = document.tobytes()
    step = create_registry().create("diff", {"mode": "text"})
    result = step.execute(ExecutionContext("r", "e", "t", rendered_pdf=two, baseline_pdf=one))
    assert isinstance(result, StepFailure) and result.error == "page_count_mismatch"

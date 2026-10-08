"""Built-in step registration with explicit service injection."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from smartcomms_workbench.baseline.store import BaselineStore
from smartcomms_workbench.client.gateway import SmartCommsClient
from smartcomms_workbench.client.models import (
    DraftResult,
    JobResult,
    JobStatus,
    ReviewCaseOperation,
    ReviewCaseResult,
    SoapFault,
    SubmitResult,
    TransportFailure,
)
from smartcomms_workbench.diff.masking import BoundingBoxMaskRule, MaskRule, RegexMaskRule
from smartcomms_workbench.diff.report import render_html_report
from smartcomms_workbench.diff.text import compare_pdf_text
from smartcomms_workbench.diff.visual import ComparisonOptions, ComparisonOutcome, compare_pdf_bytes
from smartcomms_workbench.payload.hydrator import HydrationFailure, hydrate_xml
from smartcomms_workbench.payload.mutator import XPathMutation, mutate_xml, validate_xml
from smartcomms_workbench.payload.pdf_extract import PdfExtractionFailure, extract_pdf
from smartcomms_workbench.pipeline.context import ExecutionContext
from smartcomms_workbench.pipeline.registry import StepRegistry, default_registry
from smartcomms_workbench.pipeline.step import Step, StepFailure, StepResult, StepSuccess


@dataclass(frozen=True)
class PipelineServices:
    client: SmartCommsClient | None = None
    baselines: BaselineStore | None = None


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MaskSpec(Options):
    rule_type: Literal["regex", "bbox"]
    pattern: str | None = None
    bbox: list[float] | None = None
    pages: list[int] | None = None

    def rule(self) -> MaskRule:
        pages = tuple(self.pages) if self.pages is not None else None
        if self.rule_type == "regex":
            if self.pattern is None or self.bbox is not None:
                raise ValueError("Regex mask requires pattern, not bbox")
            return RegexMaskRule(self.pattern, pages)
        if self.bbox is None or len(self.bbox) != 4 or self.pattern is not None:
            raise ValueError("BBox mask requires four coordinates, not pattern")
        return BoundingBoxMaskRule((self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]), pages)


class DiffOptions(Options):
    mode: Literal["visual", "text"] = "visual"
    dpi: int = 144
    channel_tolerance: int = 0
    max_changed_pixel_ratio: float = 0.0
    page_size_tolerance: float = 0.01
    min_ssim: float | None = None
    masks: list[MaskSpec] = Field(default_factory=list)
    baseline_name: str | None = None
    report: bool = False

    def comparison(self) -> ComparisonOptions:
        return ComparisonOptions(
            dpi=self.dpi,
            channel_tolerance=self.channel_tolerance,
            max_changed_pixel_ratio=self.max_changed_pixel_ratio,
            page_size_tolerance=self.page_size_tolerance,
            masks=tuple(mask.rule() for mask in self.masks),
            min_ssim=self.min_ssim,
        )


class DiffStep(Step):
    def __init__(self, options: DiffOptions, baselines: BaselineStore | None = None) -> None:
        self.options = options
        self.baselines = baselines
        self.comparison = options.comparison()
        if options.mode == "text" and any(isinstance(rule, BoundingBoxMaskRule) for rule in self.comparison.masks):
            raise ValueError("Text comparison supports regex masks only")
        if options.mode == "text" and options.report:
            raise ValueError("HTML image reports require visual mode")
        if options.mode == "text" and options.min_ssim is not None:
            raise ValueError("SSIM requires visual mode")
        if options.baseline_name and baselines is None:
            raise ValueError("baseline_name requires a baseline store")

    def execute(self, context: ExecutionContext) -> StepResult:
        baseline = context.baseline_pdf
        if self.options.baseline_name:
            assert self.baselines is not None
            baseline = self.baselines.resolve(self.options.baseline_name)
        if baseline is None or context.rendered_pdf is None:
            return StepFailure("diff", "Both baseline and rendered PDF are required")
        next_context = context.evolve(baseline_pdf=baseline)
        if self.options.mode == "text":
            text_result = compare_pdf_text(
                baseline,
                context.rendered_pdf,
                masks=tuple(rule for rule in self.comparison.masks if isinstance(rule, RegexMaskRule)),
            )
            next_context = next_context.with_artifact("text_comparison", text_result)
            if text_result.outcome != "match":
                return StepFailure("diff", text_result.outcome, {"comparison": text_result})
        else:
            result = compare_pdf_bytes(baseline, context.rendered_pdf, options=self.comparison)
            next_context = next_context.with_artifact("comparison", result)
            metrics = [page.metrics for page in result.pages if page.metrics is not None]
            ratios = [metric.changed_pixel_ratio for metric in metrics if metric.changed_pixel_ratio is not None]
            if ratios:
                next_context = next_context.evolve(
                    metrics={**next_context.metrics, "max_changed_pixel_ratio": max(ratios)}
                )
            similarities = [page.ssim for page in result.pages if page.ssim is not None]
            if similarities:
                next_context = next_context.evolve(metrics={**next_context.metrics, "min_ssim": min(similarities)})
            report = render_html_report(baseline, context.rendered_pdf, result) if self.options.report else None
            if report is not None:
                next_context = next_context.with_artifact("report_html", report)
            if result.outcome != ComparisonOutcome.MATCH:
                details: dict[str, object] = {"comparison": result}
                if report is not None:
                    details["report_html"] = report
                return StepFailure("diff", result.outcome.value, details)
        return StepSuccess("diff", next_context)


class HydrateOptions(Options):
    template: str
    data: dict[str, object]


class HydrateStep(Step):
    def __init__(self, options: HydrateOptions) -> None:
        self.options = options

    def execute(self, context: ExecutionContext) -> StepResult:
        result = hydrate_xml(self.options.template, self.options.data)
        if isinstance(result, HydrationFailure):
            return StepFailure("hydrate", result.message, {"code": result.code})
        return StepSuccess("hydrate", context.evolve(payload_xml=result.xml))


class MutationSpec(Options):
    xpath: str
    operation: Literal["set_text", "set_attribute", "remove"] = "set_text"
    value: str = ""
    attribute: str | None = None


class MutateOptions(Options):
    mutations: list[MutationSpec]
    namespaces: dict[str, str] = Field(default_factory=dict)


class MutateStep(Step):
    def __init__(self, options: MutateOptions) -> None:
        self.namespaces = options.namespaces
        self.mutations = tuple(XPathMutation(**spec.model_dump()) for spec in options.mutations)

    def execute(self, context: ExecutionContext) -> StepResult:
        if context.payload_xml is None:
            return StepFailure("mutate", "Payload XML is required")
        result = mutate_xml(context.payload_xml, self.mutations, namespaces=self.namespaces)
        if isinstance(result, HydrationFailure):
            return StepFailure("mutate", result.message)
        return StepSuccess("mutate", context.evolve(payload_xml=result.xml))


class ValidateOptions(Options):
    schema_xml: str = Field(alias="schema")


class ValidateStep(Step):
    def __init__(self, options: ValidateOptions) -> None:
        self.schema = options.schema_xml
        validate_xml("<validation/>", self.schema)

    def execute(self, context: ExecutionContext) -> StepResult:
        if context.payload_xml is None:
            return StepFailure("validate", "Payload XML is required")
        result = validate_xml(context.payload_xml, self.schema)
        if isinstance(result, HydrationFailure):
            return StepFailure("validate", result.message)
        return StepSuccess("validate", context)


class ClientOptions(Options):
    pass


class SubmitStep(Step):
    def __init__(self, client: SmartCommsClient) -> None:
        self.client = client

    def execute(self, context: ExecutionContext) -> StepResult:
        if context.payload_xml is None:
            return StepFailure("sc_submit", "Payload XML is required")
        result = self.client.submit(context.template_id, context.payload_xml)
        if isinstance(result, TransportFailure):
            return StepFailure("sc_submit", result.message, {"kind": result.kind.value})
        return StepSuccess(
            "sc_submit", context.with_artifact("submission", result).with_artifact("job_id", result.job_id)
        )


class DraftOptions(Options):
    operation: Literal["retrieve", "flatten", "expand"] = "retrieve"


class DraftStep(Step):
    def __init__(self, options: DraftOptions, client: SmartCommsClient) -> None:
        self.options = options
        self.client = client

    def execute(self, context: ExecutionContext) -> StepResult:
        if self.options.operation == "retrieve":
            job_id = context.artifacts.get("job_id")
            if not isinstance(job_id, str):
                return StepFailure("sc_draft", "A submission job ID is required")
            result = self.client.draft(job_id)
        else:
            if context.payload_xml is None:
                return StepFailure("sc_draft", "Review-case payload is required")
            result = self.client.review_case(ReviewCaseOperation(self.options.operation), context.payload_xml)
        if isinstance(result, TransportFailure):
            return StepFailure("sc_draft", result.message, {"kind": result.kind.value})
        if isinstance(result, SoapFault):
            return StepFailure("sc_draft", result.reason, {"code": result.code})
        return StepSuccess("sc_draft", context.with_artifact("draft", result).evolve(payload_xml=result.data))


class JobOptions(Options):
    max_attempts: int = Field(default=1, ge=1)
    interval_seconds: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class JobStep(Step):
    def __init__(self, options: JobOptions, client: SmartCommsClient) -> None:
        self.options = options
        self.client = client

    def execute(self, context: ExecutionContext) -> StepResult:
        job_id = context.artifacts.get("job_id")
        if not isinstance(job_id, str):
            return StepFailure("sc_job", "A submission job ID is required")
        result = self.client.wait(
            job_id, max_attempts=self.options.max_attempts, interval_seconds=self.options.interval_seconds
        )
        if isinstance(result, TransportFailure):
            return StepFailure("sc_job", result.message, {"kind": result.kind.value})
        if result.status == JobStatus.FAILED:
            return StepFailure("sc_job", result.error or "SmartComms job failed")
        return StepSuccess("sc_job", context.with_artifact("job", result))


class ExtractOptions(Options):
    source: str = "submission"
    format: Literal["pdf", "base64", "xml"] = "pdf"
    xpath: str = "//*[local-name()='bytes']/text()"


class ExtractStep(Step):
    def __init__(self, options: ExtractOptions) -> None:
        self.options = options

    def execute(self, context: ExecutionContext) -> StepResult:
        data = context.artifacts.get(self.options.source)
        if isinstance(data, SubmitResult | JobResult):
            data = data.pdf
        elif isinstance(data, DraftResult | ReviewCaseResult):
            data = data.data
        if not isinstance(data, bytes | str):
            return StepFailure("pdf_extract", f"No PDF data in artifact {self.options.source!r}")
        result = extract_pdf(data, format=self.options.format, xpath=self.options.xpath)
        if isinstance(result, PdfExtractionFailure):
            return StepFailure("pdf_extract", result.error)
        return StepSuccess("pdf_extract", context.evolve(rendered_pdf=result))


def create_registry(services: PipelineServices = PipelineServices()) -> StepRegistry:
    registry = default_registry.copy()
    registry.register("hydrate", lambda options: HydrateStep(HydrateOptions.model_validate(options)))
    registry.register("mutate", lambda options: MutateStep(MutateOptions.model_validate(options)))
    registry.register("validate", lambda options: ValidateStep(ValidateOptions.model_validate(options)))
    registry.register("pdf_extract", lambda options: ExtractStep(ExtractOptions.model_validate(options)))
    registry.register("diff", lambda options: DiffStep(DiffOptions.model_validate(options), services.baselines))

    def client() -> SmartCommsClient:
        if services.client is None:
            raise ValueError("SmartComms steps require an injected client")
        return services.client

    def submit(options: Mapping[str, object]) -> Step:
        ClientOptions.model_validate(options)
        configured = client()
        if configured.rest is None:
            raise ValueError("sc_submit requires a REST client")
        return SubmitStep(configured)

    def draft(options: Mapping[str, object]) -> Step:
        parsed = DraftOptions.model_validate(options)
        configured = client()
        if parsed.operation == "retrieve":
            if configured.rest is None or configured.rest.contract.draft_url is None:
                raise ValueError("Draft retrieval requires a REST draft URL")
        elif configured.soap is None:
            raise ValueError("Flatten/expand requires a SOAP client")
        return DraftStep(parsed, configured)

    def job(options: Mapping[str, object]) -> Step:
        configured = client()
        if configured.rest is None or configured.rest.contract.job_url is None:
            raise ValueError("sc_job requires a REST job URL")
        return JobStep(JobOptions.model_validate(options), configured)

    registry.register("sc_submit", submit)
    registry.register("sc_draft", draft)
    registry.register("sc_job", job)
    return registry

"""Thin sc-bench command adapters: filesystem I/O stays at this boundary."""

import json
from collections.abc import Callable
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from uuid import uuid4

import click
import yaml
from pydantic import ValidationError

from smartcomms_workbench.baseline.git_store import GitBaselineStore
from smartcomms_workbench.cli.configuration import ClientConfigurationError, configured_client
from smartcomms_workbench.diff.pdf import PdfInputError
from smartcomms_workbench.payload.hydrator import HydrationFailure
from smartcomms_workbench.payload.matrix import hydrate_matrix, parse_matrix
from smartcomms_workbench.payload.mutator import XPathMutation, mutate_xml, validate_xml
from smartcomms_workbench.pipeline.builtins import (
    DiffOptions,
    DiffStep,
    MutateOptions,
    PipelineServices,
    create_registry,
)
from smartcomms_workbench.pipeline.context import ExecutionContext
from smartcomms_workbench.pipeline.runner import Pipeline
from smartcomms_workbench.pipeline.step import StepFailure, StepSuccess

INPUT_FILE = click.Path(exists=True, dir_okay=False, path_type=Path)
OUTPUT_FILE = click.Path(dir_okay=False, path_type=Path)
DIRECTORY = click.Path(file_okay=False, path_type=Path)


def input_errors[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return function(*args, **kwargs)
        except (OSError, yaml.YAMLError, ValidationError, PdfInputError, ClientConfigurationError) as error:
            raise click.ClickException(str(error)) from error

    return wrapped


@click.group()
def cli() -> None:
    """SmartComms automation, payload generation and document verification."""


def _check_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise click.ClickException("Output directory must be empty; choose a new --output directory")


def _diff_options(
    text: bool, masks: Path | None, dpi: int, tolerance: int, threshold: float, report: bool, min_ssim: float | None
) -> DiffOptions:
    rules = yaml.safe_load(masks.read_text(encoding="utf-8")) if masks else []
    return DiffOptions.model_validate(
        {
            "mode": "text" if text else "visual",
            "masks": rules,
            "dpi": dpi,
            "channel_tolerance": tolerance,
            "max_changed_pixel_ratio": threshold,
            "report": report,
            "min_ssim": min_ssim,
        }
    )


def _comparison(reference: bytes, candidate: bytes, options: DiffOptions, report: Path | None) -> None:
    try:
        step = DiffStep(options)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    result = step.execute(ExecutionContext("diff", "", "", baseline_pdf=reference, rendered_pdf=candidate))
    if isinstance(result, StepFailure):
        html = result.details.get("report_html")
        comparison = result.details.get("comparison")
        if comparison is not None:
            click.echo(json.dumps(asdict(comparison)))
        else:
            click.echo(result.error)
    else:
        html = result.context.artifacts.get("report_html")
        comparison = result.context.artifacts.get("comparison", result.context.artifacts.get("text_comparison"))
        click.echo(json.dumps(asdict(comparison)))
    if report is not None and isinstance(html, str):
        report.write_text(html, encoding="utf-8")
    if isinstance(result, StepFailure):
        raise click.ClickException(result.error)


@cli.command("diff")
@click.argument("reference", type=INPUT_FILE)
@click.argument("candidate", type=INPUT_FILE)
@click.option("--text", is_flag=True, help="Compare PDF text tokens instead of pixels.")
@click.option("--masks", type=INPUT_FILE, help="YAML list of regex/bbox masks.")
@click.option("--dpi", type=click.IntRange(min=1), default=144, show_default=True)
@click.option("--channel-tolerance", type=click.IntRange(0, 255), default=0)
@click.option("--max-changed-pixel-ratio", type=click.FloatRange(0, 1), default=0.0)
@click.option(
    "--min-ssim", type=click.FloatRange(-1, 1), help="Require mask-aware SSIM in addition to pixel thresholds."
)
@click.option("--report", type=OUTPUT_FILE, help="Write a self-contained visual HTML report, also on mismatch.")
@input_errors
def diff_command(
    reference: Path,
    candidate: Path,
    text: bool,
    masks: Path | None,
    dpi: int,
    channel_tolerance: int,
    max_changed_pixel_ratio: float,
    report: Path | None,
    min_ssim: float | None,
) -> None:
    options = _diff_options(text, masks, dpi, channel_tolerance, max_changed_pixel_ratio, report is not None, min_ssim)
    _comparison(reference.read_bytes(), candidate.read_bytes(), options, report)


@cli.group()
def baseline() -> None:
    """Approve or compare directory-backed, manually Git-versioned baselines."""


@baseline.command("approve")
@click.argument("name")
@click.argument("candidate", type=INPUT_FILE)
@click.option("--store", type=DIRECTORY, default=Path("baselines"), show_default=True)
@input_errors
def baseline_approve(name: str, candidate: Path, store: Path) -> None:
    try:
        GitBaselineStore(store).approve(name, candidate.read_bytes())
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Approved baseline: {name}")


@baseline.command("check")
@click.argument("name")
@click.argument("candidate", type=INPUT_FILE)
@click.option("--store", type=DIRECTORY, default=Path("baselines"), show_default=True)
@click.option("--text", is_flag=True)
@click.option("--masks", type=INPUT_FILE)
@click.option("--dpi", type=click.IntRange(min=1), default=144)
@click.option("--channel-tolerance", type=click.IntRange(0, 255), default=0)
@click.option("--max-changed-pixel-ratio", type=click.FloatRange(0, 1), default=0.0)
@click.option("--min-ssim", type=click.FloatRange(-1, 1))
@click.option("--report", type=OUTPUT_FILE)
@input_errors
def baseline_check(
    name: str,
    candidate: Path,
    store: Path,
    text: bool,
    masks: Path | None,
    dpi: int,
    channel_tolerance: int,
    max_changed_pixel_ratio: float,
    report: Path | None,
    min_ssim: float | None,
) -> None:
    try:
        reference = GitBaselineStore(store).resolve(name)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    if reference is None:
        raise click.ClickException(f"No approved baseline: {name}")
    options = _diff_options(text, masks, dpi, channel_tolerance, max_changed_pixel_ratio, report is not None, min_ssim)
    _comparison(reference, candidate.read_bytes(), options, report)


@cli.command("hydrate")
@click.argument("template", type=INPUT_FILE)
@click.option("--data", type=INPUT_FILE, required=True, help="CSV or JSON record matrix.")
@click.option("--format", "matrix_format", type=click.Choice(["csv", "json"]))
@click.option("--output", type=DIRECTORY, required=True)
@click.option("--mutations", type=INPUT_FILE, help="YAML mutations/namespaces configuration.")
@click.option("--schema", type=INPUT_FILE, help="Validate each rendered payload against an XSD.")
@input_errors
def hydrate_command(
    template: Path,
    data: Path,
    matrix_format: str | None,
    output: Path,
    mutations: Path | None,
    schema: Path | None,
) -> None:
    _check_output(output)
    selected = matrix_format or data.suffix.lstrip(".").lower()
    if selected not in ("csv", "json"):
        raise click.ClickException("Use a .csv/.json data file or specify --format")
    try:
        records = parse_matrix(data.read_text(encoding="utf-8"), format=selected)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    if not records:
        raise click.ClickException("Payload matrix contains no records")
    mutation_options = (
        MutateOptions.model_validate(yaml.safe_load(mutations.read_text(encoding="utf-8"))) if mutations else None
    )
    schema_xml = schema.read_text(encoding="utf-8") if schema else None
    xmls: list[str] = []
    for index, result in enumerate(hydrate_matrix(template.read_text(encoding="utf-8"), records), 1):
        if not isinstance(result, HydrationFailure) and mutation_options is not None:
            result = mutate_xml(
                result.xml,
                [XPathMutation(**spec.model_dump()) for spec in mutation_options.mutations],
                namespaces=mutation_options.namespaces,
            )
        if not isinstance(result, HydrationFailure) and schema_xml is not None:
            result = validate_xml(result.xml, schema_xml)
        if isinstance(result, HydrationFailure):
            raise click.ClickException(f"Record {index}: {result.message}")
        xmls.append(result.xml)
    output.mkdir(parents=True, exist_ok=True)
    for index, xml in enumerate(xmls, 1):
        (output / f"payload-{index:04d}.xml").write_text(xml, encoding="utf-8")
    click.echo(f"Hydrated {len(xmls)} payloads into {output}")


@cli.command("run")
@click.argument("recipe_path", type=INPUT_FILE)
@click.option("--client", "client_path", type=INPUT_FILE, help="Endpoint configuration with credential env references.")
@click.option("--output", type=DIRECTORY, default=Path("artifacts"), show_default=True)
@input_errors
def run_command(recipe_path: Path, client_path: Path | None, output: Path) -> None:
    _check_output(output)
    recipe = Pipeline.parse_recipe(recipe_path.read_text(encoding="utf-8"))
    root = recipe_path.resolve().parent
    context = ExecutionContext(
        recipe.context.run_id or str(uuid4()),
        recipe.context.environment,
        recipe.context.template_id,
        payload_xml=(root / recipe.payload).read_text(encoding="utf-8") if recipe.payload else None,
        rendered_pdf=(root / recipe.candidate).read_bytes() if recipe.candidate else None,
        baseline_pdf=(root / recipe.baseline).read_bytes() if recipe.baseline else None,
    )
    baselines = GitBaselineStore(root / recipe.baseline_store) if recipe.baseline_store else None
    # Configuration failures are adapted before execution; programming bugs in
    # step.execute deliberately remain uncaught.
    with configured_client(client_path) as client:
        try:
            registry = create_registry(PipelineServices(client, baselines))
            pipeline = Pipeline.from_recipe(recipe, registry)
        except ValueError as error:
            raise click.ClickException(str(error)) from error
        result = pipeline.run(context)
    output.mkdir(parents=True, exist_ok=True)
    if result.context.rendered_pdf is not None:
        (output / "rendered.pdf").write_bytes(result.context.rendered_pdf)
    if result.context.payload_xml is not None:
        (output / "payload.xml").write_text(result.context.payload_xml, encoding="utf-8")
    html = result.context.artifacts.get("report_html")
    if result.failure is not None:
        html = result.failure.details.get("report_html", html)
    if isinstance(html, str):
        (output / "report.html").write_text(html, encoding="utf-8")
    summary = {
        "success": result.success,
        "run_id": result.context.run_id,
        "environment": result.context.environment,
        "template_id": result.context.template_id,
        "metrics": dict(result.context.metrics),
        "artifacts": list(result.context.artifacts),
        "steps": [
            {
                "name": item.step_name,
                "success": isinstance(item, StepSuccess),
                "message": item.message if isinstance(item, StepSuccess) else item.error,
            }
            for item in result.results
        ],
    }
    comparison = result.context.artifacts.get("comparison", result.context.artifacts.get("text_comparison"))
    if result.failure is not None:
        comparison = result.failure.details.get("comparison", comparison)
    if comparison is not None:
        summary["comparison"] = asdict(comparison)
    serialized = json.dumps(summary, indent=2)
    (output / "run.json").write_text(serialized, encoding="utf-8")
    click.echo(serialized)
    if result.failure is not None:
        raise click.ClickException(f"{result.failure.step_name}: {result.failure.error}")

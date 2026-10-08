import base64
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

from smartcomms_workbench.cli.main import cli


def test_cli_help_has_all_surfaces() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for name in ("run", "diff", "baseline", "hydrate"):
        assert name in result.output


def test_cli_baseline_and_diff_workflows(tmp_path: Path, pdf_factory: Callable[[str], bytes]) -> None:
    runner = CliRunner()
    candidate = tmp_path / "candidate.pdf"
    candidate.write_bytes(pdf_factory("one"))
    store = tmp_path / "baselines"
    assert runner.invoke(cli, ["baseline", "approve", "t", str(candidate), "--store", str(store)]).exit_code == 0
    assert runner.invoke(cli, ["baseline", "check", "t", str(candidate), "--store", str(store)]).exit_code == 0
    assert runner.invoke(cli, ["baseline", "check", "missing", str(candidate), "--store", str(store)]).exit_code == 1
    candidate.write_bytes(pdf_factory("two"))
    report = tmp_path / "report.html"
    result = runner.invoke(cli, ["diff", str(store / "t.pdf"), str(candidate), "--report", str(report)])
    assert result.exit_code == 1 and "visual_mismatch" in result.output
    assert report.exists() and "data:image/png;base64," in report.read_text(encoding="utf-8")
    result = runner.invoke(cli, ["diff", str(candidate), str(candidate), "--text"])
    assert result.exit_code == 0 and '"outcome": "match"' in result.output


def test_cli_hydration_persists_matrix(tmp_path: Path) -> None:
    template, data, output = tmp_path / "template.xml", tmp_path / "matrix.json", tmp_path / "out"
    template.write_text("<r>{{ value }}</r>", encoding="utf-8")
    data.write_text('[{"value":"A & B"},{"value":"C"}]', encoding="utf-8")
    result = CliRunner().invoke(cli, ["hydrate", str(template), "--data", str(data), "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert (output / "payload-0001.xml").read_text(encoding="utf-8") == "<r>A &amp; B</r>"
    assert (output / "payload-0002.xml").read_text(encoding="utf-8") == "<r>C</r>"


def test_cli_hydration_no_partial_outputs_on_validation_failure(tmp_path: Path) -> None:
    template, data, output = tmp_path / "template.xml", tmp_path / "matrix.csv", tmp_path / "out"
    template.write_text("<r>{{ value }}</r>", encoding="utf-8")
    data.write_text("wrong\nvalue\n", encoding="utf-8")
    result = CliRunner().invoke(cli, ["hydrate", str(template), "--data", str(data), "--output", str(output)])
    assert result.exit_code == 1 and "Record 1" in result.output
    assert not output.exists()


@pytest.mark.parametrize("same", [True, False])
def test_cli_run_resolves_recipe_paths_and_persists_outcome(
    tmp_path: Path,
    pdf_factory: Callable[[str], bytes],
    same: bool,
) -> None:
    recipe = tmp_path / "pipeline.yaml"
    (tmp_path / "reference.pdf").write_bytes(pdf_factory("one"))
    (tmp_path / "candidate.pdf").write_bytes(pdf_factory("one" if same else "two"))
    recipe.write_text(
        """
context: {environment: test, template_id: t}
baseline: reference.pdf
candidate: candidate.pdf
steps:
  - name: diff
    options: {report: true}
""",
        encoding="utf-8",
    )
    output = tmp_path / "artifacts"
    result = CliRunner().invoke(cli, ["run", str(recipe), "--output", str(output)])
    assert result.exit_code == (0 if same else 1), result.output
    summary = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert summary["success"] is same
    assert summary["comparison"]["outcome"] == ("match" if same else "visual_mismatch")
    assert (output / "rendered.pdf").read_bytes() == (tmp_path / "candidate.pdf").read_bytes()
    assert (output / "report.html").is_file()


def test_cli_bad_recipe_is_explicit(tmp_path: Path) -> None:
    recipe = tmp_path / "pipeline.yaml"
    recipe.write_text("steps: []", encoding="utf-8")
    output = tmp_path / "artifacts"
    result = CliRunner().invoke(cli, ["run", str(recipe), "--output", str(output)])
    assert result.exit_code == 1 and "Error:" in result.output
    assert not output.exists()


def test_cli_never_reuses_stale_output(tmp_path: Path) -> None:
    recipe = tmp_path / "pipeline.yaml"
    recipe.write_text(
        'context: {environment: test, template_id: t}\nsteps: [{name: hydrate, options: {template: "<r/>", data: {}}}]',
        encoding="utf-8",
    )
    output = tmp_path / "artifacts"
    output.mkdir()
    old = output / "run.json"
    old.write_text("old-result", encoding="utf-8")
    result = CliRunner().invoke(cli, ["run", str(recipe), "--output", str(output)])
    assert result.exit_code == 1 and "must be empty" in result.output
    assert old.read_text(encoding="utf-8") == "old-result"


def test_cli_auth_missing_credentials_is_explicit(tmp_path: Path) -> None:
    recipe = tmp_path / "pipeline.yaml"
    recipe.write_text("context: {environment: test, template_id: t}\nsteps: [{name: sc_submit}]", encoding="utf-8")
    config = tmp_path / "client.yaml"
    config.write_text("rest: {submit_url: https://example.test/submit}\nauth: {type: bearer}", encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        [
            "run",
            str(recipe),
            "--client",
            str(config),
            "--output",
            str(tmp_path / "artifacts"),
        ],
    )
    assert result.exit_code == 1 and "environment variable reference" in result.output


def test_cli_ssim_and_regex_masks(tmp_path: Path, pdf_factory: Callable[[str], bytes]) -> None:
    left, right, masks = tmp_path / "left.pdf", tmp_path / "right.pdf", tmp_path / "masks.yaml"
    left.write_bytes(pdf_factory("Date: 123"))
    right.write_bytes(pdf_factory("Date: 456"))
    masks.write_text('- {rule_type: regex, pattern: "Date: [0-9]+"}', encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        [
            "diff",
            str(left),
            str(right),
            "--masks",
            str(masks),
            "--min-ssim",
            "1",
        ],
    )
    assert result.exit_code == 0 and '"ssim": 1.0' in result.output


def test_cli_hydration_wires_mutations_and_schema(tmp_path: Path) -> None:
    template, data = tmp_path / "template.xml", tmp_path / "matrix.json"
    mutations, schema = tmp_path / "mutations.yaml", tmp_path / "schema.xsd"
    template.write_text("<amount>{{ value }}</amount>", encoding="utf-8")
    data.write_text('[{"value":"invalid-number"}]', encoding="utf-8")
    mutations.write_text('mutations: [{xpath: /amount, value: "42"}]', encoding="utf-8")
    schema.write_text(
        '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">'
        '<xs:element name="amount" type="xs:integer"/></xs:schema>',
        encoding="utf-8",
    )
    output = tmp_path / "out"
    result = CliRunner().invoke(
        cli,
        [
            "hydrate",
            str(template),
            "--data",
            str(data),
            "--mutations",
            str(mutations),
            "--schema",
            str(schema),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output / "payload-0001.xml").read_text(encoding="utf-8") == "<amount>42</amount>"


def test_cli_network_recipe_uses_oauth2_and_persists_pdf(
    tmp_path: Path,
    pdf_factory: Callable[[str], bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = pdf_factory("network workflow")
    recipe, config, output = tmp_path / "pipeline.yaml", tmp_path / "client.yaml", tmp_path / "out"
    recipe.write_text(
        """
context: {environment: test, template_id: t}
steps:
  - name: hydrate
    options: {template: "<r/>", data: {}}
  - name: sc_submit
  - name: pdf_extract
""",
        encoding="utf-8",
    )
    config.write_text(
        """
rest: {submit_url: https://example.test/submit}
auth:
  type: oauth2
  token_url: https://example.test/token
  client_id_env: SC_TEST_CLIENT_ID
  client_secret_env: SC_TEST_CLIENT_SECRET
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("SC_TEST_CLIENT_ID", "test-id")
    monkeypatch.setenv("SC_TEST_CLIENT_SECRET", "test-secret")
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/token":
            assert request.headers["Authorization"].startswith("Basic ")
            return httpx.Response(200, json={"token_type": "Bearer", "access_token": "test-access-token"})
        assert request.headers["Authorization"] == "Bearer test-access-token"
        assert json.loads(request.content) == {"template_id": "t", "payload_xml": "<r/>"}
        return httpx.Response(200, json={"job_id": "j", "pdf": base64.b64encode(pdf).decode()})

    real_client = httpx.Client

    def http_client(*, timeout: float) -> httpx.Client:
        return real_client(timeout=timeout, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(httpx, "Client", http_client)
    result = CliRunner().invoke(cli, ["run", str(recipe), "--client", str(config), "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert calls == ["/token", "/submit"]
    assert (output / "rendered.pdf").read_bytes() == pdf
    assert "test-access-token" not in (output / "run.json").read_text(encoding="utf-8")

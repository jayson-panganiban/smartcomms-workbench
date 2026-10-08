from collections.abc import Callable
from pathlib import Path

import pytest

from smartcomms_workbench.baseline.git_store import GitBaselineStore
from smartcomms_workbench.diff.masking import RegexMaskRule
from smartcomms_workbench.diff.pdf import PdfInputError
from smartcomms_workbench.diff.report import render_html_report
from smartcomms_workbench.diff.text import compare_pdf_text, compare_text
from smartcomms_workbench.diff.visual import compare_pdf_bytes


def test_baseline_atomic_approval_and_replacement(tmp_path: Path, pdf_factory: Callable[[str], bytes]) -> None:
    store = GitBaselineStore(tmp_path / "baselines")
    first, second = pdf_factory("one"), pdf_factory("two")
    assert store.resolve("template-1") is None
    store.approve("template-1", first)
    assert store.resolve("template-1") == first
    store.approve("template-1", second)
    assert store.resolve("template-1") == second
    with pytest.raises(PdfInputError):
        store.approve("template-1", b"invalid")
    assert store.resolve("template-1") == second
    assert list((tmp_path / "baselines").iterdir()) == [tmp_path / "baselines" / "template-1.pdf"]


@pytest.mark.parametrize(
    "name", ["", "..", "../escape", r"..\escape", "C:drive", "CON", "aux.txt", "LPT1", "trailing."]
)
def test_baseline_rejects_unsafe_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        GitBaselineStore(tmp_path).resolve(name)


def test_token_comparison_reports_changes_and_ignores_whitespace() -> None:
    assert compare_text("A \n B", "A B").outcome == "match"
    result = compare_text("A B C", "A D C E")
    assert result.outcome == "text_mismatch"
    assert [(change.operation, change.reference, change.candidate) for change in result.changes] == [
        ("replace", ("B",), ("D",)),
        ("insert", (), ("E",)),
    ]
    assert compare_text("A", "a").outcome == "text_mismatch"


def test_pdf_text_regex_masking_and_invalid_inputs(pdf_factory: Callable[[str], bytes]) -> None:
    left, right = pdf_factory("Date: 123"), pdf_factory("Date: 456")
    assert compare_pdf_text(left, right).outcome == "text_mismatch"
    assert compare_pdf_text(left, right, masks=[RegexMaskRule(r"Date: \d+", (1,))]).outcome == "match"
    assert compare_pdf_text(left, right, masks=[RegexMaskRule(r"Date: \d+", (2,))]).outcome == "text_mismatch"
    assert compare_pdf_text(b"bad", right).outcome == "invalid_pdf"


def test_html_report_is_self_contained_and_escapes_title(pdf_factory: Callable[[str], bytes]) -> None:
    left, right = pdf_factory("one"), pdf_factory("two")
    result = compare_pdf_bytes(left, right)
    html = render_html_report(left, right, result, title="<script>bad</script>")
    assert "<script>" not in html
    assert "&lt;script&gt;bad&lt;/script&gt;" in html
    assert html.count("data:image/png;base64,") == 2
    assert "visual_mismatch" in html and "Changed pixel ratio" in html
    invalid = render_html_report(b"bad", right, compare_pdf_bytes(b"bad", right))
    assert "invalid_pdf" in invalid and "not a readable PDF" in invalid

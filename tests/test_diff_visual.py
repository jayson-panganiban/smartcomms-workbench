from dataclasses import FrozenInstanceError
from math import nextafter

import pymupdf
import pytest

from smartcomms_workbench.diff.masking import BoundingBoxMaskRule, RegexMaskRule
from smartcomms_workbench.diff.visual import ComparisonOptions, ComparisonOutcome, compare_pdf_bytes


def make_pdf(
    *,
    page_count: int = 1,
    size: tuple[float, float] = (72, 72),
    changed_pages: tuple[int, ...] = (),
    text: str | None = None,
    rotation: int = 0,
    crop: bool = False,
) -> bytes:
    with pymupdf.open() as document:
        for index in range(page_count):
            page = document.new_page(width=size[0], height=size[1])
            if index + 1 in changed_pages:
                page.draw_rect(pymupdf.Rect(8, 8, 16, 16), color=None, fill=(0, 0, 0))
            if text:
                page.insert_text((8, 40), text, fontsize=8)
            if crop:
                page.set_cropbox(pymupdf.Rect(4, 4, size[0] - 4, size[1] - 4))
            page.set_rotation(rotation)
        return document.tobytes(no_new_id=True)


def test_identical_documents_have_zero_metrics() -> None:
    pdf = make_pdf(page_count=2, text="Hello")
    result = compare_pdf_bytes(pdf, pdf)
    assert result.outcome == ComparisonOutcome.MATCH
    assert result.reference_page_count == result.candidate_page_count == 2
    assert tuple(page.page for page in result.pages) == (1, 2)
    for page in result.pages:
        assert page.reference_pixels == page.candidate_pixels == (144, 144)
        assert page.metrics is not None
        assert page.metrics.comparable_pixels == 144 * 144
        assert page.metrics.changed_pixels == 0
        assert page.metrics.changed_pixel_ratio == 0
        assert page.metrics.mean_absolute_error == 0


def test_visible_difference_is_explicit() -> None:
    result = compare_pdf_bytes(make_pdf(), make_pdf(changed_pages=(1,)))
    assert result.outcome == ComparisonOutcome.VISUAL_MISMATCH
    metrics = result.pages[0].metrics
    assert metrics is not None
    assert metrics.changed_pixels > 0
    assert metrics.changed_pixel_ratio is not None and metrics.changed_pixel_ratio > 0
    assert metrics.mean_absolute_error is not None and metrics.mean_absolute_error > 0


def test_geometric_mask_excludes_visible_difference_on_blank_page() -> None:
    options = ComparisonOptions(masks=(BoundingBoxMaskRule((6, 6, 18, 18)),))
    result = compare_pdf_bytes(make_pdf(), make_pdf(changed_pages=(1,)), options=options)
    assert result.outcome == ComparisonOutcome.MATCH
    assert result.pages[0].metrics is not None
    assert result.pages[0].metrics.changed_pixels == 0
    assert result.pages[0].metrics.comparable_pixels == 144 * 144 - 24 * 24


def test_mask_outside_difference_does_not_hide_it() -> None:
    options = ComparisonOptions(masks=(BoundingBoxMaskRule((30, 30, 40, 40)),))
    result = compare_pdf_bytes(make_pdf(), make_pdf(changed_pages=(1,)), options=options)
    assert result.outcome == ComparisonOutcome.VISUAL_MISMATCH


def test_page_selectors_remain_one_based() -> None:
    reference = make_pdf(page_count=2)
    candidate = make_pdf(page_count=2, changed_pages=(1, 2))
    options = ComparisonOptions(masks=(BoundingBoxMaskRule((6, 6, 18, 18), pages=(1,)),))
    result = compare_pdf_bytes(reference, candidate, options=options)
    assert result.outcome == ComparisonOutcome.VISUAL_MISMATCH
    assert result.pages[0].outcome == ComparisonOutcome.MATCH
    assert result.pages[1].outcome == ComparisonOutcome.VISUAL_MISMATCH


def test_empty_and_absent_page_selectors_do_not_mask() -> None:
    reference = make_pdf()
    candidate = make_pdf(changed_pages=(1,))
    for selected in ((), (2,)):
        options = ComparisonOptions(masks=(BoundingBoxMaskRule((0, 0, 72, 72), pages=selected),))
        assert compare_pdf_bytes(reference, candidate, options=options).outcome == ComparisonOutcome.VISUAL_MISMATCH


def test_regex_masks_cover_union_from_both_documents() -> None:
    options = ComparisonOptions(masks=(RegexMaskRule(r"Ref: \d+"),))
    reference = make_pdf(text="Ref: 123")
    candidate = make_pdf(text="Ref: 99999")
    assert compare_pdf_bytes(reference, candidate).outcome == ComparisonOutcome.VISUAL_MISMATCH
    assert compare_pdf_bytes(reference, candidate, options=options).outcome == ComparisonOutcome.MATCH
    assert compare_pdf_bytes(reference, make_pdf(), options=options).outcome == ComparisonOutcome.MATCH
    assert compare_pdf_bytes(make_pdf(), candidate, options=options).outcome == ComparisonOutcome.MATCH


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_regex_masks_follow_rotation_and_crop(rotation: int) -> None:
    options = ComparisonOptions(masks=(RegexMaskRule(r"Ref: \d+"),))
    reference = make_pdf(size=(100, 72), text="Ref: 123", rotation=rotation, crop=True)
    candidate = make_pdf(size=(100, 72), text="Ref: 999", rotation=rotation, crop=True)
    assert compare_pdf_bytes(reference, candidate, options=options).outcome == ComparisonOutcome.MATCH


def test_page_count_mismatch_has_no_fake_metrics() -> None:
    result = compare_pdf_bytes(make_pdf(), make_pdf(page_count=2))
    assert result.outcome == ComparisonOutcome.PAGE_COUNT_MISMATCH
    assert (result.reference_page_count, result.candidate_page_count) == (1, 2)
    assert result.pages == ()


def test_page_size_mismatch_has_no_fake_metrics() -> None:
    result = compare_pdf_bytes(make_pdf(), make_pdf(size=(73, 72)))
    assert result.outcome == ComparisonOutcome.PAGE_SIZE_MISMATCH
    assert result.pages[0].reference_size == (72, 72)
    assert result.pages[0].candidate_size == (73, 72)
    assert result.pages[0].metrics is None
    assert result.pages[0].reference_pixels is None


def test_raster_size_difference_is_not_resized_even_with_large_point_tolerance() -> None:
    result = compare_pdf_bytes(make_pdf(), make_pdf(size=(73, 72)), options=ComparisonOptions(page_size_tolerance=2))
    assert result.outcome == ComparisonOutcome.PAGE_SIZE_MISMATCH
    assert result.pages[0].reference_pixels == (144, 144)
    assert result.pages[0].candidate_pixels == (146, 144)
    assert result.pages[0].metrics is None


def test_point_size_tolerance_is_inclusive() -> None:
    reference = make_pdf(size=(71, 72))
    candidate = make_pdf(size=(71.003, 72))
    first = compare_pdf_bytes(reference, candidate, options=ComparisonOptions(page_size_tolerance=0))
    assert first.outcome == ComparisonOutcome.PAGE_SIZE_MISMATCH
    measured_delta = first.pages[0].candidate_size[0] - first.pages[0].reference_size[0]
    # At 10 DPI this sub-point difference does not alter raster dimensions.
    equal = ComparisonOptions(dpi=10, page_size_tolerance=measured_delta)
    below = ComparisonOptions(dpi=10, page_size_tolerance=nextafter(measured_delta, 0))
    assert compare_pdf_bytes(reference, candidate, options=equal).outcome == ComparisonOutcome.MATCH
    assert compare_pdf_bytes(reference, candidate, options=below).outcome == ComparisonOutcome.PAGE_SIZE_MISMATCH


@pytest.mark.parametrize("invalid", [b"", b"not a PDF", b"%PDF-1.7\ntruncated"])
@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_invalid_document_identifies_input_side(invalid: bytes, side: str) -> None:
    valid = make_pdf()
    reference, candidate = (invalid, valid) if side == "reference" else (valid, invalid)
    result = compare_pdf_bytes(reference, candidate)
    assert result.outcome == ComparisonOutcome.INVALID_PDF
    assert result.invalid_side == side
    assert result.error
    assert result.pages == ()


def test_fully_masked_page_is_inconclusive_not_success() -> None:
    pdf = make_pdf()
    options = ComparisonOptions(masks=(BoundingBoxMaskRule((0, 0, 72, 72)),))
    result = compare_pdf_bytes(pdf, pdf, options=options)
    assert result.outcome == ComparisonOutcome.NO_COMPARABLE_PIXELS
    assert result.pages[0].metrics is not None
    assert result.pages[0].metrics.comparable_pixels == 0
    assert result.pages[0].metrics.changed_pixel_ratio is None


def test_mismatch_is_not_diluted_by_matching_or_masked_pages() -> None:
    options = ComparisonOptions(masks=(BoundingBoxMaskRule((0, 0, 72, 72), pages=(1,)),))
    result = compare_pdf_bytes(make_pdf(page_count=3), make_pdf(page_count=3, changed_pages=(2,)), options=options)
    assert result.outcome == ComparisonOutcome.VISUAL_MISMATCH
    assert [page.outcome for page in result.pages] == [
        ComparisonOutcome.NO_COMPARABLE_PIXELS,
        ComparisonOutcome.VISUAL_MISMATCH,
        ComparisonOutcome.MATCH,
    ]


def test_threshold_boundary_is_inclusive_in_real_pdf_comparison() -> None:
    reference = make_pdf()
    candidate = make_pdf(changed_pages=(1,))
    initial = compare_pdf_bytes(reference, candidate)
    metrics = initial.pages[0].metrics
    assert metrics is not None
    assert metrics.changed_pixel_ratio is not None
    exact = ComparisonOptions(max_changed_pixel_ratio=metrics.changed_pixel_ratio)
    below = ComparisonOptions(max_changed_pixel_ratio=nextafter(metrics.changed_pixel_ratio, 0))
    assert compare_pdf_bytes(reference, candidate, options=exact).outcome == ComparisonOutcome.MATCH
    assert compare_pdf_bytes(reference, candidate, options=below).outcome == ComparisonOutcome.VISUAL_MISMATCH


def test_channel_tolerance_and_dpi_are_used() -> None:
    result = compare_pdf_bytes(
        make_pdf(), make_pdf(changed_pages=(1,)), options=ComparisonOptions(dpi=72, channel_tolerance=255)
    )
    assert result.outcome == ComparisonOutcome.MATCH
    assert result.pages[0].reference_pixels == (72, 72)
    assert result.pages[0].metrics is not None
    assert result.pages[0].metrics.mean_absolute_error is not None
    assert result.pages[0].metrics.mean_absolute_error > 0


def test_programming_type_errors_propagate() -> None:
    with pytest.raises(TypeError, match="bytes"):
        compare_pdf_bytes("not bytes", make_pdf())  # ty: ignore[invalid-argument-type] - exercise runtime contract
    with pytest.raises(TypeError, match="ComparisonOptions"):
        compare_pdf_bytes(make_pdf(), make_pdf(), options=None)  # ty: ignore[invalid-argument-type]


def test_invalid_regex_propagates_even_with_no_matching_pages() -> None:
    import re

    with pytest.raises(re.PatternError):
        ComparisonOptions(masks=(RegexMaskRule("[", pages=()),))


def test_results_and_options_are_frozen() -> None:
    options = ComparisonOptions()
    result = compare_pdf_bytes(make_pdf(), make_pdf(), options=options)
    for target, attribute, value in (
        (options, "dpi", 72),
        (result, "outcome", ComparisonOutcome.MATCH),
        (result.pages[0], "page", 2),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(target, attribute, value)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_ratio_threshold_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="max_changed_pixel_ratio"):
        ComparisonOptions(max_changed_pixel_ratio=threshold)


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_page_size_tolerance_rejected(tolerance: float) -> None:
    with pytest.raises(ValueError, match="page_size_tolerance"):
        ComparisonOptions(page_size_tolerance=tolerance)


@pytest.mark.parametrize("dpi", [0, -1])
def test_invalid_dpi_rejected(dpi: int) -> None:
    with pytest.raises(ValueError, match="dpi"):
        ComparisonOptions(dpi=dpi)


@pytest.mark.parametrize("tolerance", [-1, 256])
def test_invalid_channel_tolerance_rejected(tolerance: int) -> None:
    with pytest.raises(ValueError, match="channel_tolerance"):
        ComparisonOptions(channel_tolerance=tolerance)


def test_wrong_option_types_raise() -> None:
    with pytest.raises(TypeError, match="dpi"):
        ComparisonOptions(dpi=True)
    with pytest.raises(TypeError, match="channel_tolerance"):
        ComparisonOptions(channel_tolerance=True)
    with pytest.raises(TypeError, match="immutable tuple"):
        ComparisonOptions(masks=[])  # ty: ignore[invalid-argument-type] - exercise runtime contract
    with pytest.raises(TypeError, match="MaskRule"):
        ComparisonOptions(masks=("unknown",))  # ty: ignore[invalid-argument-type] - exercise runtime contract
    with pytest.raises(ValueError, match="max_changed_pixel_ratio"):
        ComparisonOptions(max_changed_pixel_ratio=True)
    with pytest.raises(ValueError, match="page_size_tolerance"):
        ComparisonOptions(page_size_tolerance=True)

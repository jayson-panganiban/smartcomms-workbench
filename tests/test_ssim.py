from collections.abc import Callable
from math import nextafter

import numpy as np
import pytest

from smartcomms_workbench.diff.image import calculate_structural_similarity
from smartcomms_workbench.diff.masking import BoundingBoxMaskRule
from smartcomms_workbench.diff.visual import ComparisonOptions, ComparisonOutcome, compare_pdf_bytes


def test_ssim_masks_exclude_influenced_windows_and_do_not_mutate_inputs() -> None:
    left = np.full((30, 30, 3), 255, dtype=np.uint8)
    right = left.copy()
    right[10:15, 10:15] = 0
    assert calculate_structural_similarity(left, left) == 1.0
    score = calculate_structural_similarity(left, right)
    assert score is not None and score < 1.0
    assert calculate_structural_similarity(left, right, masks=[(10, 10, 15, 15)]) == 1.0
    assert np.all(left == 255) and np.all(right[10:15, 10:15] == 0)


def test_ssim_without_valid_windows_is_inconclusive() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert calculate_structural_similarity(image, image, masks=[(0, 0, 10, 10)]) is None
    assert calculate_structural_similarity(image[:2], image[:2]) is None
    assert calculate_structural_similarity(image[:3, :3], image[:3, :3]) == 1


def test_pdf_ssim_is_opt_in_and_threshold_is_inclusive(pdf_factory: Callable[[str], bytes]) -> None:
    left, right = pdf_factory("one"), pdf_factory("two")
    unconfigured = compare_pdf_bytes(left, right, options=ComparisonOptions(max_changed_pixel_ratio=1))
    assert unconfigured.outcome == ComparisonOutcome.MATCH and unconfigured.pages[0].ssim is None
    measured = compare_pdf_bytes(left, right, options=ComparisonOptions(max_changed_pixel_ratio=1, min_ssim=-1))
    score = measured.pages[0].ssim
    assert score is not None and score < 1
    assert (
        compare_pdf_bytes(left, right, options=ComparisonOptions(max_changed_pixel_ratio=1, min_ssim=score)).outcome
        == ComparisonOutcome.MATCH
    )
    assert (
        compare_pdf_bytes(
            left, right, options=ComparisonOptions(max_changed_pixel_ratio=1, min_ssim=nextafter(score, 1))
        ).outcome
        == ComparisonOutcome.VISUAL_MISMATCH
    )
    assert (
        compare_pdf_bytes(
            left, right, options=ComparisonOptions(min_ssim=1, masks=(BoundingBoxMaskRule((0, 0, 200, 100)),))
        ).outcome
        == ComparisonOutcome.NO_COMPARABLE_PIXELS
    )


@pytest.mark.parametrize("threshold", [-2, 2, float("nan"), float("inf"), True])
def test_ssim_threshold_validation(threshold: float) -> None:
    with pytest.raises(ValueError, match="min_ssim"):
        ComparisonOptions(min_ssim=threshold)

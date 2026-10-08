"""Direct tests for pure RGB metrics and point-to-pixel conversion."""

from dataclasses import FrozenInstanceError
from math import inf, nan, nextafter
from typing import Any

import numpy as np
import pytest

from smartcomms_workbench.diff.image import ImageMetrics, bbox_to_pixels, calculate_image_metrics, ratio_is_match


def test_exact_metrics_and_channel_tolerance() -> None:
    reference = np.zeros((1, 3, 3), dtype=np.uint8)
    candidate = np.array([[[0, 0, 0], [3, 4, 5], [10, 0, 0]]], dtype=np.uint8)
    assert calculate_image_metrics(reference, candidate) == ImageMetrics(3, 2, 2 / 3, 22 / 9)
    assert calculate_image_metrics(reference, candidate, channel_tolerance=5) == ImageMetrics(3, 1, 1 / 3, 22 / 9)
    assert calculate_image_metrics(reference, candidate, channel_tolerance=255) == ImageMetrics(3, 0, 0.0, 22 / 9)
    assert calculate_image_metrics(reference, reference) == ImageMetrics(3, 0, 0.0, 0.0)


def test_uint8_subtraction_does_not_overflow() -> None:
    reference = np.array([[[0, 255, 0]]], dtype=np.uint8)
    candidate = np.array([[[255, 0, 255]]], dtype=np.uint8)
    assert calculate_image_metrics(reference, candidate) == ImageMetrics(1, 1, 1.0, 255.0)
    assert calculate_image_metrics(candidate, reference) == ImageMetrics(1, 1, 1.0, 255.0)


def test_union_masks_overlap_and_zero_area() -> None:
    reference = np.zeros((2, 3, 3), dtype=np.uint8)
    candidate = np.full((2, 3, 3), 255, dtype=np.uint8)
    masks = [(0, 0, 2, 1), (1, 0, 3, 1), (0, 0, 2, 1), (2, 1, 2, 2), (0, 2, 3, 2)]
    assert calculate_image_metrics(reference, candidate, masks=masks) == ImageMetrics(3, 3, 1.0, 255.0)


def test_masks_exclude_error_and_changes() -> None:
    reference = np.zeros((1, 2, 3), dtype=np.uint8)
    candidate = np.array([[[255, 255, 255], [3, 0, 0]]], dtype=np.uint8)
    assert calculate_image_metrics(reference, candidate, masks=[(0, 0, 1, 1)]) == ImageMetrics(1, 1, 1.0, 1.0)


def test_all_masked_is_inconclusive() -> None:
    reference = np.zeros((2, 2, 3), dtype=np.uint8)
    candidate = np.full_like(reference, 255)
    assert calculate_image_metrics(reference, candidate, masks=[(0, 0, 1, 2), (1, 0, 2, 2)]) == ImageMetrics(
        0, 0, None, None
    )


def test_inputs_preserved_including_noncontiguous_readonly_views() -> None:
    reference = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)[:, ::2]
    candidate = reference[::-1]
    before_reference = reference.copy()
    before_candidate = candidate.copy()
    reference.flags.writeable = False
    candidate.flags.writeable = False
    masks = [(0, 0, 1, 2)]
    calculate_image_metrics(reference, candidate, masks=masks)
    np.testing.assert_array_equal(reference, before_reference)
    np.testing.assert_array_equal(candidate, before_candidate)
    assert masks == [(0, 0, 1, 2)]


@pytest.mark.parametrize(
    "attribute", ["comparable_pixels", "changed_pixels", "changed_pixel_ratio", "mean_absolute_error"]
)
def test_results_are_immutable(attribute: str) -> None:
    result = ImageMetrics(1, 0, 0.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        setattr(result, attribute, 1)


@pytest.mark.parametrize(
    ("bbox", "dpi", "expected"),
    [
        ((0.25, 1.25, 2.25, 3.25), 72, (0, 1, 3, 4)),
        ((0.25, 1.25, 2.25, 3.25), 144, (0, 2, 5, 7)),
        ((-2.5, -3.5, 20.5, 30.5), 72, (0, 0, 10, 8)),
        ((-3.0, -4.0, -1.0, -2.0), 72, (0, 0, 0, 0)),
        ((11.0, 9.0, 12.0, 10.0), 72, (10, 8, 10, 8)),
        ((1.0, 2.0, 3.0, 4.0), 72, (1, 2, 3, 4)),
        ((-1e308, -1e308, 1e308, 1e308), 144, (0, 0, 10, 8)),
    ],
)
def test_bbox_rounding_and_clipping(
    bbox: tuple[float, float, float, float], dpi: int, expected: tuple[int, int, int, int]
) -> None:
    assert bbox_to_pixels(bbox, dpi=dpi, width=10, height=8) == expected


@pytest.mark.parametrize("name", ["dpi", "width", "height"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_bbox_dimensions(name: str, value: Any) -> None:
    kwargs = {"dpi": 72, "width": 10, "height": 8}
    kwargs[name] = value
    with pytest.raises(ValueError):
        bbox_to_pixels((0.0, 0.0, 1.0, 1.0), **kwargs)


@pytest.mark.parametrize(
    "bbox",
    [
        (0, 0, 0, 1),
        (0, 2, 1, 1),
        (2, 0, 1, 1),
        (0, 0, 1, 0),
        (nan, 0, 1, 1),
        (0, inf, 1, 2),
        (0, 0, -inf, 1),
        (0, 0, 1, nan),
        (0, 0, 1),
        (False, 0, 1, 1),
        ("0", 0, 1, 1),
    ],
)
def test_invalid_bbox_coordinates(bbox: Any) -> None:
    with pytest.raises(ValueError):
        bbox_to_pixels(bbox, dpi=72, width=10, height=8)


@pytest.mark.parametrize("shape", [(2, 2), (2, 2, 1), (2, 2, 4), (0, 2, 3), (2, 0, 3), (1, 2, 2, 3)])
def test_invalid_image_shapes(shape: tuple[int, ...]) -> None:
    invalid = np.zeros(shape, dtype=np.uint8)
    valid = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        calculate_image_metrics(invalid, valid)
    with pytest.raises(ValueError):
        calculate_image_metrics(valid, invalid)


def test_unequal_image_shapes() -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        calculate_image_metrics(np.zeros((1, 2, 3), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.uint8))


@pytest.mark.parametrize("dtype", [np.float32, np.int16, np.bool_])
def test_invalid_image_dtypes(dtype: Any) -> None:
    invalid = np.zeros((1, 1, 3), dtype=dtype)
    valid = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="uint8"):
        calculate_image_metrics(invalid, valid)
    with pytest.raises(ValueError, match="uint8"):
        calculate_image_metrics(valid, invalid)


@pytest.mark.parametrize("value", [nan, inf, -inf])
def test_nonfinite_images_rejected(value: float) -> None:
    invalid: Any = np.full((1, 1, 3), value)
    valid = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="uint8"):
        calculate_image_metrics(invalid, valid)


def test_masks_are_validated_even_when_all_pixels_are_excluded() -> None:
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        calculate_image_metrics(image, image, masks=[(0, 0, 1, 1), (0, 0, 2, 1)])


@pytest.mark.parametrize("tolerance", [-1, 256, True, 0.5, nan, inf, "0"])
def test_invalid_channel_tolerances(tolerance: Any) -> None:
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        calculate_image_metrics(image, image, channel_tolerance=tolerance)


@pytest.mark.parametrize(
    "rect",
    [
        (-1, 0, 1, 1),
        (0, 0, 3, 1),
        (0, -1, 1, 1),
        (0, 0, 1, 3),
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        (0.0, 0, 1, 1),
        (False, 0, 1, 1),
        (0, 0, nan, 1),
        (0, 0, 1, inf),
        (0, 0, 1),
    ],
)
def test_invalid_mask_rectangles(rect: Any) -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        calculate_image_metrics(image, image, masks=[rect])


def test_ratio_match_inclusive_boundary_and_next_ratio() -> None:
    assert ratio_is_match(1, 4, 0.25)
    assert not ratio_is_match(2, 4, 0.25)
    assert not ratio_is_match(1, 4, nextafter(0.25, 0.0))
    assert ratio_is_match(0, 4, 0.0)
    assert ratio_is_match(4, 4, 1.0)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, nan, inf, -inf, True, "0.5"])
def test_invalid_ratio_threshold(threshold: Any) -> None:
    with pytest.raises(ValueError):
        ratio_is_match(1, 2, threshold)


@pytest.mark.parametrize(
    ("changed", "comparable"), [(0, 0), (0, -1), (-1, 2), (3, 2), (True, 2), (0, True), (0.5, 2), (0, 2.0)]
)
def test_invalid_ratio_counts(changed: Any, comparable: Any) -> None:
    with pytest.raises(ValueError):
        ratio_is_match(changed, comparable, 0.5)

"""Pure RGB image comparison using absolute channel differences, not blueprint SSIM.

Pixels change when any channel exceeds the channel tolerance. Mean absolute error
uses every unmasked RGB channel, including differences below that tolerance, and
is measured on the original 0..255 scale. Masks exclude pixels rather than paint
either input. Fully masked images have no conclusive ratio or error.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor, isfinite

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ImageMetrics:
    """An immutable comparison snapshot of pixels outside the union of masks."""

    comparable_pixels: int
    changed_pixels: int
    changed_pixel_ratio: float | None
    mean_absolute_error: float | None


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def bbox_to_pixels(
    bbox: tuple[float, float, float, float], *, dpi: int, width: int, height: int
) -> tuple[int, int, int, int]:
    """Convert a positive-area point bbox to a clipped, half-open pixel rectangle.

    Coordinates may be negative but must be finite and ordered. Lower bounds
    round down and upper bounds round up at dpi/72 pixels per point. Clipping can
    produce an empty rectangle, which excludes no pixels.
    """
    _positive_integer(dpi, "dpi")
    _positive_integer(width, "width")
    _positive_integer(height, "height")
    if len(bbox) != 4 or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) for value in bbox
    ):
        raise ValueError("bbox must contain four finite point coordinates")
    x0, y0, x1, y1 = bbox
    if x0 >= x1 or y0 >= y1:
        raise ValueError("bbox must have positive area and ordered coordinates")
    scale = dpi / 72
    # Clamp before rounding to avoid overflow for finite coordinates far off-page.
    return (
        floor(min(width, max(0, x0 * scale))),
        floor(min(height, max(0, y0 * scale))),
        ceil(min(width, max(0, x1 * scale))),
        ceil(min(height, max(0, y1 * scale))),
    )


def calculate_image_metrics(
    reference: NDArray[np.uint8],
    candidate: NDArray[np.uint8],
    *,
    masks: Sequence[tuple[int, int, int, int]] = (),
    channel_tolerance: int = 0,
) -> ImageMetrics:
    """Compare equal-sized, nonempty HxWx3 uint8 images without mutating them.

    Mask rectangles are half-open, ordered integer coordinates within the image.
    Overlaps are excluded only once, and zero-area masks have no effect. No
    resizing, SSIM calculation, file access, or painted image copies occur.
    """
    if (
        isinstance(channel_tolerance, bool)
        or not isinstance(channel_tolerance, int)
        or not 0 <= channel_tolerance <= 255
    ):
        raise ValueError("channel_tolerance must be an integer from 0 to 255")
    for image in (reference, candidate):
        if not isinstance(image, np.ndarray):
            raise TypeError("images must be NumPy arrays")
        if image.dtype != np.uint8:
            raise ValueError("images must have uint8 dtype")
        if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("images must have nonempty HxWx3 shapes")
    if reference.shape != candidate.shape:
        raise ValueError("images must have equal shapes")
    height, width, _ = reference.shape
    comparable = np.ones((height, width), dtype=np.bool_)
    for rect in masks:
        if len(rect) != 4 or any(isinstance(value, bool) or not isinstance(value, int) for value in rect):
            raise ValueError("mask rectangles must contain four integers")
        x0, y0, x1, y1 = rect
        if not (0 <= x0 <= x1 <= width and 0 <= y0 <= y1 <= height):
            raise ValueError("mask rectangles must be ordered and within image extents")
        comparable[y0:y1, x0:x1] = False
    comparable_pixels = int(np.count_nonzero(comparable))
    if comparable_pixels == 0:
        return ImageMetrics(0, 0, None, None)

    delta = np.subtract(reference, candidate, dtype=np.int16)
    np.abs(delta, out=delta)
    changed = np.max(delta, axis=2) > channel_tolerance
    np.logical_and(changed, comparable, out=changed)
    changed_pixels = int(np.count_nonzero(changed))
    total_error = int(np.sum(delta, where=comparable[..., None], dtype=np.int64))
    return ImageMetrics(
        comparable_pixels,
        changed_pixels,
        changed_pixels / comparable_pixels,
        total_error / (3 * comparable_pixels),
    )


def ratio_is_match(changed_pixels: int, comparable_pixels: int, threshold: float) -> bool:
    """Test the changed-pixel ratio against an inclusive finite 0..1 threshold.

    A zero denominator is inconclusive and must be handled by orchestration,
    rather than classified as a match here.
    """
    _positive_integer(comparable_pixels, "comparable_pixels")
    if (
        isinstance(changed_pixels, bool)
        or not isinstance(changed_pixels, int)
        or not 0 <= changed_pixels <= comparable_pixels
    ):
        raise ValueError("changed_pixels must be an integer between zero and comparable_pixels")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("threshold must be finite and between zero and one")
    return changed_pixels / comparable_pixels <= threshold

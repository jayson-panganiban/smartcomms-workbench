"""Visual comparison of supplied PDF bytes, without filesystem or network I/O.

Pages are compared independently at 144 DPI by default. Masks from both sides
are unioned in effective, cropped/rotated page coordinates before metrics are
calculated. Regex masks cover entire extracted text blocks, not characters.
Pixels exceeding the RGB channel tolerance count as changed; each page passes
when its unmasked changed-pixel ratio is <= the configured threshold.

Page-count mismatch precedes page-size mismatch, which precedes visual metrics.
An unmasked visual mismatch takes precedence over an entirely masked page;
otherwise any entirely masked page makes the document inconclusive. No images,
PDF buffers, reports, or pipeline state are retained in the returned result.
"""

from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Literal

import pymupdf

from smartcomms_workbench.diff.image import (
    ImageMetrics,
    bbox_to_pixels,
    calculate_image_metrics,
    ratio_is_match,
)
from smartcomms_workbench.diff.masking import (
    BoundingBoxMaskRule,
    MaskRule,
    PageTextBlocks,
    RegexMaskRule,
    TextBlock,
    find_mask_rectangles,
)
from smartcomms_workbench.diff.pdf import (
    PageInfo,
    PdfInputError,
    extract_text_blocks,
    open_pdf,
    page_info,
    render_page,
)


class ComparisonOutcome(StrEnum):
    """Expected comparison and invalid-document outcomes, not programming bugs."""

    MATCH = "match"
    VISUAL_MISMATCH = "visual_mismatch"
    PAGE_COUNT_MISMATCH = "page_count_mismatch"
    PAGE_SIZE_MISMATCH = "page_size_mismatch"
    INVALID_PDF = "invalid_pdf"
    NO_COMPARABLE_PIXELS = "no_comparable_pixels"


@dataclass(frozen=True)
class ComparisonOptions:
    """Immutable configuration; invalid configuration raises, never becomes a match.

    Mask page selectors are one-based. Bboxes are top-left PDF points.
    Page size tolerance is inclusive and does not permit resizing or padding.
    Channel tolerance is an integer in 0..255; ratio threshold is in 0..1.
    """

    dpi: int = 144
    masks: tuple[MaskRule, ...] = ()
    channel_tolerance: int = 0
    max_changed_pixel_ratio: float = 0.0
    page_size_tolerance: float = 0.01

    def __post_init__(self) -> None:
        if isinstance(self.dpi, bool) or not isinstance(self.dpi, int):
            raise TypeError("dpi must be an integer")
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")
        if isinstance(self.channel_tolerance, bool) or not isinstance(self.channel_tolerance, int):
            raise TypeError("channel_tolerance must be an integer")
        if not 0 <= self.channel_tolerance <= 255:
            raise ValueError("channel_tolerance must be in 0..255")
        if (
            isinstance(self.max_changed_pixel_ratio, bool)
            or not isinstance(self.max_changed_pixel_ratio, (int, float))
            or not isfinite(self.max_changed_pixel_ratio)
            or not 0 <= self.max_changed_pixel_ratio <= 1
        ):
            raise ValueError("max_changed_pixel_ratio must be finite and in 0..1")
        if (
            isinstance(self.page_size_tolerance, bool)
            or not isinstance(self.page_size_tolerance, (int, float))
            or not isfinite(self.page_size_tolerance)
            or self.page_size_tolerance < 0
        ):
            raise ValueError("page_size_tolerance must be finite and nonnegative")
        if not isinstance(self.masks, tuple):
            raise TypeError("masks must be an immutable tuple of mask rules")
        if any(not isinstance(rule, RegexMaskRule | BoundingBoxMaskRule) for rule in self.masks):
            raise TypeError("masks must contain only MaskRule values")
        # Validate regex syntax even for documents with no text or incompatible pages.
        find_mask_rectangles((), list(self.masks))


@dataclass(frozen=True)
class PageComparison:
    """One-based page outcome and scalar metrics; pixel sizes are (width, height).

    Pixel dimensions/metrics are absent for a pre-render page-size mismatch.
    Metrics also remain absent when raster dimensions cannot be compared.
    """

    page: int
    outcome: ComparisonOutcome
    reference_size: tuple[float, float]
    candidate_size: tuple[float, float]
    reference_pixels: tuple[int, int] | None = None
    candidate_pixels: tuple[int, int] | None = None
    metrics: ImageMetrics | None = None


@dataclass(frozen=True)
class ComparisonResult:
    """Typed document outcome with immutable per-page results.

    Counts are available for successfully opened documents. INVALID_PDF names
    the input side and error; it has no misleading comparison metrics.
    PAGE_COUNT_MISMATCH has no paired-page results. Pre-render size mismatch
    returns the mismatched pages only; normal comparison returns every page.
    """

    outcome: ComparisonOutcome
    pages: tuple[PageComparison, ...] = ()
    reference_page_count: int | None = None
    candidate_page_count: int | None = None
    invalid_side: Literal["reference", "candidate"] | None = None
    error: str | None = None


def _size(info: PageInfo) -> tuple[float, float]:
    return info.width, info.height


def _page_masks(page: pymupdf.Page, number: int, rules: list[MaskRule]) -> list[tuple[float, float, float, float]]:
    blocks = (
        tuple(TextBlock(block.text, block.bbox) for block in extract_text_blocks(page))
        if any(isinstance(rule, RegexMaskRule) and (rule.pages is None or number in rule.pages) for rule in rules)
        else ()
    )
    return [rect.bbox for rect in find_mask_rectangles((PageTextBlocks(number, blocks),), rules)]


def compare_pdf_bytes(
    reference_pdf: bytes,
    candidate_pdf: bytes,
    *,
    options: ComparisonOptions = ComparisonOptions(),
) -> ComparisonResult:
    """Compare in-memory PDFs; only PdfInputError becomes INVALID_PDF.

    Repaired but readable PDFs can be compared. All other programming/library
    errors propagate. PDFs and pixmaps are processed page-by-page and closed
    deterministically; the core never writes artifacts or reads paths.
    """
    if not isinstance(reference_pdf, bytes) or not isinstance(candidate_pdf, bytes):
        raise TypeError("PDF inputs must be bytes")
    if not isinstance(options, ComparisonOptions):
        raise TypeError("options must be ComparisonOptions")
    reference_count: int | None = None
    candidate_count: int | None = None
    side: Literal["reference", "candidate"] = "reference"
    with ExitStack() as resources:
        try:
            reference = open_pdf(reference_pdf)
            resources.callback(reference.close)
            reference_count = len(reference)
            side = "candidate"
            candidate = open_pdf(candidate_pdf)
            resources.callback(candidate.close)
            candidate_count = len(candidate)
            if reference_count != candidate_count:
                return ComparisonResult(
                    ComparisonOutcome.PAGE_COUNT_MISMATCH,
                    reference_page_count=reference_count,
                    candidate_page_count=candidate_count,
                )

            sizes: list[tuple[PageInfo, PageInfo]] = []
            size_mismatches: list[PageComparison] = []
            for index in range(reference_count):
                side = "reference"
                ref_info = page_info(reference, index)
                side = "candidate"
                cand_info = page_info(candidate, index)
                sizes.append((ref_info, cand_info))
                if (
                    abs(ref_info.width - cand_info.width) > options.page_size_tolerance
                    or abs(ref_info.height - cand_info.height) > options.page_size_tolerance
                ):
                    size_mismatches.append(
                        PageComparison(
                            index + 1, ComparisonOutcome.PAGE_SIZE_MISMATCH, _size(ref_info), _size(cand_info)
                        )
                    )
            if size_mismatches:
                return ComparisonResult(
                    ComparisonOutcome.PAGE_SIZE_MISMATCH, tuple(size_mismatches), reference_count, candidate_count
                )

            pages: list[PageComparison] = []
            rules = list(options.masks)
            for index, (ref_info, cand_info) in enumerate(sizes):
                side = "reference"
                ref_page = reference[index]
                reference_image = render_page(ref_page, dpi=options.dpi)
                rectangles = _page_masks(ref_page, index + 1, rules)
                side = "candidate"
                cand_page = candidate[index]
                candidate_image = render_page(cand_page, dpi=options.dpi)
                rectangles.extend(_page_masks(cand_page, index + 1, rules))
                ref_pixels = (reference_image.shape[1], reference_image.shape[0])
                cand_pixels = (candidate_image.shape[1], candidate_image.shape[0])
                if ref_pixels != cand_pixels:
                    pages.append(
                        PageComparison(
                            index + 1,
                            ComparisonOutcome.PAGE_SIZE_MISMATCH,
                            _size(ref_info),
                            _size(cand_info),
                            ref_pixels,
                            cand_pixels,
                        )
                    )
                    del reference_image, candidate_image
                    continue
                masks = [
                    bbox_to_pixels(bbox, dpi=options.dpi, width=ref_pixels[0], height=ref_pixels[1])
                    for bbox in rectangles
                ]
                metrics = calculate_image_metrics(
                    reference_image, candidate_image, masks=masks, channel_tolerance=options.channel_tolerance
                )
                del reference_image, candidate_image
                if metrics.comparable_pixels == 0:
                    outcome = ComparisonOutcome.NO_COMPARABLE_PIXELS
                elif ratio_is_match(metrics.changed_pixels, metrics.comparable_pixels, options.max_changed_pixel_ratio):
                    outcome = ComparisonOutcome.MATCH
                else:
                    outcome = ComparisonOutcome.VISUAL_MISMATCH
                pages.append(
                    PageComparison(
                        index + 1, outcome, _size(ref_info), _size(cand_info), ref_pixels, cand_pixels, metrics
                    )
                )
            outcome = ComparisonOutcome.MATCH
            for priority in (
                ComparisonOutcome.PAGE_SIZE_MISMATCH,
                ComparisonOutcome.VISUAL_MISMATCH,
                ComparisonOutcome.NO_COMPARABLE_PIXELS,
            ):
                if any(page.outcome == priority for page in pages):
                    outcome = priority
                    break
            return ComparisonResult(outcome, tuple(pages), reference_count, candidate_count)
        except PdfInputError as error:
            return ComparisonResult(
                ComparisonOutcome.INVALID_PDF,
                reference_page_count=reference_count,
                candidate_page_count=candidate_count,
                invalid_side=side,
                error=str(error),
            )

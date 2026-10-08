"""Pure rule evaluation for selecting PDF-point rectangles to mask."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

BBox = tuple[float, float, float, float]


def _validate_pages(pages: tuple[int, ...] | None) -> None:
    if pages is None:
        return
    if not isinstance(pages, tuple):
        raise TypeError("pages must be a tuple of positive integers or None")
    for page in pages:
        _validate_page(page)


def _validate_page(page: int) -> None:
    if isinstance(page, bool) or not isinstance(page, int):
        raise TypeError("page numbers must be integers")
    if page <= 0:
        raise ValueError("page numbers must be positive")


def _validate_bbox(bbox: BBox) -> None:
    if not isinstance(bbox, tuple):
        raise TypeError("bbox must be a tuple of four finite numbers")
    if len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        raise TypeError("bbox coordinates must be numbers")
    try:
        coordinates_are_finite = all(math.isfinite(value) for value in bbox)
    except OverflowError:
        coordinates_are_finite = False
    if not coordinates_are_finite:
        raise ValueError("bbox coordinates must be finite")
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox must have positive area")


@dataclass(frozen=True)
class RegexMaskRule:
    """Mask the complete bbox of each text block matching ``pattern``.

    Patterns use Python regular-expression syntax, including inline flags.
    ``pages=None`` selects every supplied page; an empty tuple selects none.
    Page numbers are one-based.
    """

    pattern: str
    pages: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, str):
            raise TypeError("pattern must be a string")
        _validate_pages(self.pages)


@dataclass(frozen=True)
class BoundingBoxMaskRule:
    """Mask ``bbox`` on selected supplied pages using PDF-point coordinates.

    Coordinates use a top-left origin. Negative coordinates are allowed so
    callers can clip rectangles to page bounds later.
    """

    bbox: BBox
    pages: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _validate_bbox(self.bbox)
        _validate_pages(self.pages)


MaskRule = RegexMaskRule | BoundingBoxMaskRule


@dataclass(frozen=True)
class TextBlock:
    """Text and its coarse PDF-point bounding box.

    Regex masking returns the whole block box, not character-level geometry.
    """

    text: str
    bbox: BBox

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        _validate_bbox(self.bbox)


@dataclass(frozen=True)
class PageTextBlocks:
    """Text blocks supplied for a one-based page; blocks may be empty."""

    page: int
    blocks: tuple[TextBlock, ...]

    def __post_init__(self) -> None:
        _validate_page(self.page)
        if not isinstance(self.blocks, tuple):
            raise TypeError("blocks must be a tuple of TextBlock values")
        if any(not isinstance(block, TextBlock) for block in self.blocks):
            raise TypeError("blocks must contain only TextBlock values")


@dataclass(frozen=True)
class Rect:
    """A PDF-point masking rectangle associated with a one-based page."""

    page: int
    bbox: BBox

    def __post_init__(self) -> None:
        _validate_page(self.page)
        _validate_bbox(self.bbox)


def find_mask_rectangles(text_blocks: Sequence[PageTextBlocks], rules: list[MaskRule]) -> list[Rect]:
    """Return unique mask rectangles in page, rule, and block input order.

    ``text_blocks`` contains only supplied pages, including any intentionally
    blank pages. A rule with ``pages=None`` selects all supplied pages;
    ``pages=()`` selects none, and selected pages absent from the input are
    ignored. Regexes search each block independently and mask its complete
    bbox once per matching block. Bounding-box rules emit their bbox once per
    selected supplied page. Exact duplicate rectangles are removed globally;
    overlapping rectangles are not merged.
    """
    if not isinstance(text_blocks, Sequence) or isinstance(text_blocks, (str, bytes)):
        raise TypeError("text_blocks must be a sequence of PageTextBlocks")
    if not isinstance(rules, list):
        raise TypeError("rules must be a list of mask rules")

    seen_pages: set[int] = set()
    for page_group in text_blocks:
        if not isinstance(page_group, PageTextBlocks):
            raise TypeError("text_blocks must contain only PageTextBlocks values")
        if page_group.page in seen_pages:
            raise ValueError(f"duplicate page group: {page_group.page}")
        seen_pages.add(page_group.page)

    compiled_rules: list[re.Pattern[str] | None] = []
    for rule in rules:
        if isinstance(rule, RegexMaskRule):
            compiled_rules.append(re.compile(rule.pattern))
        elif isinstance(rule, BoundingBoxMaskRule):
            compiled_rules.append(None)
        else:
            raise TypeError("rules must contain only MaskRule values")

    rectangles: list[Rect] = []
    seen_rectangles: set[Rect] = set()
    for page_group in text_blocks:
        for rule, pattern in zip(rules, compiled_rules, strict=True):
            if rule.pages is not None and page_group.page not in rule.pages:
                continue
            if isinstance(rule, BoundingBoxMaskRule):
                candidates = (Rect(page_group.page, rule.bbox),)
            else:
                assert isinstance(rule, RegexMaskRule)
                assert pattern is not None
                candidates = (
                    Rect(page_group.page, block.bbox) for block in page_group.blocks if pattern.search(block.text)
                )
            for rectangle in candidates:
                if rectangle not in seen_rectangles:
                    seen_rectangles.add(rectangle)
                    rectangles.append(rectangle)

    return rectangles

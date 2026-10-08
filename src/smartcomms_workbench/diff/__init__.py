"""Pure document-diff calculation helpers."""

from smartcomms_workbench.diff.masking import (
    BBox,
    BoundingBoxMaskRule,
    MaskRule,
    PageTextBlocks,
    Rect,
    RegexMaskRule,
    TextBlock,
    find_mask_rectangles,
)

__all__ = [
    "BBox",
    "BoundingBoxMaskRule",
    "MaskRule",
    "PageTextBlocks",
    "Rect",
    "RegexMaskRule",
    "TextBlock",
    "find_mask_rectangles",
]

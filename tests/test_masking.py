import re
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from smartcomms_workbench.diff import (
    BoundingBoxMaskRule,
    MaskRule,
    PageTextBlocks,
    Rect,
    RegexMaskRule,
    TextBlock,
    find_mask_rectangles,
)


def _assign_attribute(instance: object, name: str, value: object) -> None:
    setattr(instance, name, value)


def _construct(model: Any, **values: Any) -> Any:
    return model(**values)


def test_models_are_frozen_and_inputs_are_not_mutated() -> None:
    block = TextBlock("account 123", (1.0, 2.0, 3.0, 4.0))
    page = PageTextBlocks(1, (block,))
    rule = RegexMaskRule(r"\d+")
    groups = [page]
    rules: list[MaskRule] = [rule]

    assert find_mask_rectangles(groups, rules) == [Rect(1, block.bbox)]
    assert groups == [page]
    assert rules == [rule]
    with pytest.raises(FrozenInstanceError):
        _assign_attribute(block, "text", "changed")
    with pytest.raises(FrozenInstanceError):
        _assign_attribute(page, "page", 2)
    with pytest.raises(FrozenInstanceError):
        _assign_attribute(rule, "pattern", "changed")


def test_page_selectors_include_all_none_empty_and_ignore_absent_pages() -> None:
    groups = [
        PageTextBlocks(3, (TextBlock("match", (0.0, 0.0, 1.0, 1.0)),)),
        PageTextBlocks(1, (TextBlock("match", (2.0, 2.0, 3.0, 3.0)),)),
        PageTextBlocks(7, ()),
    ]
    rules = [
        BoundingBoxMaskRule((10.0, 10.0, 11.0, 11.0)),
        RegexMaskRule("match", pages=()),
        BoundingBoxMaskRule((20.0, 20.0, 21.0, 21.0), pages=(1, 2, 7)),
    ]

    assert find_mask_rectangles(groups, rules) == [
        Rect(3, (10.0, 10.0, 11.0, 11.0)),
        Rect(1, (10.0, 10.0, 11.0, 11.0)),
        Rect(1, (20.0, 20.0, 21.0, 21.0)),
        Rect(7, (10.0, 10.0, 11.0, 11.0)),
        Rect(7, (20.0, 20.0, 21.0, 21.0)),
    ]


def test_regex_search_flags_and_multiple_occurrences_mask_whole_block_once() -> None:
    blocks = (
        TextBlock("Reference: abc-1 and abc-2", (1.0, 2.0, 5.0, 6.0)),
        TextBlock("lowercase abc-3", (7.0, 8.0, 9.0, 10.0)),
    )
    groups = [PageTextBlocks(1, blocks)]

    assert find_mask_rectangles(groups, [RegexMaskRule("(?i)ABC-\\d")]) == [
        Rect(1, blocks[0].bbox),
        Rect(1, blocks[1].bbox),
    ]


def test_regex_does_not_match_across_blocks() -> None:
    groups = [
        PageTextBlocks(
            1,
            (
                TextBlock("secret", (0.0, 0.0, 1.0, 1.0)),
                TextBlock("value", (1.0, 0.0, 2.0, 1.0)),
            ),
        )
    ]

    assert find_mask_rectangles(groups, [RegexMaskRule("secret value")]) == []


def test_blank_pages_and_empty_inputs_are_valid() -> None:
    assert find_mask_rectangles([PageTextBlocks(4, ())], [BoundingBoxMaskRule((-1.0, -2.0, 1.0, 2.0))]) == [
        Rect(4, (-1.0, -2.0, 1.0, 2.0))
    ]
    assert find_mask_rectangles([], [RegexMaskRule("valid")]) == []


def test_invalid_bboxes_are_rejected() -> None:
    for bbox in (
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 1.0, 1.0),
        (0.0, 0.0, float("inf"), 1.0),
        (0.0, 0.0, float("nan"), 1.0),
        (0.0, 0.0, 10**1000, 1.0),
    ):
        with pytest.raises(ValueError):
            BoundingBoxMaskRule(bbox)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TextBlock("text", (1.0, 1.0, 0.0, 2.0))
    with pytest.raises(TypeError):
        _construct(Rect, page=1, bbox=[0.0, 0.0, 1.0, 1.0])


@pytest.mark.parametrize("page", [0, -1])
def test_nonpositive_page_numbers_are_rejected(page: int) -> None:
    with pytest.raises(ValueError):
        PageTextBlocks(page, ())
    with pytest.raises(ValueError):
        RegexMaskRule("x", pages=(page,))


@pytest.mark.parametrize("page", [True, 1.0, "1"])
def test_page_numbers_must_be_integers_but_not_booleans(page: object) -> None:
    with pytest.raises(TypeError):
        _construct(PageTextBlocks, page=page, blocks=())
    with pytest.raises(TypeError):
        _construct(RegexMaskRule, pattern="x", pages=(page,))


def test_duplicate_page_groups_are_invalid() -> None:
    with pytest.raises(ValueError, match="duplicate page group"):
        find_mask_rectangles([PageTextBlocks(1, ()), PageTextBlocks(1, ())], [])


def test_exact_duplicates_are_removed_but_overlaps_remain_in_stable_order() -> None:
    same_box = (0.0, 0.0, 2.0, 2.0)
    overlapping_box = (1.0, 1.0, 3.0, 3.0)
    groups = [
        PageTextBlocks(
            2,
            (
                TextBlock("hit", same_box),
                TextBlock("hit", same_box),
                TextBlock("hit", overlapping_box),
            ),
        )
    ]
    rules = [RegexMaskRule("hit"), BoundingBoxMaskRule(same_box)]

    assert find_mask_rectangles(groups, rules) == [
        Rect(2, same_box),
        Rect(2, overlapping_box),
    ]


def test_invalid_regex_is_compiled_even_when_selector_matches_no_pages() -> None:
    with pytest.raises(re.PatternError):
        find_mask_rectangles([], [RegexMaskRule("[", pages=())])

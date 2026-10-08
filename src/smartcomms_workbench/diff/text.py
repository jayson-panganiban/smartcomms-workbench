"""Pure, whitespace-normalized token comparison with optional regex masking."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from smartcomms_workbench.diff.masking import RegexMaskRule
from smartcomms_workbench.diff.pdf import PdfInputError, extract_text_blocks, open_pdf


@dataclass(frozen=True)
class TokenChange:
    operation: Literal["replace", "delete", "insert"]
    reference: tuple[str, ...]
    candidate: tuple[str, ...]


@dataclass(frozen=True)
class TextComparison:
    outcome: Literal["match", "text_mismatch", "page_count_mismatch", "invalid_pdf"]
    changes: tuple[TokenChange, ...] = ()
    error: str | None = None


def compare_text(reference: str, candidate: str) -> TextComparison:
    """Compare tokens, ignoring whitespace but preserving case and punctuation."""
    left, right = tuple(reference.split()), tuple(candidate.split())
    changes: list[TokenChange] = []
    for operation, i, j, k, m in SequenceMatcher(a=left, b=right, autojunk=False).get_opcodes():
        if operation in ("replace", "delete", "insert"):
            changes.append(TokenChange(operation, left[i:j], right[k:m]))
    return TextComparison("text_mismatch" if changes else "match", tuple(changes))


def compare_pdf_text(reference: bytes, candidate: bytes, *, masks: Sequence[RegexMaskRule] = ()) -> TextComparison:
    """Compare corresponding PDF pages, applying one-based regex page selectors."""
    patterns = [(rule, re.compile(rule.pattern)) for rule in masks]
    try:
        with open_pdf(reference) as left, open_pdf(candidate) as right:
            if len(left) != len(right):
                return TextComparison("page_count_mismatch")
            changes: list[TokenChange] = []
            for index in range(len(left)):
                texts = [
                    "".join(block.text for block in extract_text_blocks(document[index])) for document in (left, right)
                ]
                for rule, pattern in patterns:
                    if rule.pages is None or index + 1 in rule.pages:
                        texts = [pattern.sub("", text) for text in texts]
                changes.extend(compare_text(*texts).changes)
            return TextComparison("text_mismatch" if changes else "match", tuple(changes))
    except PdfInputError as error:
        return TextComparison("invalid_pdf", error=str(error))

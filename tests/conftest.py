from collections.abc import Callable

import pymupdf
import pytest


@pytest.fixture
def pdf_factory() -> Callable[[str], bytes]:
    def make(text: str = "Hello") -> bytes:
        with pymupdf.open() as document:
            document.new_page(width=200, height=100).insert_text((20, 30), text)
            return document.tobytes(no_new_id=True)

    return make

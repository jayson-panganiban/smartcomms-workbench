"""Real PyMuPDF tests for the PDF adapter."""

from collections.abc import Generator
from contextlib import contextmanager
from gc import collect
from typing import cast
from weakref import ref

import numpy as np
import pymupdf
import pytest
from numpy.typing import NDArray

from smartcomms_workbench.diff.pdf import (
    ExtractedTextBlock,
    PageInfo,
    PdfInputError,
    extract_text_blocks,
    open_pdf,
    page_info,
    render_page,
)


def _pdf_bytes(
    *,
    width: float = 200,
    height: float = 100,
    text: str = "Hello PDF",
    origin: tuple[float, float] = (20, 30),
    cropbox: pymupdf.Rect | None = None,
    rotation: int = 0,
) -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=width, height=height)
    page.insert_text(origin, text)
    if cropbox is not None:
        page.set_cropbox(cropbox)
    if rotation:
        page.set_rotation(rotation)
    try:
        return document.tobytes(no_new_id=True)
    finally:
        document.close()


def _zero_page_pdf_bytes() -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [] /Count 0 >>",
    ]
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{number} 0 obj\n".encode("ascii"))
        content.extend(body)
        content.extend(b"\nendobj\n")
    xref_offset = len(content)
    content.extend(b"xref\n0 3\n0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010} 00000 n \n".encode("ascii"))
    content.extend(f"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii"))
    return bytes(content)


def _encrypted_pdf_bytes() -> bytes:
    document = pymupdf.open()
    document.new_page()
    try:
        return document.tobytes(
            encryption=cast(int, pymupdf.__dict__["PDF_ENCRYPT_AES_256"]),
            owner_pw="owner-password",
            user_pw="user-password",
            no_new_id=True,
        )
    finally:
        document.close()


@contextmanager
def _opened(pdf_bytes: bytes) -> Generator[pymupdf.Document]:
    with open_pdf(pdf_bytes) as document:
        yield document


def test_open_pdf_returns_caller_owned_document() -> None:
    data = _pdf_bytes()
    document = open_pdf(data)
    try:
        assert isinstance(document, pymupdf.Document)
        assert document.page_count == 1
    finally:
        document.close()


def test_render_page_returns_rgb_shape_white_background_and_deterministic_pixels() -> None:
    with _opened(_pdf_bytes()) as document:
        page = document[0]
        first = render_page(page)
        second = render_page(page, dpi=144)

    assert first.shape == (200, 400, 3)
    assert first.dtype == np.uint8
    assert first[0, 0].tolist() == [255, 255, 255]
    np.testing.assert_array_equal(first, second)


def test_render_page_array_keeps_pixmap_buffer_alive() -> None:
    with _opened(_pdf_bytes()) as document:
        image = render_page(document[0])
    assert image[0, 0].tolist() == [255, 255, 255]


def test_render_views_retain_pixmap_owner_without_copying() -> None:
    with _opened(_pdf_bytes()) as document:
        image = render_page(document[0])
        owner = ref(image.base)
        plain = np.asarray(image)
        sliced = plain[::2, ::2]
        assert np.shares_memory(image, plain)
        assert np.shares_memory(image, sliced)
    del image, plain
    collect()
    assert owner() is not None
    assert sliced[0, 0].tolist() == [255, 255, 255]
    del sliced
    collect()
    assert owner() is None


def test_extract_text_blocks_returns_text_and_top_left_bbox() -> None:
    with _opened(_pdf_bytes(text="Hello PDF", origin=(20, 30))) as document:
        blocks = extract_text_blocks(document[0])

    assert len(blocks) == 1
    assert isinstance(blocks[0], ExtractedTextBlock)
    assert blocks[0].text == "Hello PDF\n"
    assert blocks[0].bbox == pytest.approx((20.0, 18.174999, 70.116005, 33.289), abs=0.001)


def test_page_info_uses_one_based_index_and_effective_page_dimensions() -> None:
    data = _pdf_bytes(width=200, height=100, cropbox=pymupdf.Rect(20, 10, 180, 90), rotation=90)
    with _opened(data) as document:
        assert page_info(document, 0) == PageInfo(page=1, width=80, height=160)


def test_text_bbox_is_normalized_for_crop_and_rotation() -> None:
    data = _pdf_bytes(
        text="Crop rotate",
        origin=(40, 40),
        cropbox=pymupdf.Rect(20, 10, 180, 90),
        rotation=90,
    )
    with _opened(data) as document:
        blocks = extract_text_blocks(document[0])

    assert len(blocks) == 1
    assert blocks[0].text == "Crop rotate\n"
    x0, y0, x1, y1 = blocks[0].bbox
    assert (x0, y0, x1, y1) == pytest.approx((46.711, 20, 61.825, 75.022), abs=0.01)


@pytest.mark.parametrize("data", [b"", b"not a PDF", b"%PDF-1.7\nnot structurally readable"])
def test_open_pdf_rejects_empty_or_unreadable_bytes(data: bytes) -> None:
    with pytest.raises(PdfInputError):
        open_pdf(data)


def test_open_pdf_rejects_readable_non_pdf_input() -> None:
    with _opened(_pdf_bytes()) as document:
        png = document[0].get_pixmap().tobytes("png")
    with pytest.raises(PdfInputError):
        open_pdf(png)


def test_readable_repaired_pdf_can_be_compared() -> None:
    data = _pdf_bytes()
    damaged = data[: data.index(b"xref\n")]
    with _opened(damaged) as document:
        assert document.is_repaired
        assert page_info(document, 0) == PageInfo(page=1, width=200, height=100)
        actual = render_page(document[0])
    with _opened(data) as document:
        expected = render_page(document[0])
    np.testing.assert_array_equal(actual, expected)


def test_open_pdf_rejects_encrypted_and_zero_page_pdfs() -> None:
    for data in (_encrypted_pdf_bytes(), _zero_page_pdf_bytes()):
        with pytest.raises(PdfInputError):
            open_pdf(data)


def test_open_pdf_rejects_non_bytes_input() -> None:
    with pytest.raises(TypeError, match="pdf_bytes must be bytes"):
        open_pdf(cast(bytes, bytearray(b"%PDF")))


@pytest.mark.parametrize("dpi", [0, -1])
def test_render_page_rejects_nonpositive_dpi(dpi: int) -> None:
    with _opened(_pdf_bytes()) as document, pytest.raises(ValueError, match="dpi must be positive"):
        render_page(document[0], dpi=dpi)


@pytest.mark.parametrize("dpi", [144.0, True])
def test_render_page_rejects_noninteger_dpi(dpi: float | bool) -> None:
    with _opened(_pdf_bytes()) as document, pytest.raises(TypeError, match="dpi must be an integer"):
        render_page(document[0], dpi=cast(int, dpi))


def test_page_info_rejects_invalid_index_types_and_out_of_range_indices() -> None:
    with _opened(_pdf_bytes()) as document:
        with pytest.raises(TypeError, match="index must be an integer"):
            page_info(document, True)
        with pytest.raises(IndexError, match="page index out of range"):
            page_info(document, 1)


def test_render_page_is_typed_as_rgb_uint8_array() -> None:
    with _opened(_pdf_bytes()) as document:
        image: NDArray[np.uint8] = render_page(document[0])
    assert image.ndim == 3

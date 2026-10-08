"""In-memory PDF opening, page metadata, text extraction, and rendering."""

from dataclasses import dataclass

import numpy as np
import pymupdf
from numpy.typing import NDArray


class PdfInputError(ValueError):
    """A supplied PDF is empty, unreadable, encrypted, or has no pages."""


@dataclass(frozen=True, slots=True)
class PageInfo:
    """One-based page number and effective (cropped, rotated) dimensions in points."""

    page: int
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class ExtractedTextBlock:
    """Text and bounding box in top-left coordinates of the effective page."""

    text: str
    bbox: tuple[float, float, float, float]


class _PixmapBuffer:
    def __init__(self, pixmap: pymupdf.Pixmap) -> None:
        self._pixmap = pixmap
        self._samples = np.frombuffer(pixmap.samples_mv, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)

    @property
    def __array_interface__(self) -> dict[str, object]:
        # NumPy retains this owner as array.base, including after np.asarray/slicing.
        # samples_mv itself has no owning object to keep the pixmap alive.
        return self._samples.__array_interface__


def open_pdf(pdf_bytes: bytes) -> pymupdf.Document:
    """Open PDF bytes without file or network I/O.

    The caller owns the returned document and must close it, preferably with
    ``with open_pdf(data) as document:``.
    """
    if not isinstance(pdf_bytes, bytes):
        raise TypeError("pdf_bytes must be bytes")

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as error:
        raise PdfInputError("The supplied bytes are not a readable PDF") from error

    if document.needs_pass:
        document.close()
        raise PdfInputError("The supplied PDF requires a password")
    if not document.is_pdf:
        document.close()
        raise PdfInputError("The supplied document is not a PDF")
    if document.page_count == 0:
        document.close()
        raise PdfInputError("The supplied PDF contains no pages")
    return document


def page_info(document: pymupdf.Document, index: int) -> PageInfo:
    """Return effective page dimensions for a zero-based page index."""
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("index must be an integer")
    if index < 0 or index >= document.page_count:
        raise IndexError("page index out of range")
    page = document[index]
    return PageInfo(page=index + 1, width=page.rect.width, height=page.rect.height)


def extract_text_blocks(page: pymupdf.Page) -> tuple[ExtractedTextBlock, ...]:
    """Extract text blocks with bounding boxes rotated to the effective page."""
    try:
        raw_blocks = page.get_text("blocks")
    except pymupdf.FileDataError as error:
        raise PdfInputError("Could not extract text from the PDF page") from error

    blocks: list[ExtractedTextBlock] = []
    for block in raw_blocks:
        x0, y0, x1, y1, text, _, block_type = block
        if block_type == 0:
            x0, y0, x1, y1 = pymupdf.Rect(x0, y0, x1, y1) * page.rotation_matrix
            blocks.append(ExtractedTextBlock(text=text, bbox=(x0, y0, x1, y1)))
    return tuple(blocks)


def render_page(page: pymupdf.Page, *, dpi: int = 144) -> NDArray[np.uint8]:
    """Render a page to an RGB uint8 view (height, width, 3) with white background.

    The returned array is a zero-copy view of the PyMuPDF pixmap buffer and
    keeps that buffer alive for the lifetime of the array.
    """
    if not isinstance(dpi, int) or isinstance(dpi, bool):
        raise TypeError("dpi must be an integer")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    try:
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(dpi / 72, dpi / 72),
            colorspace=pymupdf.csRGB,
            alpha=False,
            annots=True,
        )
    except pymupdf.FileDataError as error:
        raise PdfInputError("Could not render the PDF page") from error

    return np.asarray(_PixmapBuffer(pixmap), dtype=np.uint8)

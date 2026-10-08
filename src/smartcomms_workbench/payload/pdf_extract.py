"""Pure PDF extraction from raw bytes, base64, or an XML envelope."""

import base64
import binascii
from dataclasses import dataclass
from typing import Literal

from lxml import etree

from smartcomms_workbench.diff.pdf import PdfInputError, open_pdf
from smartcomms_workbench.payload.xml import safe_parser


@dataclass(frozen=True)
class PdfExtractionFailure:
    error: str


def extract_pdf(
    data: bytes | str,
    *,
    format: Literal["pdf", "base64", "xml"] = "pdf",
    xpath: str = "//*[local-name()='bytes']/text()",
) -> bytes | PdfExtractionFailure:
    if format == "pdf":
        if not isinstance(data, bytes):
            return PdfExtractionFailure("Raw PDF data must be bytes")
        pdf = data
    elif format in ("xml", "base64"):
        try:
            text = data.decode("utf-8") if isinstance(data, bytes) else data
        except UnicodeDecodeError:
            return PdfExtractionFailure("Encoded PDF payload must be UTF-8 text")
        if format == "xml":
            try:
                root = etree.fromstring(text.encode("utf-8"), parser=safe_parser())
            except etree.XMLSyntaxError as error:
                return PdfExtractionFailure(str(error))
            values = root.xpath(xpath)
            if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str):
                return PdfExtractionFailure("PDF XPath must select exactly one text value")
            text = values[0]
        try:
            pdf = base64.b64decode("".join(text.split()), validate=True)
        except (binascii.Error, ValueError):
            return PdfExtractionFailure("PDF payload is not valid base64")
    else:
        raise ValueError(f"Unknown PDF extraction format: {format}")
    try:
        with open_pdf(pdf):
            pass
    except PdfInputError as error:
        return PdfExtractionFailure(str(error))
    return pdf

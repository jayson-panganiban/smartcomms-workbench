"""Pure SOAP 1.2 construction and parsing for the legacy review-case services."""

import base64
import binascii
from xml.etree import ElementTree as ET

from smartcomms_workbench.client.models import (
    ReviewCaseOperation,
    ReviewCaseResult,
    SoapFault,
    TransportFailure,
    TransportFailureKind,
)

SOAP_NS = "http://www.w3.org/2003/05/soap-envelope"
APP_NS = "http://www.cgu.com.au/dms/outbound/thunderhead/2012/01/24"

_OPERATION_NAMES = {
    ReviewCaseOperation.FLATTEN: "FlattenReviewCase",
    ReviewCaseOperation.EXPAND: "ExpandReviewCase",
}


def build_envelope(operation: ReviewCaseOperation, data: str) -> bytes:
    """Serialize escaped request data, keeping the bytes child unqualified."""
    name = _OPERATION_NAMES[operation]
    envelope = ET.Element(f"{{{SOAP_NS}}}Envelope")
    ET.SubElement(envelope, f"{{{SOAP_NS}}}Header")
    body = ET.SubElement(envelope, f"{{{SOAP_NS}}}Body")
    request = ET.SubElement(body, f"{{{APP_NS}}}{name}Request")
    ET.SubElement(request, "bytes").text = data
    return ET.tostring(envelope, encoding="utf-8")


def _parse_fault(fault: ET.Element) -> SoapFault | TransportFailure:
    code = fault.find(f"{{{SOAP_NS}}}Code/{{{SOAP_NS}}}Value")
    reason = fault.find(f"{{{SOAP_NS}}}Reason/{{{SOAP_NS}}}Text")
    if code is None or not code.text or reason is None or not reason.text:
        return TransportFailure(TransportFailureKind.INVALID_RESPONSE, "SOAP Fault is missing its code or reason.")
    detail = fault.find(f"{{{SOAP_NS}}}Detail")
    return SoapFault(
        code=code.text,
        reason=reason.text,
        detail=ET.tostring(detail, encoding="unicode") if detail is not None else None,
    )


def parse_response(operation: ReviewCaseOperation, body: bytes) -> ReviewCaseResult | SoapFault | TransportFailure:
    """Parse faults before successes; retain the legacy unqualified bytes fallback.

    Flatten decodes base64 UTF-8, while Expand leaves the response text unchanged.
    Invalid payload encodings and malformed XML are malformed responses; missing
    payloads or required fault fields are invalid response shapes.
    """
    name = _OPERATION_NAMES[operation]
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        return TransportFailure(TransportFailureKind.MALFORMED_RESPONSE, f"Malformed SOAP XML: {exc}")

    fault = next(root.iter(f"{{{SOAP_NS}}}Fault"), None)
    if fault is not None:
        return _parse_fault(fault)

    payload = root.find(f".//{{{APP_NS}}}{name}Response/bytes")
    if payload is None:
        payload = next(root.iter("bytes"), None)
    if payload is None or not payload.text or len(payload):
        return TransportFailure(TransportFailureKind.INVALID_RESPONSE, "SOAP response has no text-only bytes payload.")

    data = payload.text
    if operation is ReviewCaseOperation.FLATTEN:
        try:
            # XML Schema base64Binary permits whitespace between encoded characters.
            encoded = data.translate(str.maketrans("", "", " \t\r\n")).encode("ascii")
            data = base64.b64decode(encoded, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeEncodeError, UnicodeDecodeError) as exc:
            return TransportFailure(
                TransportFailureKind.MALFORMED_RESPONSE, f"Invalid base64 UTF-8 SOAP payload: {exc}"
            )
    return ReviewCaseResult(operation, data)

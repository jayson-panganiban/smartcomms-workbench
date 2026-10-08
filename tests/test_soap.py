import base64
from xml.etree import ElementTree as ET

import pytest

from smartcomms_workbench.client.models import (
    ReviewCaseOperation,
    ReviewCaseResult,
    SoapFault,
    TransportFailure,
    TransportFailureKind,
)
from smartcomms_workbench.client.soap import APP_NS, SOAP_NS, build_envelope, parse_response


def response_xml(operation_name: str, data: str) -> bytes:
    return (
        f'<soap:Envelope xmlns:soap="{SOAP_NS}" xmlns:app="{APP_NS}">'
        f"<soap:Body><app:{operation_name}Response><bytes>{data}</bytes>"
        f"</app:{operation_name}Response></soap:Body></soap:Envelope>"
    ).encode()


@pytest.mark.parametrize(
    ("operation", "request_name"),
    [
        (ReviewCaseOperation.FLATTEN, "FlattenReviewCaseRequest"),
        (ReviewCaseOperation.EXPAND, "ExpandReviewCaseRequest"),
    ],
)
@pytest.mark.parametrize("data", ['<xml attr="value">A & B — café</xml>', "", "YWJj"])
def test_build_envelope_shape_and_escaping(operation: ReviewCaseOperation, request_name: str, data: str) -> None:
    envelope = build_envelope(operation, data)
    assert isinstance(envelope, bytes)
    root = ET.fromstring(envelope)
    assert root.tag == f"{{{SOAP_NS}}}Envelope"
    assert [child.tag for child in root] == [f"{{{SOAP_NS}}}Header", f"{{{SOAP_NS}}}Body"]
    assert len(root[0]) == 0
    assert root[0].text is None
    request = root[1][0]
    assert request.tag == f"{{{APP_NS}}}{request_name}"
    assert len(request) == 1
    assert request[0].tag == "bytes"
    assert (request[0].text or "") == data
    assert len(request[0]) == 0


def test_flatten_decodes_base64_utf8() -> None:
    data = "<review>café & 文</review>"
    encoded = base64.b64encode(data.encode()).decode()
    result = parse_response(ReviewCaseOperation.FLATTEN, response_xml("FlattenReviewCase", encoded))
    assert result == ReviewCaseResult(ReviewCaseOperation.FLATTEN, data)


def test_flatten_accepts_base64_xml_whitespace() -> None:
    result = parse_response(ReviewCaseOperation.FLATTEN, response_xml("FlattenReviewCase", " YW \tJj\r\n"))
    assert result == ReviewCaseResult(ReviewCaseOperation.FLATTEN, "abc")


def test_expand_returns_bytes_text_unchanged() -> None:
    result = parse_response(ReviewCaseOperation.EXPAND, response_xml("ExpandReviewCase", " YWJj\n&amp;café "))
    assert result == ReviewCaseResult(ReviewCaseOperation.EXPAND, " YWJj\n&café ")


@pytest.mark.parametrize(
    ("operation", "data", "expected"),
    [
        (ReviewCaseOperation.FLATTEN, "YWJj", "abc"),
        (ReviewCaseOperation.EXPAND, "YWJj", "YWJj"),
    ],
)
def test_unqualified_bytes_fallback(operation: ReviewCaseOperation, data: str, expected: str) -> None:
    result = parse_response(operation, f"<legacy><wrapper><bytes>{data}</bytes></wrapper></legacy>".encode())
    assert result == ReviewCaseResult(operation, expected)


def test_operation_response_takes_precedence_over_fallback_bytes() -> None:
    body = (
        f'<root xmlns:app="{APP_NS}"><bytes>other</bytes>'
        "<app:ExpandReviewCaseResponse><bytes>expected</bytes></app:ExpandReviewCaseResponse></root>"
    ).encode()
    assert parse_response(ReviewCaseOperation.EXPAND, body) == ReviewCaseResult(ReviewCaseOperation.EXPAND, "expected")


@pytest.mark.parametrize("include_detail", [False, True])
def test_soap_fault_takes_precedence_over_success(include_detail: bool) -> None:
    detail = "<soap:Detail><issue>Invalid &amp; missing</issue></soap:Detail>" if include_detail else ""
    body = (
        f'<soap:Envelope xmlns:soap="{SOAP_NS}"><soap:Body><bytes>YWJj</bytes>'
        "<soap:Fault><soap:Code><soap:Value>soap:Sender</soap:Value></soap:Code>"
        '<soap:Reason><soap:Text xml:lang="en">Invalid request</soap:Text></soap:Reason>'
        f"{detail}</soap:Fault></soap:Body></soap:Envelope>"
    ).encode()
    result = parse_response(ReviewCaseOperation.FLATTEN, body)
    assert isinstance(result, SoapFault)
    assert result.code == "soap:Sender"
    assert result.reason == "Invalid request"
    if include_detail:
        assert result.detail is not None
        parsed_detail = ET.fromstring(result.detail)
        assert parsed_detail.tag == f"{{{SOAP_NS}}}Detail"
        assert parsed_detail[0].text == "Invalid & missing"
    else:
        assert result.detail is None


@pytest.mark.parametrize("body", [b"", b"<broken>", b"\xff", b"<root><bytes></root>"])
def test_malformed_xml_is_typed_failure(body: bytes) -> None:
    result = parse_response(ReviewCaseOperation.EXPAND, body)
    assert isinstance(result, TransportFailure)
    assert result.kind is TransportFailureKind.MALFORMED_RESPONSE
    assert result.message
    assert result.status_code is None


@pytest.mark.parametrize(
    "body",
    [
        b"<root/>",
        b"<root><bytes/></root>",
        b"<root><bytes><nested>data</nested></bytes></root>",
        f'<root xmlns:app="{APP_NS}"><app:bytes>data</app:bytes></root>'.encode(),
        f'<soap:Fault xmlns:soap="{SOAP_NS}"><soap:Reason><soap:Text>reason</soap:Text></soap:Reason></soap:Fault>'.encode(),
        f'<soap:Fault xmlns:soap="{SOAP_NS}"><soap:Code><soap:Value>code</soap:Value></soap:Code></soap:Fault>'.encode(),
    ],
)
def test_invalid_response_shape_is_typed_failure(body: bytes) -> None:
    result = parse_response(ReviewCaseOperation.EXPAND, body)
    assert isinstance(result, TransportFailure)
    assert result.kind is TransportFailureKind.INVALID_RESPONSE


@pytest.mark.parametrize("encoded", ["not base64!", "YWJ", "!!!!", "café", "//4="])
def test_invalid_base64_or_utf8_is_typed_failure(encoded: str) -> None:
    result = parse_response(ReviewCaseOperation.FLATTEN, response_xml("FlattenReviewCase", encoded))
    assert isinstance(result, TransportFailure)
    assert result.kind is TransportFailureKind.MALFORMED_RESPONSE


def test_programming_type_error_is_not_converted_to_transport_failure() -> None:
    with pytest.raises(TypeError):
        parse_response(ReviewCaseOperation.EXPAND, 42)  # ty: ignore[invalid-argument-type] - exercise misuse

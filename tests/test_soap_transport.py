import httpx
import pytest

from smartcomms_workbench.client.models import (
    ReviewCaseOperation,
    ReviewCaseRequest,
    ReviewCaseResult,
    SoapFault,
    TransportFailure,
    TransportFailureKind,
)
from smartcomms_workbench.client.soap import APP_NS, SOAP_NS
from smartcomms_workbench.client.transport import ReviewCaseSoapClient


class FakeResponse:
    def __init__(self, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content


class FakeHttp:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, bytes, dict[str, str], float]] = []

    def post(
        self,
        url: str,
        *,
        content: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> FakeResponse:
        self.calls.append((url, content, headers, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def response_xml(operation: str, data: str) -> bytes:
    return (
        f'<soap:Envelope xmlns:soap="{SOAP_NS}" xmlns:app="{APP_NS}">'
        f"<soap:Body><app:{operation}Response><bytes>{data}</bytes></app:{operation}Response>"
        "</soap:Body></soap:Envelope>"
    ).encode()


def make_client(fake_http: FakeHttp) -> ReviewCaseSoapClient:
    return ReviewCaseSoapClient(
        flatten_endpoint="https://appliance.example/flatten",
        expand_endpoint="https://appliance.example/expand",
        http=fake_http,
    )


@pytest.mark.parametrize(
    ("operation", "name", "endpoint", "content_type", "body"),
    [
        (
            ReviewCaseOperation.FLATTEN,
            "FlattenReviewCase",
            "https://appliance.example/flatten",
            "text/xml; charset=utf-8",
            "PHJldmlldz5vazwvcmV2aWV3Pg==",
        ),
        (
            ReviewCaseOperation.EXPAND,
            "ExpandReviewCase",
            "https://appliance.example/expand",
            "application/xml; charset=utf-8",
            "expanded-data",
        ),
    ],
)
def test_execute_posts_legacy_operation_and_parses_success(
    operation: ReviewCaseOperation,
    name: str,
    endpoint: str,
    content_type: str,
    body: str,
) -> None:
    fake_http = FakeHttp(FakeResponse(200, response_xml(name, body)))
    client = make_client(fake_http)

    result = client.execute(ReviewCaseRequest(operation, "<input/>"))

    if operation is ReviewCaseOperation.FLATTEN:
        assert result == ReviewCaseResult(operation, "<review>ok</review>")
    else:
        assert result == ReviewCaseResult(operation, "expanded-data")
    assert len(fake_http.calls) == 1
    called_url, content, headers, timeout = fake_http.calls[0]
    assert called_url == endpoint
    assert headers == {"Content-Type": content_type}
    assert timeout == 30.0
    assert name.encode() in content
    assert b"<bytes>&lt;input/&gt;</bytes>" in content


def test_soap_fault_is_returned_even_with_non_success_http_status() -> None:
    fault_body = (
        f'<soap:Envelope xmlns:soap="{SOAP_NS}"><soap:Body><soap:Fault>'
        "<soap:Code><soap:Value>soap:Receiver</soap:Value></soap:Code>"
        "<soap:Reason><soap:Text>Remote processing failed</soap:Text></soap:Reason>"
        "</soap:Fault></soap:Body></soap:Envelope>"
    ).encode()
    client = make_client(FakeHttp(FakeResponse(500, fault_body)))

    result = client.execute(ReviewCaseRequest(ReviewCaseOperation.EXPAND, "payload"))

    assert result == SoapFault("soap:Receiver", "Remote processing failed")


@pytest.mark.parametrize("status_code", [300, 400, 503])
def test_non_success_status_returns_http_failure(status_code: int) -> None:
    client = make_client(FakeHttp(FakeResponse(status_code, response_xml("ExpandReviewCase", "unexpected"))))

    result = client.execute(ReviewCaseRequest(ReviewCaseOperation.EXPAND, "payload"))

    assert result == TransportFailure(
        TransportFailureKind.HTTP_ERROR,
        f"SOAP endpoint returned HTTP {status_code}.",
        status_code,
    )


def test_success_http_with_malformed_soap_returns_protocol_failure() -> None:
    client = make_client(FakeHttp(FakeResponse(200, b"<broken")))

    result = client.execute(ReviewCaseRequest(ReviewCaseOperation.EXPAND, "payload"))

    assert isinstance(result, TransportFailure)
    assert result.kind is TransportFailureKind.MALFORMED_RESPONSE


def test_timeout_returns_typed_failure_without_retry() -> None:
    fake_http = FakeHttp(httpx.ReadTimeout("timed out"))
    client = make_client(fake_http)

    result = client.execute(ReviewCaseRequest(ReviewCaseOperation.EXPAND, "payload"))

    assert result == TransportFailure(TransportFailureKind.TIMEOUT, "timed out")
    assert len(fake_http.calls) == 1


def test_connection_failure_returns_typed_failure() -> None:
    fake_http = FakeHttp(httpx.ConnectError("connection refused"))
    client = make_client(fake_http)

    result = client.execute(ReviewCaseRequest(ReviewCaseOperation.EXPAND, "payload"))

    assert result == TransportFailure(TransportFailureKind.CONNECTION_ERROR, "connection refused")


def test_unexpected_programming_error_propagates() -> None:
    client = make_client(FakeHttp(TypeError("bug")))

    with pytest.raises(TypeError, match="bug"):
        client.execute(ReviewCaseRequest(ReviewCaseOperation.EXPAND, "payload"))

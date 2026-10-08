import base64
import json
from collections.abc import Callable

import httpx
import pytest

from smartcomms_workbench.client.auth import OAuth1Signer, OAuth2Client, OAuth2Token
from smartcomms_workbench.client.gateway import SmartCommsClient
from smartcomms_workbench.client.models import (
    DraftResult,
    JobResult,
    JobStatus,
    SubmitResult,
    TransportFailure,
    TransportFailureKind,
)
from smartcomms_workbench.client.rest import CloudRestClient, RestContract

CONTRACT = RestContract(
    "https://example.test/submit", "https://example.test/draft/{job_id}", "https://example.test/jobs/{job_id}"
)


def test_configured_contract_and_gateway(pdf_factory: Callable[[str], bytes]) -> None:
    pdf = pdf_factory("REST")
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST":
            assert json.loads(request.content) == {"template": "template-1", "xml": "<r/>"}
            return httpx.Response(200, json={"id": "job-1", "document": base64.b64encode(pdf).decode()})
        if "draft" in request.url.path:
            return httpx.Response(200, json={"content": "<draft/>"})
        return httpx.Response(200, json={"state": "completed", "document": base64.b64encode(pdf).decode()})

    contract = RestContract(
        CONTRACT.submit_url,
        CONTRACT.draft_url,
        CONTRACT.job_url,
        template_field="template",
        payload_field="xml",
        job_id_field="id",
        pdf_field="document",
        draft_field="content",
        status_field="state",
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = SmartCommsClient(rest=CloudRestClient(contract, http))
        assert client.submit("template-1", "<r/>") == SubmitResult("job-1", pdf)
        assert client.draft("job-1") == DraftResult("<draft/>")
        assert client.job("job-1") == JobResult("job-1", JobStatus.COMPLETED, pdf)
    assert len(calls) == 3


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (httpx.Response(503), TransportFailureKind.HTTP_ERROR),
        (httpx.Response(200, content=b"not-json"), TransportFailureKind.MALFORMED_RESPONSE),
        (httpx.Response(200, json=[]), TransportFailureKind.INVALID_RESPONSE),
        (httpx.Response(200, json={}), TransportFailureKind.INVALID_RESPONSE),
        (httpx.Response(200, json={"job_id": "j", "pdf": "!bad"}), TransportFailureKind.MALFORMED_RESPONSE),
        (httpx.Response(200, json={"job_id": "j", "pdf": None}), TransportFailureKind.INVALID_RESPONSE),
    ],
)
def test_rest_failure_shapes(response: httpx.Response, kind: TransportFailureKind) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: response)) as http:
        result = CloudRestClient(CONTRACT, http).submit("t", "<r/>")
    assert isinstance(result, TransportFailure)
    assert result.kind == kind


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (httpx.ReadTimeout("timeout"), TransportFailureKind.TIMEOUT),
        (httpx.ConnectError("offline"), TransportFailureKind.CONNECTION_ERROR),
    ],
)
def test_rest_network_failures(error: httpx.RequestError, kind: TransportFailureKind) -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        raise error

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        result = CloudRestClient(CONTRACT, http).submit("t", "<r/>")
    assert isinstance(result, TransportFailure) and result.kind == kind


def test_programming_errors_not_suppressed() -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        raise TypeError("bug")

    with httpx.Client(transport=httpx.MockTransport(respond)) as http, pytest.raises(TypeError, match="bug"):
        CloudRestClient(CONTRACT, http).submit("t", "<r/>")
    with pytest.raises(ValueError, match="configured"):
        SmartCommsClient().submit("t", "<r/>")


def test_job_polling_is_bounded_and_failure_explicit() -> None:
    calls = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"status": "running" if calls < 2 else "completed"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = CloudRestClient(CONTRACT, http)
        result = client.wait("j", max_attempts=2)
        assert isinstance(result, JobResult) and result.status == JobStatus.COMPLETED
    assert calls == 2
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"status": "pending"}))) as http:
        exhausted = CloudRestClient(CONTRACT, http).wait("j", max_attempts=2)
    assert isinstance(exhausted, TransportFailure) and exhausted.kind == TransportFailureKind.TIMEOUT


def test_oauth1_signs_json_and_get_requests() -> None:
    signatures: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"].startswith("OAuth ")
        signatures.append(request.headers["Authorization"])
        return httpx.Response(200, json={"job_id": "j", "status": "completed"})

    with httpx.Client(
        transport=httpx.MockTransport(respond),
        auth=OAuth1Signer("consumer", "secret", token="token", token_secret="token-secret"),
    ) as http:
        http.post("https://example.test", json={"xml": "<r/>"})
        http.get("https://example.test")
    assert len(signatures) == 2 and signatures[0] != signatures[1]
    assert "oauth_signature=" in signatures[0]


def test_oauth2_client_credentials_request() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"].startswith("Basic ")
        assert b"grant_type=client_credentials" in request.content
        assert b"scope=documents" in request.content
        return httpx.Response(200, json={"access_token": "token", "token_type": "Bearer", "expires_in": 3600})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        assert OAuth2Client(http).acquire_token(
            "https://example.test/token", "id", "secret", scope="documents"
        ) == OAuth2Token("token", 3600)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"access_token": "", "token_type": "Bearer"},
        {"access_token": "t", "token_type": "MAC"},
        {"access_token": "t", "token_type": "Bearer", "expires_in": -1},
    ],
)
def test_oauth2_invalid_tokens_are_failures(payload: dict[str, object]) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as http:
        assert isinstance(OAuth2Client(http).acquire_token("https://example.test/token", "id", "s"), TransportFailure)


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "unknown"},
        {"status": "completed", "error": 1},
        {"status": "completed", "pdf": "!"},
    ],
)
def test_job_invalid_responses_are_explicit(payload: dict[str, object]) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as http:
        assert isinstance(CloudRestClient(CONTRACT, http).job("j"), TransportFailure)


def test_missing_draft_is_not_success() -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": ""}))) as http:
        assert isinstance(CloudRestClient(CONTRACT, http).draft("j"), TransportFailure)


@pytest.mark.parametrize("response", [httpx.Response(401), httpx.Response(200, content=b"invalid JSON")])
def test_oauth2_http_and_malformed_failures(response: httpx.Response) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: response)) as http:
        assert isinstance(OAuth2Client(http).acquire_token("https://example.test/token", "id", "s"), TransportFailure)

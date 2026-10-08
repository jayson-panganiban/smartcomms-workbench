"""HTTP transport for the legacy SmartComms Review Case SOAP operations."""

from typing import Protocol

import httpx

from smartcomms_workbench.client.models import (
    ReviewCaseOperation,
    ReviewCaseRequest,
    ReviewCaseResult,
    SoapFault,
    TransportFailure,
    TransportFailureKind,
)
from smartcomms_workbench.client.soap import build_envelope, parse_response


class HttpResponse(Protocol):
    """The response data the transport needs from an HTTP adapter."""

    @property
    def status_code(self) -> int: ...

    @property
    def content(self) -> bytes: ...


class HttpTransport(Protocol):
    """The minimal HTTP seam used by the SOAP client."""

    def post(
        self,
        url: str,
        *,
        content: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> HttpResponse:
        """POST bytes and return the raw status and body."""
        ...


type ReviewCaseOutcome = ReviewCaseResult | SoapFault | TransportFailure


class ReviewCaseSoapClient:
    """Call the legacy Flatten and Expand Review Case SOAP endpoints.

    The caller supplies the endpoint URLs and owns the injected HTTP transport's
    lifetime. SOAP construction and parsing remain pure in ``client.soap``.
    """

    def __init__(
        self,
        *,
        flatten_endpoint: str,
        expand_endpoint: str,
        http: HttpTransport,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._endpoints = {
            ReviewCaseOperation.FLATTEN: flatten_endpoint,
            ReviewCaseOperation.EXPAND: expand_endpoint,
        }
        self._http = http
        self._timeout_seconds = timeout_seconds

    def execute(self, request: ReviewCaseRequest) -> ReviewCaseOutcome:
        """Send one review-case request and return its typed expected outcome."""
        endpoint = self._endpoints[request.operation]
        headers = {
            "Content-Type": (
                "text/xml; charset=utf-8"
                if request.operation is ReviewCaseOperation.FLATTEN
                else "application/xml; charset=utf-8"
            )
        }
        envelope = build_envelope(request.operation, request.data)
        try:
            response = self._http.post(
                endpoint,
                content=envelope,
                headers=headers,
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            return TransportFailure(TransportFailureKind.TIMEOUT, str(exc))
        except httpx.RequestError as exc:
            return TransportFailure(TransportFailureKind.CONNECTION_ERROR, str(exc))

        parsed = parse_response(request.operation, response.content)
        if isinstance(parsed, SoapFault):
            return parsed
        if not 200 <= response.status_code < 300:
            return TransportFailure(
                TransportFailureKind.HTTP_ERROR,
                f"SOAP endpoint returned HTTP {response.status_code}.",
                status_code=response.status_code,
            )
        return parsed

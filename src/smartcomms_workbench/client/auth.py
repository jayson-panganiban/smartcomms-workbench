"""OAuth adapters at the HTTP boundary; credentials are supplied by consumers."""

from collections.abc import Generator
from dataclasses import dataclass, field
from math import isfinite

import httpx
from oauthlib.oauth1 import Client as OAuth1Client

from smartcomms_workbench.client.models import TransportFailure, TransportFailureKind


class OAuth1Signer(httpx.Auth):
    """Sign each request using OAuth1 HMAC-SHA1 with a fresh nonce/timestamp."""

    requires_request_body = True

    def __init__(
        self, consumer_key: str, consumer_secret: str, *, token: str | None = None, token_secret: str | None = None
    ) -> None:
        self._signer = OAuth1Client(
            consumer_key, client_secret=consumer_secret, resource_owner_key=token, resource_owner_secret=token_secret
        )

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response]:
        url, headers, _ = self._signer.sign(
            str(request.url),
            http_method=request.method,
            body=request.content.decode("utf-8") if request.content else None,
            headers=dict(request.headers),
        )
        request.url = httpx.URL(url)
        request.headers.update(headers)
        yield request


@dataclass(frozen=True)
class OAuth2Token:
    access_token: str = field(repr=False)
    expires_in: float | None = None


class OAuth2Client:
    """Explicit client-credentials token acquisition; no hidden token caching."""

    def __init__(self, http: httpx.Client, *, timeout_seconds: float = 30.0) -> None:
        if isinstance(timeout_seconds, bool) or not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self._http = http
        self._timeout = timeout_seconds

    def acquire_token(
        self, endpoint: str, client_id: str, client_secret: str, *, scope: str | None = None
    ) -> OAuth2Token | TransportFailure:
        data = {"grant_type": "client_credentials"}
        if scope is not None:
            data["scope"] = scope
        try:
            response = self._http.post(endpoint, data=data, auth=(client_id, client_secret), timeout=self._timeout)
        except httpx.TimeoutException as error:
            return TransportFailure(TransportFailureKind.TIMEOUT, str(error))
        except httpx.RequestError as error:
            return TransportFailure(TransportFailureKind.CONNECTION_ERROR, str(error))
        if not response.is_success:
            return TransportFailure(
                TransportFailureKind.HTTP_ERROR,
                f"OAuth2 endpoint returned HTTP {response.status_code}.",
                response.status_code,
            )
        try:
            payload = response.json()
        except ValueError:
            return TransportFailure(TransportFailureKind.MALFORMED_RESPONSE, "OAuth2 response is not JSON.")
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("access_token"), str)
            or not payload["access_token"]
            or str(payload.get("token_type", "")).lower() != "bearer"
        ):
            return TransportFailure(TransportFailureKind.INVALID_RESPONSE, "OAuth2 response needs a Bearer token.")
        expiry = payload.get("expires_in")
        if expiry is not None and (
            isinstance(expiry, bool) or not isinstance(expiry, (float, int)) or not isfinite(expiry) or expiry < 0
        ):
            return TransportFailure(TransportFailureKind.INVALID_RESPONSE, "Invalid OAuth2 token expiry.")
        return OAuth2Token(payload["access_token"], expiry)

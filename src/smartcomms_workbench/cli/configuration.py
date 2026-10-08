"""Validated client configuration with environment-only credential references."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field

from smartcomms_workbench.client.auth import OAuth1Signer, OAuth2Client
from smartcomms_workbench.client.gateway import SmartCommsClient
from smartcomms_workbench.client.models import TransportFailure
from smartcomms_workbench.client.rest import CloudRestClient, RestContract
from smartcomms_workbench.client.transport import ReviewCaseSoapClient


class Configuration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ClientConfigurationError(ValueError):
    """An expected missing/invalid boundary configuration or authentication failure."""


class RestConfiguration(Configuration):
    submit_url: str
    draft_url: str | None = None
    job_url: str | None = None
    template_field: str = "template_id"
    payload_field: str = "payload_xml"
    job_id_field: str = "job_id"
    pdf_field: str = "pdf"
    draft_field: str = "data"
    status_field: str = "status"
    error_field: str = "error"


class SoapConfiguration(Configuration):
    flatten_endpoint: str
    expand_endpoint: str


class AuthConfiguration(Configuration):
    type: Literal["none", "bearer", "oauth1", "oauth2"] = "none"
    token_env: str | None = None
    consumer_key_env: str | None = None
    consumer_secret_env: str | None = None
    token_secret_env: str | None = None
    token_url: str | None = None
    client_id_env: str | None = None
    client_secret_env: str | None = None
    scope: str | None = None


class ClientConfiguration(Configuration):
    rest: RestConfiguration | None = None
    soap: SoapConfiguration | None = None
    auth: AuthConfiguration = Field(default_factory=AuthConfiguration)
    timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)


def _credential(name: str | None) -> str:
    if not name:
        raise ClientConfigurationError("Authentication requires an environment variable reference")
    value = os.environ.get(name)
    if not value:
        raise ClientConfigurationError(f"Credential environment variable is missing or empty: {name}")
    return value


@contextmanager
def configured_client(path: Path | None) -> Generator[SmartCommsClient | None]:
    """Own the HTTP lifetime; no network is used when no client is configured."""
    if path is None:
        yield None
        return
    config = ClientConfiguration.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if config.rest is None and config.soap is None:
        raise ClientConfigurationError("Client configuration needs REST or SOAP endpoints")
    try:
        contract = RestContract(**config.rest.model_dump()) if config.rest is not None else None
    except ValueError as error:
        raise ClientConfigurationError(str(error)) from error
    with httpx.Client(timeout=config.timeout_seconds) as http:
        auth = config.auth
        if auth.type == "bearer":
            http.headers["Authorization"] = f"Bearer {_credential(auth.token_env)}"
        elif auth.type == "oauth1":
            if (auth.token_env is None) != (auth.token_secret_env is None):
                raise ClientConfigurationError("OAuth1 token_env and token_secret_env must be supplied together")
            http.auth = OAuth1Signer(
                _credential(auth.consumer_key_env),
                _credential(auth.consumer_secret_env),
                token=_credential(auth.token_env) if auth.token_env else None,
                token_secret=_credential(auth.token_secret_env) if auth.token_secret_env else None,
            )
        elif auth.type == "oauth2":
            if auth.token_url is None:
                raise ClientConfigurationError("OAuth2 requires token_url")
            token = OAuth2Client(http, timeout_seconds=config.timeout_seconds).acquire_token(
                auth.token_url, _credential(auth.client_id_env), _credential(auth.client_secret_env), scope=auth.scope
            )
            if isinstance(token, TransportFailure):
                raise ClientConfigurationError(f"OAuth2 authentication failed: {token.message}")
            http.headers["Authorization"] = f"Bearer {token.access_token}"
        rest = CloudRestClient(contract, http, timeout_seconds=config.timeout_seconds) if contract else None
        soap = (
            ReviewCaseSoapClient(
                flatten_endpoint=config.soap.flatten_endpoint,
                expand_endpoint=config.soap.expand_endpoint,
                http=http,
                timeout_seconds=config.timeout_seconds,
            )
            if config.soap is not None
            else None
        )
        yield SmartCommsClient(rest=rest, soap=soap)

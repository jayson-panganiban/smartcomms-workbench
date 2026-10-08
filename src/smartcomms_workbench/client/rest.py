"""Configurable JSON integration, not an assumed SmartComms vendor API schema."""

import base64
import binascii
from dataclasses import dataclass
from math import isfinite
from time import sleep
from urllib.parse import quote, urlsplit

import httpx

from smartcomms_workbench.client.models import (
    DraftResult,
    JobResult,
    JobStatus,
    SubmitResult,
    TransportFailure,
    TransportFailureKind,
)


@dataclass(frozen=True)
class RestContract:
    """Explicit URLs and JSON field mappings for a deployed integration.

    Draft/job URLs may include ``{job_id}``; returned PDF fields contain base64.
    Submit sends template/payload fields; draft and job queries use GET.
    """

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

    def __post_init__(self) -> None:
        for url in (self.submit_url, self.draft_url, self.job_url):
            if url is not None and (urlsplit(url).scheme not in ("http", "https") or not urlsplit(url).netloc):
                raise ValueError("REST endpoints must be absolute HTTP(S) URLs")
        fields = (
            self.template_field,
            self.payload_field,
            self.job_id_field,
            self.pdf_field,
            self.draft_field,
            self.status_field,
            self.error_field,
        )
        if any(not field for field in fields) or self.template_field == self.payload_field:
            raise ValueError("REST field names must be nonempty and request fields distinct")


def _decode_pdf(payload: dict[str, object], field: str) -> bytes | TransportFailure | None:
    if field not in payload:
        return None
    value = payload[field]
    if not isinstance(value, str) or not value:
        return TransportFailure(TransportFailureKind.INVALID_RESPONSE, "PDF field must be a nonempty base64 string.")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return TransportFailure(TransportFailureKind.MALFORMED_RESPONSE, "PDF field is not valid base64.")


class CloudRestClient:
    """The caller owns the HTTP client's lifetime and configures its authentication."""

    def __init__(self, contract: RestContract, http: httpx.Client, *, timeout_seconds: float = 30.0) -> None:
        if isinstance(timeout_seconds, bool) or not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self.contract = contract
        self._http = http
        self._timeout = timeout_seconds

    def _request(
        self, method: str, url: str, *, body: dict[str, str] | None = None
    ) -> dict[str, object] | TransportFailure:
        try:
            response = self._http.request(method, url, json=body, timeout=self._timeout)
        except httpx.TimeoutException as error:
            return TransportFailure(TransportFailureKind.TIMEOUT, str(error))
        except httpx.RequestError as error:
            return TransportFailure(TransportFailureKind.CONNECTION_ERROR, str(error))
        if not response.is_success:
            return TransportFailure(
                TransportFailureKind.HTTP_ERROR,
                f"REST endpoint returned HTTP {response.status_code}.",
                response.status_code,
            )
        try:
            payload = response.json()
        except ValueError:
            return TransportFailure(TransportFailureKind.MALFORMED_RESPONSE, "REST response is not JSON.")
        if not isinstance(payload, dict):
            return TransportFailure(TransportFailureKind.INVALID_RESPONSE, "REST response must be a JSON object.")
        return payload

    def submit(self, template_id: str, payload_xml: str) -> SubmitResult | TransportFailure:
        payload = self._request(
            "POST",
            self.contract.submit_url,
            body={self.contract.template_field: template_id, self.contract.payload_field: payload_xml},
        )
        if isinstance(payload, TransportFailure):
            return payload
        job_id = payload.get(self.contract.job_id_field)
        if not isinstance(job_id, str) or not job_id:
            return TransportFailure(TransportFailureKind.INVALID_RESPONSE, "Submission response needs a job ID.")
        pdf = _decode_pdf(payload, self.contract.pdf_field)
        return pdf if isinstance(pdf, TransportFailure) else SubmitResult(job_id, pdf)

    def _job_endpoint(self, url: str | None, job_id: str) -> str:
        if url is None:
            raise ValueError("The requested REST operation has no configured URL")
        if not job_id:
            raise ValueError("job_id must be nonempty")
        return url.replace("{job_id}", quote(job_id, safe=""))

    def draft(self, job_id: str) -> DraftResult | TransportFailure:
        payload = self._request("GET", self._job_endpoint(self.contract.draft_url, job_id))
        if isinstance(payload, TransportFailure):
            return payload
        data = payload.get(self.contract.draft_field)
        if not isinstance(data, str) or not data:
            return TransportFailure(TransportFailureKind.INVALID_RESPONSE, "Draft response needs nonempty text data.")
        return DraftResult(data)

    def job(self, job_id: str) -> JobResult | TransportFailure:
        payload = self._request("GET", self._job_endpoint(self.contract.job_url, job_id))
        if isinstance(payload, TransportFailure):
            return payload
        status = payload.get(self.contract.status_field)
        if not isinstance(status, str) or status not in JobStatus:
            return TransportFailure(TransportFailureKind.INVALID_RESPONSE, "Job response has an unknown status.")
        error = payload.get(self.contract.error_field)
        if error is not None and not isinstance(error, str):
            return TransportFailure(TransportFailureKind.INVALID_RESPONSE, "Job error must be text.")
        pdf = _decode_pdf(payload, self.contract.pdf_field)
        if isinstance(pdf, TransportFailure):
            return pdf
        return JobResult(job_id, JobStatus(status), pdf, error)

    def wait(
        self, job_id: str, *, max_attempts: int = 1, interval_seconds: float = 0.0
    ) -> JobResult | TransportFailure:
        """Bounded polling, with waiting confined to the transport layer."""
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if isinstance(interval_seconds, bool) or not isfinite(interval_seconds) or interval_seconds < 0:
            raise ValueError("interval_seconds must be finite and nonnegative")
        for attempt in range(max_attempts):
            if attempt:
                sleep(interval_seconds)
            result = self.job(job_id)
            if isinstance(result, TransportFailure) or result.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                return result
        return TransportFailure(TransportFailureKind.TIMEOUT, f"Job did not finish in {max_attempts} attempts.")

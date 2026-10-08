"""Immutable inputs and outcomes for the legacy review-case SOAP operations."""

from dataclasses import dataclass
from enum import StrEnum


class ReviewCaseOperation(StrEnum):
    """Legacy ao-tools operations, not REST job submission operations."""

    FLATTEN = "flatten"
    EXPAND = "expand"


@dataclass(frozen=True)
class ReviewCaseRequest:
    """Operation and review-case data supplied to the SOAP transport."""

    operation: ReviewCaseOperation
    data: str


@dataclass(frozen=True)
class ReviewCaseResult:
    """Flatten carries decoded XML; Expand carries response bytes element text."""

    operation: ReviewCaseOperation
    data: str


@dataclass(frozen=True)
class SoapFault:
    """An expected SOAP 1.2 fault, separate from transport failures."""

    code: str | None
    reason: str
    detail: str | None = None


class TransportFailureKind(StrEnum):
    """Expected failures at the HTTP and SOAP response boundaries."""

    HTTP_ERROR = "http_error"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    MALFORMED_RESPONSE = "malformed_response"
    INVALID_RESPONSE = "invalid_response"


@dataclass(frozen=True)
class TransportFailure:
    """An expected transport failure, not a caught programming exception."""

    kind: TransportFailureKind
    message: str
    status_code: int | None = None


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class SubmitResult:
    job_id: str
    pdf: bytes | None = None


@dataclass(frozen=True)
class DraftResult:
    data: str


@dataclass(frozen=True)
class JobResult:
    job_id: str
    status: JobStatus
    pdf: bytes | None = None
    error: str | None = None

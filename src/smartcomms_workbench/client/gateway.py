"""Unified, explicitly configured client facade."""

from smartcomms_workbench.client.models import (
    DraftResult,
    JobResult,
    ReviewCaseOperation,
    ReviewCaseRequest,
    SubmitResult,
    TransportFailure,
)
from smartcomms_workbench.client.rest import CloudRestClient
from smartcomms_workbench.client.transport import ReviewCaseOutcome, ReviewCaseSoapClient


class SmartCommsClient:
    def __init__(self, *, rest: CloudRestClient | None = None, soap: ReviewCaseSoapClient | None = None) -> None:
        self.rest = rest
        self.soap = soap

    def submit(self, template_id: str, payload_xml: str) -> SubmitResult | TransportFailure:
        if self.rest is None:
            raise ValueError("Submission requires a configured REST client")
        return self.rest.submit(template_id, payload_xml)

    def draft(self, job_id: str) -> DraftResult | TransportFailure:
        if self.rest is None:
            raise ValueError("Draft retrieval requires a configured REST client")
        return self.rest.draft(job_id)

    def job(self, job_id: str) -> JobResult | TransportFailure:
        if self.rest is None:
            raise ValueError("Job retrieval requires a configured REST client")
        return self.rest.job(job_id)

    def wait(
        self, job_id: str, *, max_attempts: int = 1, interval_seconds: float = 0.0
    ) -> JobResult | TransportFailure:
        if self.rest is None:
            raise ValueError("Job polling requires a configured REST client")
        return self.rest.wait(job_id, max_attempts=max_attempts, interval_seconds=interval_seconds)

    def review_case(self, operation: ReviewCaseOperation, data: str) -> ReviewCaseOutcome:
        if self.soap is None:
            raise ValueError("Review-case operations require a configured SOAP client")
        return self.soap.execute(ReviewCaseRequest(operation, data))

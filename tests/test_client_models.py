from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from smartcomms_workbench.client.models import (
    ReviewCaseOperation,
    ReviewCaseRequest,
    ReviewCaseResult,
    SoapFault,
    TransportFailure,
    TransportFailureKind,
)


def test_review_case_operation_values_are_stable() -> None:
    assert {operation.name: operation.value for operation in ReviewCaseOperation} == {
        "FLATTEN": "flatten",
        "EXPAND": "expand",
    }
    assert ReviewCaseOperation("flatten") is ReviewCaseOperation.FLATTEN
    assert ReviewCaseOperation("expand") is ReviewCaseOperation.EXPAND
    with pytest.raises(ValueError):
        ReviewCaseOperation("submit_job")


@pytest.mark.parametrize("operation", list(ReviewCaseOperation))
def test_request_preserves_operation_and_payload(operation: ReviewCaseOperation) -> None:
    data = "<reviewCase>résumé &amp; notes</reviewCase>"
    request = ReviewCaseRequest(operation, data)

    assert request.operation is operation
    assert request.data == data
    assert request == ReviewCaseRequest(operation=operation, data=data)


@pytest.mark.parametrize(
    ("operation", "data"),
    [
        (ReviewCaseOperation.FLATTEN, "<reviewCase>résumé</reviewCase>"),
        (ReviewCaseOperation.EXPAND, "PHJldmlld0Nhc2UvPg=="),
        (ReviewCaseOperation.FLATTEN, ""),
        (ReviewCaseOperation.EXPAND, ""),
    ],
)
def test_success_preserves_operation_specific_result_text(operation: ReviewCaseOperation, data: str) -> None:
    result = ReviewCaseResult(operation, data)

    assert result.operation is operation
    assert result.data == data
    assert result == ReviewCaseResult(operation=operation, data=data)


def test_fault_preserves_soap_fields_and_optional_detail() -> None:
    fault = SoapFault("env:Sender", "Review case invalid", "<validation>Missing field</validation>")

    assert fault.code == "env:Sender"
    assert fault.reason == "Review case invalid"
    assert fault.detail == "<validation>Missing field</validation>"
    assert SoapFault(None, "Unknown fault").detail is None
    assert SoapFault(None, "Unknown fault").code is None


def test_transport_failure_kinds_are_distinct_and_stable() -> None:
    assert {kind.name: kind.value for kind in TransportFailureKind} == {
        "HTTP_ERROR": "http_error",
        "TIMEOUT": "timeout",
        "CONNECTION_ERROR": "connection_error",
        "MALFORMED_RESPONSE": "malformed_response",
        "INVALID_RESPONSE": "invalid_response",
    }
    with pytest.raises(ValueError):
        TransportFailureKind("programming_error")


@pytest.mark.parametrize("kind", list(TransportFailureKind))
def test_transport_failure_preserves_kind_and_message(kind: TransportFailureKind) -> None:
    failure = TransportFailure(kind, "Failure description")

    assert failure.kind is kind
    assert failure.message == "Failure description"
    assert failure.status_code is None


def test_http_failure_preserves_status_code() -> None:
    failure = TransportFailure(TransportFailureKind.HTTP_ERROR, "Service unavailable", 503)

    assert failure.status_code == 503
    assert failure == TransportFailure(TransportFailureKind.HTTP_ERROR, "Service unavailable", status_code=503)
    assert not isinstance(failure, SoapFault)


@pytest.mark.parametrize(
    ("model", "field_name"),
    [
        (ReviewCaseRequest(ReviewCaseOperation.FLATTEN, "<case/>"), "operation"),
        (ReviewCaseRequest(ReviewCaseOperation.FLATTEN, "<case/>"), "data"),
        (ReviewCaseResult(ReviewCaseOperation.EXPAND, "Ynl0ZXM="), "operation"),
        (ReviewCaseResult(ReviewCaseOperation.EXPAND, "Ynl0ZXM="), "data"),
        (SoapFault("env:Receiver", "Failed"), "code"),
        (SoapFault("env:Receiver", "Failed"), "reason"),
        (SoapFault("env:Receiver", "Failed"), "detail"),
        (TransportFailure(TransportFailureKind.TIMEOUT, "Timed out"), "kind"),
        (TransportFailure(TransportFailureKind.TIMEOUT, "Timed out"), "message"),
        (TransportFailure(TransportFailureKind.TIMEOUT, "Timed out"), "status_code"),
    ],
)
def test_all_model_fields_are_frozen(
    model: ReviewCaseRequest | ReviewCaseResult | SoapFault | TransportFailure, field_name: str
) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(model, field_name, None)


@pytest.mark.parametrize("model_type", [ReviewCaseRequest, ReviewCaseResult, SoapFault, TransportFailure])
def test_constructor_programming_errors_are_not_converted_to_failures(
    model_type: type[ReviewCaseRequest | ReviewCaseResult | SoapFault | TransportFailure],
) -> None:
    invalid_fields: dict[str, Any] = {"unknown_field": "invalid"}
    with pytest.raises(TypeError):
        model_type(**invalid_fields)

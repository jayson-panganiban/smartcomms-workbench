"""Pure record parsing and XML hydration for CSV and JSON matrices."""

import csv
import io
import json
from collections.abc import Mapping, Sequence
from typing import Literal

from smartcomms_workbench.payload.hydrator import HydrationResult, hydrate_xml


def parse_matrix(text: str, *, format: Literal["csv", "json"]) -> tuple[Mapping[str, object], ...]:
    """Parse an explicit matrix format; malformed input raises ValueError."""
    text = text.removeprefix("\ufeff")
    if format == "json":
        records = json.loads(text)
        if not isinstance(records, list) or any(
            not isinstance(record, dict) or any(not isinstance(key, str) for key in record) for record in records
        ):
            raise ValueError("JSON matrix must be an array of objects")
        return tuple(records)
    if format != "csv":
        raise ValueError("Matrix format must be csv or json")
    reader = csv.DictReader(io.StringIO(text), strict=True)
    try:
        headers = reader.fieldnames
    except csv.Error as error:
        raise ValueError(f"Invalid CSV matrix: {error}") from error
    if not headers or any(not header for header in headers) or len(set(headers)) != len(headers):
        raise ValueError("CSV matrix needs unique, nonempty column names")
    rows: list[Mapping[str, object]] = []
    try:
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("CSV row does not match its header")
            rows.append(row)
    except csv.Error as error:
        raise ValueError(f"Invalid CSV matrix: {error}") from error
    return tuple(rows)


def hydrate_matrix(template: str, records: Sequence[Mapping[str, object]]) -> tuple[HydrationResult, ...]:
    """Render each record in order without side effects."""
    return tuple(hydrate_xml(template, record) for record in records)

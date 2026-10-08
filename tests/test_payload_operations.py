import base64
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest
from lxml import etree

from smartcomms_workbench.payload.hydrator import HydrationFailure, HydrationSuccess
from smartcomms_workbench.payload.matrix import hydrate_matrix, parse_matrix
from smartcomms_workbench.payload.mutator import XPathMutation, mutate_xml, validate_xml
from smartcomms_workbench.payload.pdf_extract import PdfExtractionFailure, extract_pdf

SCHEMA = """<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
<xs:element name="amount" type="xs:integer"/></xs:schema>"""


@pytest.mark.parametrize(
    ("format", "text"),
    [("csv", 'name,amount\n"A & B",1\nC,2\n'), ("json", '[{"name":"A & B","amount":1},{"name":"C","amount":2}]')],
)
def test_matrix_preserves_order_and_escapes_xml(format: Literal["csv", "json"], text: str) -> None:
    records = parse_matrix(text, format=format)
    results = hydrate_matrix("<item>{{ name }}:{{ amount }}</item>", records)
    assert results == (HydrationSuccess("<item>A &amp; B:1</item>"), HydrationSuccess("<item>C:2</item>"))


@pytest.mark.parametrize(
    ("format", "text"),
    [
        ("csv", ""),
        ("csv", "a,a\n1,2"),
        ("csv", "a,b\n1"),
        ("csv", "a\n1,2"),
        ("json", "{}"),
        ("json", "[1]"),
        ("json", "broken"),
    ],
)
def test_matrix_invalid_shape_is_explicit(format: Literal["csv", "json"], text: str) -> None:
    with pytest.raises(ValueError):
        parse_matrix(text, format=format)


def test_matrix_failure_is_per_record() -> None:
    results = hydrate_matrix("<item>{{ name }}</item>", [{"name": "A"}, {}])
    assert isinstance(results[0], HydrationSuccess)
    assert isinstance(results[1], HydrationFailure)
    assert parse_matrix("[]", format="json") == ()


def test_xpath_operations_are_ordered_pure_and_namespace_aware() -> None:
    xml = '<r xmlns="urn:example"><item>old</item><item>remove</item></r>'
    result = mutate_xml(
        xml,
        [
            XPathMutation("//n:item[1]", value="A & B"),
            XPathMutation("//n:item[1]", "set_attribute", "42", "id"),
            XPathMutation("//n:item[2]", "remove"),
        ],
        namespaces={"n": "urn:example"},
    )
    assert result == HydrationSuccess('<r xmlns="urn:example"><item id="42">A &amp; B</item></r>')
    assert "old" in xml


@pytest.mark.parametrize("xpath", ["//missing", "//item/text()", "count(//item)", "/r"])
def test_xpath_expected_failures(xpath: str) -> None:
    operation = "remove" if xpath == "/r" else "set_text"
    assert isinstance(mutate_xml("<r><item>x</item></r>", [XPathMutation(xpath, operation)]), HydrationFailure)


def test_invalid_xpath_and_schema_are_programming_errors() -> None:
    with pytest.raises(etree.XPathError):
        mutate_xml("<r/>", [XPathMutation("[broken")])
    with pytest.raises(etree.XMLSchemaParseError):
        validate_xml("<r/>", "<not-a-schema/>")
    with pytest.raises(ValueError, match="attribute"):
        XPathMutation("//r", "set_attribute")


def test_schema_external_resources_cannot_perform_io(tmp_path: Path) -> None:
    schema = tmp_path / "external.xsd"
    schema.write_text(SCHEMA, encoding="utf-8")
    include = (
        '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">'
        f'<xs:include schemaLocation="{schema.as_uri()}"/></xs:schema>'
    )
    with pytest.raises((ValueError, etree.XMLSchemaParseError), match="External XML resources"):
        validate_xml("<amount>3</amount>", include)


def test_schema_validates_inputs_without_mutation() -> None:
    assert validate_xml("<amount>3</amount>", SCHEMA) == HydrationSuccess("<amount>3</amount>")
    assert isinstance(validate_xml("<amount>wrong</amount>", SCHEMA), HydrationFailure)
    assert isinstance(validate_xml("bad XML", SCHEMA), HydrationFailure)
    assert isinstance(mutate_xml("bad XML", []), HydrationFailure)


def test_external_entities_are_not_resolved() -> None:
    xml = '<!DOCTYPE r [<!ENTITY secret SYSTEM "file:///does-not-exist">]><r>&secret;</r>'
    result = mutate_xml(xml, [])
    assert isinstance(result, HydrationSuccess)
    assert "&secret;" in result.xml


def test_pdf_extraction_formats(pdf_factory: Callable[[str], bytes]) -> None:
    pdf = pdf_factory("PDF")
    encoded = base64.b64encode(pdf).decode()
    assert extract_pdf(pdf) is pdf
    assert extract_pdf(encoded, format="base64") == pdf
    assert extract_pdf(f"<envelope><bytes>\n{encoded}\n</bytes></envelope>", format="xml") == pdf
    assert isinstance(extract_pdf("<r><bytes>x</bytes><bytes>y</bytes></r>", format="xml"), PdfExtractionFailure)


@pytest.mark.parametrize(
    ("data", "format"),
    [
        (b"invalid", "pdf"),
        ("invalid", "pdf"),
        ("invalid!", "base64"),
        (b"\xff", "xml"),
        ("<r/>", "xml"),
        ("bad XML", "xml"),
    ],
)
def test_pdf_extraction_failures(data: bytes | str, format: Literal["pdf", "base64", "xml"]) -> None:
    assert isinstance(extract_pdf(data, format=format), PdfExtractionFailure)

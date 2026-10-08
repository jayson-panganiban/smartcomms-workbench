from xml.etree import ElementTree

import pytest
from jinja2 import TemplateSyntaxError

from smartcomms_workbench.payload.hydrator import (
    HydrationFailure,
    HydrationSuccess,
    hydrate_xml,
)


def test_hydrate_xml_renders_well_formed_xml() -> None:
    result = hydrate_xml("<payload><name>{{ name }}</name></payload>", {"name": "Ada"})

    assert isinstance(result, HydrationSuccess)
    root = ElementTree.fromstring(result.xml)
    assert root.findtext("name") == "Ada"


def test_hydrate_xml_escapes_xml_special_characters_in_text_and_attributes() -> None:
    value = """A&B <person> "quoted" 'text'"""
    result = hydrate_xml(
        '<payload note="{{ value }}"><name>{{ value }}</name></payload>',
        {"value": value},
    )

    assert isinstance(result, HydrationSuccess)
    root = ElementTree.fromstring(result.xml)
    assert root.attrib["note"] == value
    assert root.findtext("name") == value


def test_hydrate_xml_preserves_unicode() -> None:
    value = "Māori 日本語 🙂"
    result = hydrate_xml("<payload>{{ value }}</payload>", {"value": value})

    assert isinstance(result, HydrationSuccess)
    assert ElementTree.fromstring(result.xml).text == value


def test_hydrate_xml_returns_failure_for_missing_variables() -> None:
    result = hydrate_xml("<payload>{{ customer.name }}</payload>", {})

    assert isinstance(result, HydrationFailure)
    assert result.code == "missing_variable"
    assert result.message


@pytest.mark.parametrize(
    "template_text",
    [
        "<payload><item></payload>",
        "<payload>{% if include_item %}<item>{% endif %}</payload>",
    ],
)
def test_hydrate_xml_returns_failure_for_malformed_rendered_xml(template_text: str) -> None:
    result = hydrate_xml(template_text, {"include_item": True})

    assert isinstance(result, HydrationFailure)
    assert result.code == "invalid_xml"
    assert result.message


def test_hydrate_xml_allows_jinja_control_flow_in_source_template() -> None:
    result = hydrate_xml(
        "<payload>{% if include_item %}<item>{{ value }}</item>{% endif %}</payload>",
        {"include_item": True, "value": "valid"},
    )

    assert isinstance(result, HydrationSuccess)
    assert ElementTree.fromstring(result.xml).findtext("item") == "valid"


def test_hydrate_xml_propagates_jinja_template_syntax_errors() -> None:
    with pytest.raises(TemplateSyntaxError):
        hydrate_xml("<payload>{% if value %}</payload>", {"value": True})


def test_hydrate_xml_does_not_convert_programming_errors_to_failures() -> None:
    class BrokenValue:
        def __str__(self) -> str:
            raise RuntimeError("rendering bug")

    with pytest.raises(RuntimeError, match="rendering bug"):
        hydrate_xml("<payload>{{ value }}</payload>", {"value": BrokenValue()})

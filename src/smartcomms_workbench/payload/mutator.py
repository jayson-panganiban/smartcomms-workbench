"""Pure XPath mutations and XML Schema validation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from lxml import etree

from smartcomms_workbench.payload.hydrator import HydrationFailure, HydrationResult, HydrationSuccess
from smartcomms_workbench.payload.xml import safe_parser


@dataclass(frozen=True)
class XPathMutation:
    xpath: str
    operation: Literal["set_text", "set_attribute", "remove"] = "set_text"
    value: str = ""
    attribute: str | None = None

    def __post_init__(self) -> None:
        if self.operation not in ("set_text", "set_attribute", "remove"):
            raise ValueError(f"Unknown mutation operation: {self.operation}")
        if self.operation == "set_attribute" and not self.attribute:
            raise ValueError("set_attribute requires an attribute name")


def mutate_xml(
    xml: str,
    mutations: Sequence[XPathMutation],
    *,
    namespaces: Mapping[str, str] | None = None,
) -> HydrationResult:
    """Apply mutations sequentially to a private tree; unmatched paths fail explicitly.

    Invalid XPath syntax is a configuration/programming error and propagates.
    Only element selections are supported, including namespaced elements.
    """
    try:
        root = etree.fromstring(xml.encode("utf-8"), parser=safe_parser())
    except etree.XMLSyntaxError as error:
        return HydrationFailure("invalid_xml", str(error))
    for mutation in mutations:
        nodes = root.xpath(mutation.xpath, namespaces=dict(namespaces or {}))
        if (
            not isinstance(nodes, list)
            or not nodes
            or any(not isinstance(node, etree._Element) or not isinstance(node.tag, str) for node in nodes)
        ):
            return HydrationFailure("invalid_xml", f"XPath must select elements: {mutation.xpath}")
        for node in nodes:
            if mutation.operation == "remove":
                parent = node.getparent()
                if parent is None:
                    return HydrationFailure("invalid_xml", "Cannot remove the root element")
                parent.remove(node)
            elif mutation.operation == "set_attribute":
                assert mutation.attribute is not None
                node.set(mutation.attribute, mutation.value)
            else:
                node.text = mutation.value
    return HydrationSuccess(etree.tostring(root, encoding="unicode"))


def validate_xml(xml: str, schema_xml: str) -> HydrationResult:
    """Validate supplied XML against supplied XSD; invalid schema syntax raises."""
    schema_root = etree.fromstring(schema_xml.encode("utf-8"), parser=safe_parser())
    for node in schema_root.iter():
        if (
            node.tag
            in {
                "{http://www.w3.org/2001/XMLSchema}include",
                "{http://www.w3.org/2001/XMLSchema}import",
                "{http://www.w3.org/2001/XMLSchema}redefine",
            }
            and node.get("schemaLocation") is not None
        ):
            raise etree.XMLSchemaParseError("External XML resources are not supported in schemas")
    schema = etree.XMLSchema(schema_root)
    try:
        root = etree.fromstring(xml.encode("utf-8"), parser=safe_parser())
    except etree.XMLSyntaxError as error:
        return HydrationFailure("invalid_xml", str(error))
    if not schema.validate(root):
        return HydrationFailure("invalid_xml", str(schema.error_log))
    return HydrationSuccess(xml)

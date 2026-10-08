"""Pure Jinja2 hydration of XML templates from one structured record."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal
from xml.etree import ElementTree

from jinja2 import Environment, StrictUndefined, UndefinedError

type HydrationFailureCode = Literal["missing_variable", "invalid_xml"]


@dataclass(frozen=True)
class HydrationSuccess:
    """Successfully rendered, well-formed XML."""

    xml: str


@dataclass(frozen=True)
class HydrationFailure:
    """An expected input validation failure."""

    code: HydrationFailureCode
    message: str


type HydrationResult = HydrationSuccess | HydrationFailure


def hydrate_xml(template_text: str, data: Mapping[str, object]) -> HydrationResult:
    """Render one record into XML, returning expected input validation failures."""
    environment = Environment(autoescape=True, undefined=StrictUndefined)

    try:
        rendered = environment.from_string(template_text).render(data)
    except UndefinedError as error:
        return HydrationFailure("missing_variable", str(error))

    try:
        ElementTree.fromstring(rendered)
    except ElementTree.ParseError as error:
        return HydrationFailure("invalid_xml", str(error))

    return HydrationSuccess(rendered)

"""XML parsers that cannot read external resources, including XSD includes."""

from lxml import etree


class _RejectExternalResources(etree.Resolver):
    def resolve(self, url: str | None, _public_id: str | None, _context: object, /) -> None:
        raise ValueError(f"External XML resources are not supported: {url}")


def safe_parser() -> etree.XMLParser:
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    parser.resolvers.add(_RejectExternalResources())
    return parser

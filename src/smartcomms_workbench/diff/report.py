"""Pure, self-contained side-by-side HTML report generation."""

import base64
from html import escape

from smartcomms_workbench.diff.pdf import PdfInputError, open_pdf
from smartcomms_workbench.diff.visual import ComparisonResult


def _images(pdf: bytes, dpi: int) -> list[str]:
    with open_pdf(pdf) as document:
        return [
            base64.b64encode(page.get_pixmap(dpi=dpi, alpha=False).tobytes("png")).decode("ascii") for page in document
        ]


def render_html_report(
    reference: bytes,
    candidate: bytes,
    result: ComparisonResult,
    *,
    title: str = "SmartComms PDF comparison",
    dpi: int = 72,
) -> str:
    """Embed raw page previews and computed outcomes; never read or write paths."""
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    panels: list[str] = []
    for label, pdf in (("Reference", reference), ("Candidate", candidate)):
        try:
            images = _images(pdf, dpi)
            content = "".join(
                f'<figure><figcaption>Page {index}</figcaption><img alt="{label} page {index}" '
                f'src="data:image/png;base64,{image}"></figure>'
                for index, image in enumerate(images, 1)
            )
        except PdfInputError as error:
            content = f"<p>{escape(str(error))}</p>"
        panels.append(f"<section><h2>{label}</h2>{content}</section>")
    rows = "".join(
        f"<tr><td>{page.page}</td><td>{page.outcome.value}</td>"
        f"<td>{page.metrics.changed_pixel_ratio if page.metrics else 'n/a'}</td><td>{page.ssim}</td></tr>"
        for page in result.pages
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>{escape(title)}</title>"
        "<style>body{font-family:sans-serif}main{display:flex;gap:1rem}"
        "section{width:50%}img{max-width:100%}td,th{padding:.5rem}</style>"
        f"<h1>{escape(title)}</h1><p>Outcome: {result.outcome.value}</p>"
        f"<p>{escape(result.error or '')}</p>"
        "<table><tr><th>Page</th><th>Outcome</th><th>Changed pixel ratio</th><th>SSIM</th></tr>"
        f"{rows}</table><main>{''.join(panels)}</main></html>"
    )

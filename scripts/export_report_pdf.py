"""Export ``notebooks/report.ipynb`` to ``report.pdf`` with code hidden.

The notebook is the report and the working document.  The body runs to the
``## Appendix`` heading; everything after it (contribution and AI statements,
data cleaning documentation, the detail exhibits and references) is rendered as
the appendix and excluded from the ten-page body limit.

The raw nbconvert lab template is too airy for a page-limited submission, so
this script renders a compact A4 print layout: code inputs removed, figures at
near full width (they are the report's storytelling), tables centred and kept
whole, "Takeaway" paragraphs as caption-sized text, and a forced page break
before the appendix.

Run with the notebook and browser extras::

    uv run --with playwright python scripts/export_report_pdf.py

Add ``--with pymupdf`` to print the body/appendix page split.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]

PRINT_CSS = """
<style>
@media print {
  html, body, .jp-Notebook, .jp-RenderedHTMLCommon, .jp-RenderedMarkdown,
  .jp-OutputArea-output, p, li {
    font-size: 11px !important;
    line-height: 1.34 !important;
  }
  body { background: #ffffff !important; }
  /* Any stray stream/code text stays small. */
  pre, code, .jp-RenderedText pre { font-size: 9px !important; line-height: 1.34 !important; }
  .jp-Notebook { padding: 0 !important; }
  .jp-OutputPrompt, .jp-InputPrompt { display: none !important; }
  .jp-Cell-outputWrapper, .jp-OutputArea { margin-left: 0 !important; padding-left: 0 !important; }
  .jp-Cell { margin: 0 !important; padding: 0 !important; }
  .jp-RenderedHTMLCommon p, .jp-RenderedMarkdown p,
  .jp-RenderedHTMLCommon li { margin: 5px 0 !important; }
  h1 { font-size: 1.55em !important; margin: 0 0 3px !important; }
  h2 { font-size: 1.3em !important; margin: 20px 0 4px !important; }
  h3 { font-size: 1.12em !important; margin: 12px 0 3px !important; }
  h1, h2, h3 { break-after: avoid-page !important; page-break-after: avoid !important; }
  /* Centre the four title lines on the first page. */
  main > .jp-Cell:first-of-type h1,
  main > .jp-Cell:first-of-type h3,
  main > .jp-Cell:first-of-type p:nth-of-type(1) { text-align: center !important; }
  img { width: 98% !important; height: auto !important; }
  table, table td, table th { font-size: 8px !important; }
  table {
    margin-left: auto !important;
    margin-right: auto !important;
    break-inside: avoid !important;
  }
  .jp-OutputArea-child { break-inside: avoid !important; }
  .keep-together { break-inside: avoid !important; }
  /* Captions sit with their exhibit. */
  .jp-Cell:has(.takeaway) { margin-top: -3px !important; }
  .takeaway {
    font-size: 0.9em !important;
    line-height: 1.4 !important;
    color: #4a4a4a !important;
    margin: 1px 0 8px !important;
  }
  .takeaway strong { font-size: 0.9em !important; }
  ul, ol { margin: 2px 0 !important; padding-left: 18px !important; }
  /* The appendix is excluded from the ten-page body limit. */
  h2#Appendix { break-before: page !important; }
}
</style>
"""


def ordered_notebook(notebook: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Move ``appendix``-tagged cells behind the ``## Appendix`` heading."""
    cells = notebook.cells
    appendix_start = next(
        index
        for index, cell in enumerate(cells)
        if cell.cell_type == "markdown" and cell.source.lstrip().startswith("## Appendix")
    )

    def is_appendix(cell: nbformat.NotebookNode) -> bool:
        return "appendix" in (cell.metadata.get("tags") or [])

    body = [cell for cell in cells[:appendix_start] if not is_appendix(cell)]
    moved = [cell for cell in cells[:appendix_start] if is_appendix(cell)]
    heading = nbformat.v4.new_markdown_cell(
        "## Appendix\n\nSupplementary exhibits and technical documentation; "
        "the main report refers to these by letter."
    )
    tail = copy.deepcopy(list(cells[appendix_start:]))
    tail[0].source = re.sub(r"^## Appendix\s*\n+", "", tail[0].source)

    ordered = copy.deepcopy(notebook)
    ordered.cells = copy.deepcopy(body) + [heading] + copy.deepcopy(moved) + tail
    return ordered



def _split_cells(html: str) -> list[tuple[int, int]]:
    """Bounds of each top-level ``jp-Cell`` div, matched by depth counting."""
    starts = [match.start() for match in re.finditer(r'<div class="jp-Cell ', html)]
    token = re.compile(r"<div\b|</div>")
    bounds = []
    for start in starts:
        depth = 0
        position = start
        while True:
            match = token.search(html, position)
            if match is None:
                raise ValueError("unbalanced cell markup")
            position = match.end()
            depth += 1 if match.group(0) == "<div" else -1
            if depth == 0:
                bounds.append((start, position))
                break
    return bounds


def group_headings_with_exhibits(html: str) -> str:
    """Wrap each heading with its exhibit and caption so they page together."""
    bounds = _split_cells(html)
    if not bounds:
        return html
    texts = [html[start:end] for start, end in bounds]
    total = len(texts)
    groups: list[tuple[int, int]] = []
    index = 0
    while index < total:
        text = texts[index]
        plain = re.sub(r"<[^>]+>", "", text)
        heading = (
            "jp-MarkdownCell" in text
            and re.search(r"<h[23][ >]", text)
            and len(plain.split()) <= 12
        )
        if heading and index + 1 < total:
            last = index + 1
            while last < total:
                cell = texts[last]
                if (
                    last > index + 1
                    and "jp-MarkdownCell" in cell
                    and re.search(r"<h[23][ >]", cell)
                ):
                    break
                last += 1
                if "jp-OutputArea" in cell:
                    break
            if last < total and "Takeaway." in texts[last]:
                last += 1
            groups.append((index, last - 1))
            index = last
        elif (
            "jp-OutputArea" in text
            and index + 1 < total
            and ("Takeaway." in texts[index + 1] or "Verdict." in texts[index + 1])
        ):
            groups.append((index, index + 1))
            index += 2
        else:
            groups.append((index, index))
            index += 1

    output: list[str] = []
    cursor = 0
    for first, last in groups:
        if last > first:
            start, end = bounds[first][0], bounds[last][1]
            output.append(html[cursor:start])
            output.append('<div class="keep-together">\n')
            output.append(html[start:end])
            output.append("\n</div>\n")
            cursor = end
    output.append(html[cursor:])
    return "".join(output)


def load_notebook(path: Path, execute: bool) -> nbformat.NotebookNode:
    """Read the notebook, optionally running it top to bottom first."""
    node = nbformat.read(path, as_version=4)
    if execute:
        ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(
            node, {"metadata": {"path": str(path.parent)}}
        )
        print(f"executed {path} (in memory; the notebook file is left unchanged)")
    return node


def build_html(notebook: Path, destination: Path, execute: bool = False) -> None:
    exporter = HTMLExporter()
    exporter.exclude_input = True
    html, _ = exporter.from_notebook_node(
        ordered_notebook(load_notebook(notebook, execute))
    )
    html = html.replace(
        "<p><strong>Takeaway.</strong>",
        '<p class="takeaway"><strong>Takeaway.</strong>',
    )
    html = group_headings_with_exhibits(html)
    destination.write_text(html.replace("</head>", PRINT_CSS + "</head>", 1))


def render_pdf(source: Path, destination: Path) -> None:
    import asyncio

    from playwright.async_api import async_playwright

    async def launch_browser(playwright):
        try:
            return await playwright.chromium.launch(channel="chrome", headless=True)
        except Exception:
            try:
                return await playwright.chromium.launch(headless=True)
            except Exception as error:
                raise RuntimeError(
                    "No Chromium available: install Google Chrome or run "
                    "`uv run playwright install chromium`."
                ) from error

    async def run() -> None:
        async with async_playwright() as playwright:
            browser = await launch_browser(playwright)
            page = await browser.new_page()
            await page.goto(source.resolve().as_uri(), wait_until="networkidle", timeout=180_000)
            try:
                await page.wait_for_function(
                    "typeof MathJax !== 'undefined' && MathJax.Hub && "
                    "MathJax.Hub.queue && MathJax.Hub.queue.pending === 0",
                    timeout=30_000,
                )
            except Exception:
                await page.wait_for_timeout(6_000)
            await page.pdf(
                path=str(destination),
                format="A4",
                print_background=True,
                margin={"top": "10mm", "bottom": "11mm", "left": "10mm", "right": "10mm"},
                display_header_footer=False,
            )
            await browser.close()

    asyncio.run(run())


def report(destination: Path) -> None:
    try:
        import pymupdf
    except ImportError:
        print("install pymupdf to see the page split")
        return
    document = pymupdf.open(destination)
    # The appendix heading is the last standalone "Appendix" line (contents list
    # on page 1 also contains one).
    heading_pages = [
        index + 1
        for index, page in enumerate(document)
        if any(line.strip() == "Appendix" for line in page.get_text().splitlines())
    ]
    print(f"pages: {document.page_count}")
    if heading_pages:
        appendix = heading_pages[-1]
        print(f"body: pages 1-{appendix - 1} (limit 10); appendix starts on page {appendix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks" / "report.ipynb")
    parser.add_argument("--pdf", type=Path, default=ROOT / "report.pdf")
    parser.add_argument("--html", type=Path, default=ROOT / "outputs" / "report_print.html")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the notebook top to bottom before exporting (regenerates outputs)",
    )
    args = parser.parse_args()

    args.html.parent.mkdir(parents=True, exist_ok=True)
    build_html(args.notebook, args.html, execute=args.execute)
    render_pdf(args.html, args.pdf)
    report(args.pdf)


if __name__ == "__main__":
    sys.exit(main())

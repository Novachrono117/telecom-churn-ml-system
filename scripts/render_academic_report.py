"""Phase 14A: render the academic report to print-ready HTML, and to PDF if possible.

Usage
-----
::

    uv run python scripts/render_academic_report.py
    uv run python scripts/render_academic_report.py --html-only

Why the Markdown converter is written here
------------------------------------------
The root environment is part of the frozen provenance: `pyproject.toml` and `uv.lock`
describe the environment every recorded metric was produced in, and adding a Markdown
library to it so a report can be typeset would change that environment for a reason
that has nothing to do with the model.

So this module converts the subset of Markdown the report actually uses — headings,
tables, lists, fenced code, block quotes, images, emphasis, rules — in about a hundred
lines and with no import outside the standard library. It is not a general Markdown
implementation and does not try to be.

Why the PDF comes from a browser
--------------------------------
No PDF toolchain is installed (`pandoc`, `wkhtmltopdf`, `weasyprint` and `prince` are
all absent), and downloading one would mean pulling an arbitrary binary into a project
whose whole argument is provenance. Chrome and Edge are already present on this
machine, and both can print a local file to PDF headlessly. That uses software the
operator already trusts and adds nothing to the repository.

If neither browser is found, the HTML is still written and the script reports
``REPORT_PDF_EXTERNAL_RENDER_REQUIRED`` rather than pretending a PDF exists.

Draft, not final
----------------
While the report still carries ``[[EXTERNAL_INPUT_REQUIRED: …]]`` markers, the PDF is
named ``academic_report_DRAFT.pdf``. A document with unfilled cover fields is not a
submission, and naming it as one would be the whole problem this phase exists to
avoid.
"""

from __future__ import annotations

import argparse
import html
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from churn.academic.results import EXTERNAL_INPUT_MARKER
from churn.config import PROJECT_ROOT

logger = logging.getLogger("render_academic_report")

SOURCE = PROJECT_ROOT / "reports/academic/academic_report.md"
HTML_OUTPUT = PROJECT_ROOT / "reports/academic/academic_report.html"
DRAFT_PDF_OUTPUT = PROJECT_ROOT / "reports/academic/academic_report_DRAFT.pdf"
FINAL_PDF_OUTPUT = PROJECT_ROOT / "reports/academic/academic_report.pdf"

#: Browsers that can print a local file to PDF headlessly, in preference order.
BROWSER_CANDIDATES: tuple[str, ...] = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)

STYLESHEET = """
@page { size: A4; margin: 18mm 16mm; }
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.55; color: #1a1a1a; background: #fff;
  max-width: 190mm; margin: 0 auto; padding: 0 2mm;
}
h1 {
  font-size: 19pt; margin: 26px 0 12px; padding-bottom: 6px;
  border-bottom: 2px solid #2c3e50; color: #16232e; page-break-after: avoid;
}
h1:first-of-type { margin-top: 0; }
h2 { font-size: 14.5pt; margin: 20px 0 9px; color: #22303c; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 16px 0 7px; color: #2c3e50; page-break-after: avoid; }
p { margin: 8px 0; text-align: justify; }
ul, ol { margin: 8px 0 8px 22px; padding: 0; }
li { margin: 3px 0; }
a { color: #14507a; text-decoration: none; }
code {
  font-family: "Cascadia Mono", Consolas, "Courier New", monospace;
  font-size: 0.87em; background: #f2f4f6; padding: 1px 4px;
  border-radius: 3px; border: 1px solid #e2e6ea;
}
pre {
  background: #f7f9fa; border: 1px solid #dfe4e8; border-left: 3px solid #56718a;
  border-radius: 4px; padding: 9px 12px; overflow-x: auto;
  font-size: 8.6pt; line-height: 1.42; page-break-inside: avoid;
}
pre code { background: none; border: none; padding: 0; font-size: inherit; }
blockquote {
  margin: 11px 0; padding: 8px 14px; background: #f6f8fa;
  border-left: 3px solid #7b8f9e; color: #33414d; page-break-inside: avoid;
}
blockquote p { margin: 4px 0; }
table {
  border-collapse: collapse; width: 100%; margin: 11px 0;
  font-size: 9.2pt; page-break-inside: avoid;
}
th, td { border: 1px solid #cfd6dc; padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #eef2f5; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
img { max-width: 100%; height: auto; display: block; margin: 12px auto; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #d7dde2; margin: 20px 0; }
strong { font-weight: 650; color: #101a22; }
.draft-banner {
  background: #fff4e5; border: 2px solid #d98324; border-radius: 5px;
  padding: 10px 14px; margin: 0 0 18px; font-weight: 600; color: #7a4708;
}
div[align="center"] { text-align: center; }
div[align="center"] p { text-align: center; }
"""

_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?!\*)")
_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _inline(text: str) -> str:
    """Render inline Markdown, escaping everything that is not a recognised construct.

    Code spans are extracted first and reinserted last, so that a ``**`` inside
    backticks is shown rather than interpreted.
    """
    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        spans.append(match.group(1))
        return f"\x00{len(spans) - 1}\x00"

    text = _INLINE_CODE.sub(stash, text)
    text = html.escape(text, quote=False)
    text = _IMAGE.sub(lambda m: f'<img src="{m.group(2)}" alt="{m.group(1)}">', text)
    text = _LINK.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', text)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _ITALIC.sub(r"<em>\1</em>", text)
    for index, span in enumerate(spans):
        text = text.replace(f"\x00{index}\x00", f"<code>{html.escape(span, quote=False)}</code>")
    return text


def _table(rows: list[str]) -> str:
    """Render a pipe table. The second row is the alignment rule and is dropped."""

    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    header = cells(rows[0])
    body = [cells(row) for row in rows[2:]]
    out = ["<table>", "<thead><tr>"]
    out += [f"<th>{_inline(cell)}</th>" for cell in header]
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in row) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def markdown_to_html(text: str) -> str:
    """Convert the Markdown subset this report uses into HTML."""
    lines = text.split("\n")
    out: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        # Raw HTML passthrough (the cover block uses <div align="center">).
        if stripped.startswith("<") and not stripped.startswith("<!--"):
            out.append(stripped)
            index += 1
            continue

        # Fenced code.
        if stripped.startswith("```"):
            index += 1
            block: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            out.append("<pre><code>" + html.escape("\n".join(block), quote=False) + "</code></pre>")
            continue

        # Horizontal rule.
        if set(stripped) == {"-"} and len(stripped) >= 3:
            out.append("<hr>")
            index += 1
            continue

        # Heading.
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            out.append(f"<h{level}>{_inline(stripped[level:].strip())}</h{level}>")
            index += 1
            continue

        # Table: a pipe line followed by an alignment rule.
        if (
            stripped.startswith("|")
            and index + 1 < len(lines)
            and set(lines[index + 1].strip()) <= set("|-: ")
            and "-" in lines[index + 1]
        ):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(lines[index])
                index += 1
            out.append(_table(rows))
            continue

        # Block quote.
        if stripped.startswith(">"):
            quoted: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quoted.append(lines[index].strip().lstrip(">").strip())
                index += 1
            paragraphs = "\n".join(quoted).split("\n\n")
            body = "".join(
                f"<p>{_inline(' '.join(p.split()))}</p>" for p in paragraphs if p.strip()
            )
            out.append(f"<blockquote>{body}</blockquote>")
            continue

        # Lists.
        bullet = re.match(r"^([*+-]|\d+\.)\s+", stripped)
        if bullet:
            ordered = bullet.group(1)[0].isdigit()
            tag = "ol" if ordered else "ul"
            items: list[str] = []
            while index < len(lines):
                current = lines[index].strip()
                marker = re.match(r"^([*+-]|\d+\.)\s+", current)
                if marker:
                    items.append(current[marker.end() :])
                elif current and lines[index].startswith(("  ", "\t")) and items:
                    items[-1] += " " + current
                else:
                    break
                index += 1
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(i)}</li>" for i in items) + f"</{tag}>")
            continue

        # Paragraph: consume until a blank line or a block construct.
        paragraph: list[str] = []
        while index < len(lines):
            current = lines[index].strip()
            if not current or current.startswith(("#", ">", "|", "```", "<")):
                break
            if set(current) == {"-"} and len(current) >= 3:
                break
            if re.match(r"^([*+-]|\d+\.)\s+", current):
                break
            paragraph.append(current)
            index += 1
        if paragraph:
            out.append(f"<p>{_inline(' '.join(paragraph))}</p>")

    return "\n".join(out)


def build_html(markdown_text: str, is_draft: bool) -> str:
    """Wrap the converted body in a standalone, print-ready document."""
    banner = (
        '<div class="draft-banner">RASCUNHO — este documento contém campos de capa e '
        "links de entrega pendentes de ação humana. Não é a versão final de entrega.</div>"
        if is_draft
        else ""
    )
    return (
        "<!doctype html>\n"
        '<html lang="pt-BR">\n<head>\n<meta charset="utf-8">\n'
        "<title>Predição de Churn em Telecomunicações — Relatório Acadêmico</title>\n"
        f"<style>{STYLESHEET}</style>\n</head>\n<body>\n"
        f"{banner}\n{markdown_to_html(markdown_text)}\n</body>\n</html>\n"
    )


def find_browser() -> Path | None:
    """Return a browser able to print to PDF headlessly, or None."""
    for candidate in BROWSER_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path
    for name in ("chrome", "chromium", "msedge"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


#: The fixed instant written into the PDF in place of the wall clock. Same length as
#: the value Chrome emits, so byte offsets in the cross-reference table stay valid.
FIXED_PDF_DATE = b"D:19700101000000+00'00'"
_PDF_DATE = re.compile(rb"D:\d{14}\+\d{2}'\d{2}'")


def _strip_pdf_timestamps(pdf: Path) -> int:
    """Replace the browser's ``/CreationDate`` and ``/ModDate`` with a fixed instant.

    Chrome stamps the wall clock into every PDF it prints, which would make this
    artefact differ on every run and turn a re-render into a spurious diff — exactly
    what the project's determinism convention exists to prevent. The substitute is the
    same length as the value it replaces, so the byte offsets recorded in the
    cross-reference table remain correct and the file stays valid.

    Returns:
        How many timestamps were rewritten.
    """
    data = pdf.read_bytes()
    replaced, count = _PDF_DATE.subn(FIXED_PDF_DATE, data)
    if count:
        pdf.write_bytes(replaced)
    return count


def render_pdf(source_html: Path, destination: Path) -> bool:
    """Print ``source_html`` to ``destination``. Returns whether a PDF was produced."""
    browser = find_browser()
    if browser is None:
        logger.warning("REPORT_PDF_EXTERNAL_RENDER_REQUIRED: no headless browser found.")
        return False

    with tempfile.TemporaryDirectory() as profile:
        command = [
            str(browser),
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={destination}",
            source_html.resolve().as_uri(),
        ]
        result = subprocess.run(command, capture_output=True, timeout=300, check=False)

    if not destination.is_file() or destination.stat().st_size == 0:
        logger.warning(
            "REPORT_PDF_EXTERNAL_RENDER_REQUIRED: %s produced no PDF (exit %s). %s",
            browser.name,
            result.returncode,
            result.stderr.decode("utf-8", "replace")[:400],
        )
        return False

    normalised = _strip_pdf_timestamps(destination)
    logger.info(
        "Rendered PDF via %s: %s (%d bytes, %d timestamp(s) normalised)",
        browser.name,
        destination,
        destination.stat().st_size,
        normalised,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-only", action="store_true", help="Skip PDF rendering.")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    markdown_text = SOURCE.read_text(encoding="utf-8")
    marker_count = markdown_text.count(EXTERNAL_INPUT_MARKER)
    is_draft = marker_count > 0

    HTML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUTPUT.write_text(build_html(markdown_text, is_draft), encoding="utf-8", newline="\n")
    logger.info("Wrote HTML: %s (%d bytes)", HTML_OUTPUT, HTML_OUTPUT.stat().st_size)

    if is_draft:
        logger.warning(
            "%d %s marker(s) remain: the PDF is a DRAFT and must not be submitted.",
            marker_count,
            EXTERNAL_INPUT_MARKER,
        )

    if arguments.html_only:
        return 0

    destination = DRAFT_PDF_OUTPUT if is_draft else FINAL_PDF_OUTPUT
    if render_pdf(HTML_OUTPUT, destination):
        return 0

    logger.error(
        "The HTML was written and is print-ready. Produce the PDF by opening it in a "
        "browser and printing to PDF (A4, background graphics enabled)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Very small HTML rendering for bench reports."""

from __future__ import annotations

from collections.abc import Iterable
from html import escape
from pathlib import Path

from .markdown import render_compare_markdown, render_report_markdown

_BASE_CSS = """
body {font-family: -apple-system, system-ui, "Segoe UI", sans-serif; max-width: 960px; margin: 2rem auto; color: #222;}
h1 {font-size: 1.6rem;}
h2 {font-size: 1.2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.2rem;}
table {border-collapse: collapse; width: 100%; margin: 0.5rem 0 1rem;}
th, td {border: 1px solid #ddd; padding: 0.3rem 0.6rem; text-align: left; font-size: 0.92rem;}
th {background: #f6f6f6;}
code, pre {background: #f3f3f3; padding: 0 0.2rem; border-radius: 3px;}
pre {padding: 0.5rem; overflow-x: auto;}
"""


def _md_to_html(markdown_text: str) -> str:
    """Tiny Markdown subset → HTML. Handles headings, bullets, tables, code fences."""
    lines = markdown_text.splitlines()
    out: list[str] = []
    in_table = False
    in_code = False
    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            if in_code:
                out.append("</pre>")
                in_code = False
            else:
                out.append("<pre><code>")
                in_code = True
            continue
        if in_code:
            out.append(escape(line))
            continue
        if line.startswith("# "):
            out.append(f"<h1>{escape(line[2:])}</h1>")
            continue
        if line.startswith("## "):
            out.append(f"<h2>{escape(line[3:])}</h2>")
            continue
        if line.startswith("### "):
            out.append(f"<h3>{escape(line[4:])}</h3>")
            continue
        if line.startswith("- "):
            out.append(f"<li>{escape(line[2:])}</li>")
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue  # markdown alignment row
            tag = "th" if not in_table else "td"
            if not in_table:
                out.append("<table><thead><tr>")
                out.extend(f"<{tag}>{escape(c)}</{tag}>" for c in cells)
                out.append("</tr></thead><tbody>")
                in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</tbody></table>")
            in_table = False
        if not line:
            out.append("<p></p>")
            continue
        out.append(f"<p>{escape(line)}</p>")
    if in_table:
        out.append("</tbody></table>")
    if in_code:
        out.append("</pre>")
    return "\n".join(out)


def _wrap_document(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        f'<html><head><meta charset="utf-8"><title>{escape(title)}</title>'
        f"<style>{_BASE_CSS}</style></head><body>\n"
        f"{body}\n</body></html>\n"
    )


def render_report_html(result_paths: Iterable[Path]) -> str:
    md = render_report_markdown(result_paths)
    return _wrap_document("canifinetune benchmark report", _md_to_html(md))


def render_compare_html(result_paths: Iterable[Path]) -> str:
    md = render_compare_markdown(result_paths)
    return _wrap_document("canifinetune benchmark comparison", _md_to_html(md))

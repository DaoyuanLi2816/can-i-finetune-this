"""Render benchmark results into Markdown / HTML reports and comparison tables."""

from .html import render_compare_html, render_report_html
from .markdown import render_compare_markdown, render_report_markdown

__all__ = [
    "render_compare_markdown",
    "render_report_markdown",
    "render_compare_html",
    "render_report_html",
]

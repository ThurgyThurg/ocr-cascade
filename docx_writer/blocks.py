"""
The block model: a small set of typed objects that describe what should
appear in the document, independent of the OOXML representation. The
heuristic page renderer in `__init__.py` builds blocks from OCR text;
`document.py` translates blocks to OOXML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TextRun:
    """One contiguous string of text with optional run-level formatting."""
    text: str
    strike: bool = False
    comment_id: Optional[int] = None  # if set, run is wrapped in comment range


@dataclass
class ParagraphBlock:
    runs: List[TextRun] = field(default_factory=list)
    alignment: Optional[str] = None  # None | "left" | "center" | "right"


@dataclass
class TableBlock:
    """Simple m×n table with plain string cell content (no nested formatting
    needed for the case-caption layout we use this for today)."""
    rows: List[List[str]]
    column_widths_twips: Optional[List[int]] = None  # default: equal split


@dataclass
class PageBreakBlock:
    """Hard page break between OCR pages."""
    pass


Block = "ParagraphBlock | TableBlock | PageBreakBlock"

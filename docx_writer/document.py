"""
Build `word/document.xml` from a list of Block objects.

Output structure:
    <w:document>
      <w:body>
        ...block-emitted elements...
        <w:sectPr/>
      </w:body>
    </w:document>
"""

from __future__ import annotations

from typing import Iterable, List

from lxml import etree

from .blocks import Block, ParagraphBlock, TableBlock, PageBreakBlock, TextRun
from .runs import (
    W_,
    make_text_run,
    make_break_run,
    make_comment_reference_run,
    make_comment_range_start,
    make_comment_range_end,
)
from .template import W_NS


def build_document_xml(blocks: Iterable[Block]) -> bytes:
    nsmap = {"w": W_NS}
    doc = etree.Element(W_("document"), nsmap=nsmap)
    body = etree.SubElement(doc, W_("body"))

    for block in blocks:
        if isinstance(block, ParagraphBlock):
            body.append(_paragraph_element(block))
        elif isinstance(block, TableBlock):
            body.append(_table_element(block))
            # A trailing empty paragraph is required after a table or Word
            # gets unhappy parsing the section properties.
            body.append(_empty_paragraph())
        elif isinstance(block, PageBreakBlock):
            body.append(_page_break_paragraph())
        else:
            raise TypeError(f"Unknown block type: {type(block).__name__}")

    sect_pr = etree.SubElement(body, W_("sectPr"))
    pg_sz = etree.SubElement(sect_pr, W_("pgSz"))
    pg_sz.set(W_("w"), "12240")   # US Letter in twips
    pg_sz.set(W_("h"), "15840")
    pg_mar = etree.SubElement(sect_pr, W_("pgMar"))
    pg_mar.set(W_("top"), "1440")
    pg_mar.set(W_("right"), "1440")
    pg_mar.set(W_("bottom"), "1440")
    pg_mar.set(W_("left"), "1440")

    return (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        + etree.tostring(doc, xml_declaration=False)
    )


def _paragraph_element(block: ParagraphBlock) -> etree._Element:
    p = etree.Element(W_("p"))
    if block.alignment:
        ppr = etree.SubElement(p, W_("pPr"))
        jc = etree.SubElement(ppr, W_("jc"))
        jc.set(W_("val"), block.alignment)

    for run in block.runs:
        _append_run_with_comment(p, run)
    return p


def _append_run_with_comment(parent: etree._Element, run: TextRun) -> None:
    """Append a TextRun's elements to `parent`. If the run has a comment_id,
    wrap it in commentRangeStart/End markers and a commentReference."""
    if run.comment_id is not None:
        parent.append(make_comment_range_start(run.comment_id))
    parent.append(make_text_run(run.text, strike=run.strike))
    if run.comment_id is not None:
        parent.append(make_comment_range_end(run.comment_id))
        parent.append(make_comment_reference_run(run.comment_id))


def _table_element(block: TableBlock) -> etree._Element:
    n_cols = max(len(row) for row in block.rows) if block.rows else 0
    if block.column_widths_twips and len(block.column_widths_twips) == n_cols:
        widths = list(block.column_widths_twips)
    else:
        widths = [4500] * n_cols  # half-page each

    tbl = etree.Element(W_("tbl"))

    tbl_pr = etree.SubElement(tbl, W_("tblPr"))
    tbl_w = etree.SubElement(tbl_pr, W_("tblW"))
    tbl_w.set(W_("w"), "0")
    tbl_w.set(W_("type"), "auto")
    jc = etree.SubElement(tbl_pr, W_("jc"))
    jc.set(W_("val"), "center")

    tbl_grid = etree.SubElement(tbl, W_("tblGrid"))
    for w in widths:
        gc = etree.SubElement(tbl_grid, W_("gridCol"))
        gc.set(W_("w"), str(w))

    for row in block.rows:
        tr = etree.SubElement(tbl, W_("tr"))
        for i in range(n_cols):
            cell_text = row[i] if i < len(row) else ""
            tc = etree.SubElement(tr, W_("tc"))
            tc_pr = etree.SubElement(tc, W_("tcPr"))
            tc_w = etree.SubElement(tc_pr, W_("tcW"))
            tc_w.set(W_("w"), str(widths[i]))
            tc_w.set(W_("type"), "dxa")
            p = etree.SubElement(tc, W_("p"))
            p.append(make_text_run(cell_text))
    return tbl


def _empty_paragraph() -> etree._Element:
    return etree.Element(W_("p"))


def _page_break_paragraph() -> etree._Element:
    p = etree.Element(W_("p"))
    p.append(make_break_run("page"))
    return p

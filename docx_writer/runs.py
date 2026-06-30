"""
Helpers for building `<w:r>` (text run) elements with optional run-properties
like strikethrough and comment-reference markers.

These return lxml Element objects ready to be appended to a `<w:p>` parent.
"""

from __future__ import annotations

from typing import Optional, Sequence

from lxml import etree

from .template import W_NS

W = f"{{{W_NS}}}"
NSMAP = {"w": W_NS}


def W_(tag: str):
    """Convenience: clark-style name in the w: namespace."""
    return f"{W}{tag}"


def make_text_run(
    text: str,
    *,
    strike: bool = False,
    style_id: Optional[str] = None,
) -> etree._Element:
    """Build a `<w:r>` carrying one `<w:t>` and optional run-properties."""
    r = etree.Element(W_("r"))
    rpr = _build_rpr(strike=strike, style_id=style_id)
    if rpr is not None:
        r.append(rpr)
    t = etree.SubElement(r, W_("t"))
    t.set(f"{{http://www.w3.org/XML/1998/namespace}}space", "preserve")
    t.text = text
    return r


def make_break_run(break_type: str = "page") -> etree._Element:
    """Build a `<w:r>` containing a `<w:br>` (line/page break)."""
    r = etree.Element(W_("r"))
    br = etree.SubElement(r, W_("br"))
    br.set(W_("type"), break_type)
    return r


def make_comment_reference_run(comment_id: int) -> etree._Element:
    """Build the small superscript-style mark that follows a commented range
    in document.xml: `<w:r><w:rPr><w:rStyle val="CommentReference"/></w:rPr>
    <w:commentReference w:id="N"/></w:r>`."""
    r = etree.Element(W_("r"))
    rpr = etree.SubElement(r, W_("rPr"))
    rstyle = etree.SubElement(rpr, W_("rStyle"))
    rstyle.set(W_("val"), "CommentReference")
    ref = etree.SubElement(r, W_("commentReference"))
    ref.set(W_("id"), str(comment_id))
    return r


def make_comment_range_start(comment_id: int) -> etree._Element:
    el = etree.Element(W_("commentRangeStart"))
    el.set(W_("id"), str(comment_id))
    return el


def make_comment_range_end(comment_id: int) -> etree._Element:
    el = etree.Element(W_("commentRangeEnd"))
    el.set(W_("id"), str(comment_id))
    return el


def _build_rpr(*, strike: bool, style_id: Optional[str]) -> Optional[etree._Element]:
    if not strike and not style_id:
        return None
    rpr = etree.Element(W_("rPr"))
    if style_id:
        rs = etree.SubElement(rpr, W_("rStyle"))
        rs.set(W_("val"), style_id)
    if strike:
        s = etree.SubElement(rpr, W_("strike"))
        s.set(W_("val"), "true")
    return rpr

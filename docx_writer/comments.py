"""
Build `word/comments.xml` from a list of comment payloads.

Each comment has: id (int), author, initials, date (ISO 8601), text.
The comment text is rendered as a single paragraph.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from lxml import etree

from .template import W_NS
from .runs import W_, make_text_run


@dataclass
class Comment:
    id: int
    text: str
    author: str = "[handwritten]"
    initials: str = "HW"
    date: Optional[str] = None  # ISO 8601; default = utcnow at build time


def build_comments_xml(comments: Iterable[Comment]) -> bytes:
    nsmap = {"w": W_NS}
    root = etree.Element(W_("comments"), nsmap=nsmap)

    default_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for c in comments:
        cmt = etree.SubElement(root, W_("comment"))
        cmt.set(W_("id"), str(c.id))
        cmt.set(W_("author"), c.author)
        cmt.set(W_("initials"), c.initials)
        cmt.set(W_("date"), c.date or default_date)

        p = etree.SubElement(cmt, W_("p"))
        # Comment paragraphs need at least one run for Word to render them
        # cleanly; multi-line comment text is split on newlines into several
        # paragraphs.
        lines = (c.text or "").split("\n") if c.text else [""]
        first = True
        for line in lines:
            if first:
                p.append(make_text_run(line))
                first = False
            else:
                p = etree.SubElement(cmt, W_("p"))
                p.append(make_text_run(line))

    return (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        + etree.tostring(root, xml_declaration=False)
    )

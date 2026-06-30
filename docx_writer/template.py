"""
Static OOXML parts that go into every .docx we emit.

These are byte-for-byte constants — they don't depend on the document
content. The dynamic parts (`word/document.xml` and `word/comments.xml`)
are built per-document by `document.py` and `comments.py`.
"""

from __future__ import annotations


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
PR_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

CT_COMMENTS = "application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"
CT_DOCUMENT = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
CT_STYLES = "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"

REL_COMMENTS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
REL_DOCUMENT = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
REL_STYLES = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"


def content_types_xml(*, include_comments: bool = False) -> bytes:
    overrides = [
        f'<Override PartName="/word/document.xml" ContentType="{CT_DOCUMENT}"/>',
        f'<Override PartName="/word/styles.xml" ContentType="{CT_STYLES}"/>',
    ]
    if include_comments:
        overrides.append(
            f'<Override PartName="/word/comments.xml" ContentType="{CT_COMMENTS}"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Types xmlns="{CT_NS}">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(overrides)
        + '</Types>'
    ).encode("utf-8")


def root_rels_xml() -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{PR_NS}">'
        f'<Relationship Id="rId1" Type="{REL_DOCUMENT}" Target="word/document.xml"/>'
        '</Relationships>'
    ).encode("utf-8")


def document_rels_xml(*, include_comments: bool = False) -> bytes:
    rels = [
        f'<Relationship Id="rId1" Type="{REL_STYLES}" Target="styles.xml"/>',
    ]
    if include_comments:
        rels.append(
            f'<Relationship Id="rId2" Type="{REL_COMMENTS}" Target="comments.xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{PR_NS}">'
        + "".join(rels)
        + '</Relationships>'
    ).encode("utf-8")


def styles_xml() -> bytes:
    # Minimal styles part:
    #   - default Calibri 11pt body text (matches the python-docx default we had)
    #   - Normal paragraph style (the default body style)
    #   - CommentReference character style (referenced by w:commentReference markers)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:styles xmlns:w="{W_NS}">'
        '<w:docDefaults>'
        '<w:rPrDefault><w:rPr>'
        '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>'
        '<w:sz w:val="22"/>'
        '<w:szCs w:val="22"/>'
        '</w:rPr></w:rPrDefault>'
        '<w:pPrDefault><w:pPr>'
        '<w:spacing w:after="120" w:line="276" w:lineRule="auto"/>'
        '</w:pPr></w:pPrDefault>'
        '</w:docDefaults>'
        '<w:style w:type="paragraph" w:styleId="Normal" w:default="1">'
        '<w:name w:val="Normal"/>'
        '<w:qFormat/>'
        '</w:style>'
        '<w:style w:type="character" w:styleId="CommentReference">'
        '<w:name w:val="annotation reference"/>'
        '<w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr>'
        '</w:style>'
        '</w:styles>'
    ).encode("utf-8")

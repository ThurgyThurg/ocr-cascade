"""
Pack a dict of {part_name: bytes} into a .docx (OOXML ZIP container).
"""

from __future__ import annotations

import zipfile
from typing import Dict


def pack_docx(out_path: str, parts: Dict[str, bytes]) -> None:
    """Write `parts` as a ZIP at `out_path`. `[Content_Types].xml` must be
    one of the keys."""
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # OOXML spec recommends [Content_Types].xml first.
        if "[Content_Types].xml" in parts:
            zf.writestr("[Content_Types].xml", parts["[Content_Types].xml"])
        for name, data in parts.items():
            if name == "[Content_Types].xml":
                continue
            zf.writestr(name, data)

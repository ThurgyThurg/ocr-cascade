"""
Region routing configuration loader.

Loads a YAML file mapping Docling layout labels (lowercase strings like
"text", "handwritten_text", "table", "picture") to a cascade ladder
(ordered list of engine names). Provides `ladder_for(label)` with
`_default` fallback for unknown labels.

The default config ships at `region_config.yaml` next to this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "region_config.yaml"


@dataclass
class LabelRoute:
    label: str
    ladder: List[str]
    rationale: str = ""
    citation: str = ""


@dataclass
class RegionConfig:
    routes: Dict[str, LabelRoute] = field(default_factory=dict)
    default: LabelRoute = field(default_factory=lambda: LabelRoute("_default", []))

    def ladder_for(self, label: str) -> List[str]:
        route = self.routes.get(label.lower())
        if route is None:
            return list(self.default.ladder)
        return list(route.ladder)

    def route_for(self, label: str) -> LabelRoute:
        return self.routes.get(label.lower(), self.default)


def load_region_config(path: Optional[str | Path] = None) -> RegionConfig:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    routes: Dict[str, LabelRoute] = {}
    default = LabelRoute("_default", ["tesseract", "paddleocr", "glm-ocr"])

    for label, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        ladder = list(entry.get("ladder", []) or [])
        route = LabelRoute(
            label=label,
            ladder=ladder,
            rationale=entry.get("rationale", "") or "",
            citation=entry.get("citation", "") or "",
        )
        if label == "_default":
            default = route
        else:
            routes[label.lower()] = route

    return RegionConfig(routes=routes, default=default)

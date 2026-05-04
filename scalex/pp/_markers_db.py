"""Lazy loader exposing the marker dictionaries as module attributes.

The actual data lives in ``scalex/resources/markers/*.yaml`` so it can be
edited without code changes. Existing imports keep working::

    from scalex.pp._markers_db import (
        cell_type_markers_human, cell_type_markers_mouse,
        macrophage_markers, gene_modules, selected_GO_terms,
    )
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from typing import Any, Dict, List

import yaml

_MARKER_FILES: Dict[str, str] = {
    "cell_type_markers_human": "cell_type_markers_human.yaml",
    "cell_type_markers_mouse": "cell_type_markers_mouse.yaml",
    "macrophage_markers": "macrophage_markers.yaml",
    "gene_modules": "gene_modules.yaml",
}

# Reserved for project-specific or analysis-time overrides; ships empty.
selected_GO_terms: List[str] = []


@lru_cache(maxsize=None)
def _load(name: str) -> Dict[str, List[str]]:
    path = files("scalex.resources.markers") / _MARKER_FILES[name]
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def list_marker_sets() -> List[str]:
    """Names of marker dictionaries shipped with scalex."""
    return list(_MARKER_FILES) + ["selected_GO_terms"]


def __getattr__(name: str) -> Any:
    if name in _MARKER_FILES:
        return _load(name)
    raise AttributeError(f"module 'scalex.pp._markers_db' has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_MARKER_FILES) | {"selected_GO_terms"})

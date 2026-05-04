"""Motif analysis: hit instances, sequence logos, motif compendium matching.

``motif_compendium`` is imported lazily because it requires the optional
``MotifCompendium`` third-party package. Use::

    from scalex.linkage.motif.motif_compendium import ...

only when that dependency is available.
"""
from .instance import Motif
from . import logo_utils

__all__ = ["Motif", "logo_utils"]

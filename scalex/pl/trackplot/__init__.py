"""Genomic track plotting — two complementary APIs.

* Spec-based (recommended for multi-panel figures): :class:`TrackSpec` +
  :func:`compose_tracks`. Each panel is a declarative spec; the framework
  handles layout, normalization, highlighting, and saving.
* Function-based (one panel at a time): ``trackplot_coverage``,
  ``trackplot_gene``, ``trackplot_loop``, ``trackplot_scalebar``,
  ``trackplot_genome_annotation``, ``trackplot_combine``. Useful inside
  notebooks where you assemble panels by hand.

Both live here so callers can mix and match without two import paths.
"""
from .spec import (
    TrackSpec,
    TrackContext,
    register_track,
    compose_tracks,
    flank_peaks,
    assemble_shap_for_region,
)
from .legacy import (
    trackplot_theme,
    trackplot_coverage,
    trackplot_gene,
    trackplot_genome_annotation,
    trackplot_loop,
    trackplot_scalebar,
    trackplot_combine,
    trackplot_bulk,
    trackplot_empty,
    set_trackplot_label,
    set_trackplot_height,
    get_trackplot_height,
    draw_trackplot_grid,
)

__all__ = [
    # spec-based
    "TrackSpec", "TrackContext", "register_track", "compose_tracks",
    "flank_peaks", "assemble_shap_for_region",
    # function-based (legacy)
    "trackplot_theme", "trackplot_coverage", "trackplot_gene",
    "trackplot_genome_annotation", "trackplot_loop", "trackplot_scalebar",
    "trackplot_combine", "trackplot_bulk", "trackplot_empty",
    "set_trackplot_label", "set_trackplot_height", "get_trackplot_height",
    "draw_trackplot_grid",
]

"""Plotting submodule — public API.

Most-used functions (driven by the analysis-notebook scan) are re-exported
here so callers can ``from scalex.pl import embedding, plot_corr, ...``
without remembering the per-module path.

Detailed docstrings live with the implementations:

* :mod:`scalex.pl.embedding` — UMAP/PCA scatter
* :mod:`scalex.pl.correlation` — meta plots, correlation clustermap, radar
* :mod:`scalex.pl._heatmap` — gene/peak heatmaps, plot_corr, local_correlation
* :mod:`scalex.pl._dotplot` — dotplots
* :mod:`scalex.pl._jaccard` — Jaccard heatmaps
* :mod:`scalex.pl._sankey` — Sankey/alluvial diagrams
* :mod:`scalex.pl._stackedplot` — stacked bar / cross-tab
* :mod:`scalex.pl.tracks` — pyGenomeTracks-based locus visualization
* :mod:`scalex.pl.trackplot` — composable spec-based + function-based track plotting
* :mod:`scalex.pl.variant` — ChromBPNet variant effect plots
* :mod:`scalex.pl._legacy_snapatac2` — snapatac2-style plotly QC plots
  (kept for backward compatibility; reach via this submodule directly).
"""
from .embedding import plot_embedding, embedding, plot_subplots
from .correlation import (
    plot_expr,
    plot_pseudobulk_corr,
    plot_pseudobulk_corr_cross,
    plot_meta,           # alias: plot_pseudobulk_corr
    plot_meta2,          # alias: plot_pseudobulk_corr_cross
    plot_corr_clustermap,
    plot_confusion,
    reassign_cluster_with_ref,
    plot_radar,
)
from .tracks import plot_tracks
from ._dotplot import dotplot
from ._jaccard import plot_jaccard_heatmap
from ._sankey import plot_sankey
from ._stackedplot import plot_crosstab, plot_crosstab_stacked
from ._heatmap import plot_heatmap, plot_corr, plot_agg_heatmap, local_correlation_plot, get_module_series
from .trackplot import (
    TrackSpec,
    compose_tracks,
    trackplot_coverage,
    trackplot_gene,
    trackplot_loop,
    trackplot_scalebar,
    trackplot_genome_annotation,
    trackplot_combine,
)

__all__ = [
    # embedding
    "plot_embedding", "embedding", "plot_subplots",
    # meta / correlation
    "plot_expr",
    "plot_pseudobulk_corr", "plot_pseudobulk_corr_cross",
    "plot_meta", "plot_meta2",  # legacy aliases
    "plot_corr_clustermap",
    "plot_confusion", "reassign_cluster_with_ref", "plot_radar",
    # heatmap
    "plot_heatmap", "plot_corr", "plot_agg_heatmap",
    "local_correlation_plot", "get_module_series",
    # categorical
    "dotplot", "plot_jaccard_heatmap", "plot_sankey",
    "plot_crosstab", "plot_crosstab_stacked",
    # tracks
    "plot_tracks",
    "TrackSpec", "compose_tracks",
    "trackplot_coverage", "trackplot_gene", "trackplot_loop",
    "trackplot_scalebar", "trackplot_genome_annotation", "trackplot_combine",
]

"""Plotting submodule — public API.

Catalog of plotting functions, grouped by what they show and when to use them.
Every function takes an ``AnnData`` (or a small DataFrame) and returns either a
matplotlib figure / axes or a ``ClusterGrid`` you can further customise.

Pick by **what you want to see**:

================================  =================================================  =====================================
Function                          Plot type                                          Use it when
================================  =================================================  =====================================
``embedding``                     UMAP / PCA scatter (faceted by batch / condition)  You want one figure per batch coloured by cell-type
``plot_subplots``                 Grid of embeddings across multiple obs columns     You're varying two factors (e.g. condition × time)
``plot_heatmap``                  Multi-panel RNA / ATAC heatmap, k-means rows       Publication figure comparing modalities for the same cells
``plot_agg_heatmap``              Cell-type × gene mean-expression heatmap           Quick aggregated view of marker genes per cell-type
``plot_peak2gene_heatmap``        ATAC peaks ↔ RNA genes paired heatmap              You have peak-to-gene links and want to inspect them
``plot_corr``                     Cell × cell correlation clustermap                 Within-batch or cross-batch cell similarity (square or rectangular)
``plot_corr_clustermap``          Cell × cell Pearson correlation, fixed layout      You want the matplotlib (non-seaborn) version with custom side strips
``local_correlation_plot``        Generic correlation clustermap from a DataFrame    Low-level: you already have a corr matrix and want module-coloured strips
``plot_pseudobulk_corr``          Triangular pseudobulk correlation heatmap          Per-cell-type reproducibility within one dataset
``plot_pseudobulk_corr_cross``    Cross-batch pseudobulk correlation heatmap         Comparing two datasets / replicates side by side
``plot_expr``                     Per-batch violin/box of one gene                   Single-gene expression distribution across conditions
``plot_confusion``                Confusion-matrix heatmap                           Cluster-vs-label accuracy
``plot_radar``                    Radar / spider chart                               Multi-metric profile per cell-type
``dotplot``                       Categorical dot plot (size + colour)               GSEA / enrichment results, scanpy-style marker dots
``plot_jaccard_heatmap``          Jaccard index heatmap                              Set-overlap similarity between two label sets
``plot_sankey``                   Sankey / alluvial diagram                          Category flow (e.g. cluster → cell-type, before/after)
``plot_crosstab``                 Two side-by-side normalised heatmaps               Contingency table with both row- and column-normalised views
``plot_crosstab_stacked``         Stacked-bar composition                            Part-to-whole composition per group
``plot_tracks``                   pyGenomeTracks locus plot                          One-shot genomic locus figure
``trackplot_*``, ``TrackSpec``,   Composable, spec-based locus plots                 Programmatic / nested track composition
``compose_tracks``
================================  =================================================  =====================================

Combining plots
---------------
Most public functions return either a ``matplotlib.figure.Figure`` or a
``seaborn.matrix.ClusterGrid`` whose ``.fig`` you can place inside a larger
gridspec. To draw two of these into one figure, build them and then arrange the
returned figures with ``matplotlib`` directly. The track-plot family
(``trackplot_*``) is purpose-built for composition: assemble ``TrackSpec``
objects with ``compose_tracks`` for vertically stacked panels.

Alias / legacy
--------------
* ``plot_meta`` → alias for :func:`plot_pseudobulk_corr` (kept for backward compat)
* ``plot_meta2`` → alias for :func:`plot_pseudobulk_corr_cross`
* ``embedding`` → alias for :func:`plot_embedding`
* ``_legacy_snapatac2`` — snapatac2-style plotly QC plots, reach via the submodule
"""
# Keep saved PDF/SVG text editable in Illustrator / Inkscape / Figma.
# - svg.fonttype='none' emits <text> elements (system-font references).
# - pdf.fonttype=42 embeds TrueType fonts so text stays selectable.
# Users can override either rcParam after importing if they need outlined text.
import matplotlib as _mpl
_mpl.rcParams['svg.fonttype'] = 'none'
_mpl.rcParams['pdf.fonttype'] = 42
del _mpl

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
    # heatmaps
    "plot_heatmap", "plot_agg_heatmap",
    "local_correlation_plot", "get_module_series",
    # correlation
    "plot_corr",
    "plot_corr_clustermap",
    "plot_pseudobulk_corr", "plot_pseudobulk_corr_cross",
    "plot_meta", "plot_meta2",  # legacy aliases
    "plot_expr", "plot_confusion", "reassign_cluster_with_ref", "plot_radar",
    # categorical / compositional
    "dotplot", "plot_jaccard_heatmap", "plot_sankey",
    "plot_crosstab", "plot_crosstab_stacked",
    # genomic tracks
    "plot_tracks",
    "TrackSpec", "compose_tracks",
    "trackplot_coverage", "trackplot_gene", "trackplot_loop",
    "trackplot_scalebar", "trackplot_genome_annotation", "trackplot_combine",
]

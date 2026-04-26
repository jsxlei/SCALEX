"""Public plotting API — re-exports from domain submodules.

All plotting functions are organized by domain:

- :mod:`scalex.pl.embedding` — UMAP and latent-space scatter plots
- :mod:`scalex.pl.correlation` — correlation heatmaps, confusion matrix, radar chart
- :mod:`scalex.pl.tracks` — genomic track visualization
- :mod:`scalex.pl._heatmap` — multi-panel gene/peak heatmaps
- :mod:`scalex.pl._dotplot` — GO enrichment dot plots
- :mod:`scalex.pl._jaccard` — Jaccard similarity heatmaps
- :mod:`scalex.pl._sankey` — Sankey / alluvial diagrams
- :mod:`scalex.pl._stackedplot` — stacked bar and cross-tab plots
"""

from scalex.pl.embedding import embedding, plot_subplots
from scalex.pl.correlation import (
    plot_expr,
    plot_meta,
    plot_meta2,
    plot_corr_clustermap,
    plot_confusion,
    reassign_cluster_with_ref,
    plot_radar,
)
from scalex.pl.tracks import plot_tracks
from scalex.pl._dotplot import dotplot
from scalex.pl._jaccard import plot_jaccard_heatmap
from scalex.pl._sankey import plot_sankey
from scalex.pl._stackedplot import plot_crosstab, plot_crosstab_stacked
from scalex.pl._heatmap import plot_heatmap

# Define the variable '__version__':
__version__ = "1.0.7"
__author__ = "Lei Xiong"
__email__ = "jsxlei@gmail.com"

from .function import SCALEX, label_transfer
from .pl.plot import (
    embedding, plot_expr, plot_meta, plot_meta2,
    plot_confusion, plot_radar, plot_heatmap, dotplot,
    plot_jaccard_heatmap, plot_sankey, plot_crosstab,
)
from .pp.celltype import annotate, diagonal_heatmap
from .pp.markers import get_markers, find_gene_program, find_consensus_program
from .pp.enrichment import enrich_and_plot, filter_go_genes
"""SCALEX: Online single-cell data integration via a domain-conditional VAE."""

__version__ = "1.2.0"
__author__ = "Lei Xiong"
__email__ = "jsxlei@gmail.com"

from .function import SCALEX, label_transfer
from . import pp, pl, tl
from .tl import cluster
from .pl import (
    embedding, plot_expr, plot_meta, plot_meta2,
    plot_confusion, plot_radar, plot_heatmap, dotplot,
    plot_jaccard_heatmap, plot_sankey, plot_crosstab,
)
from .pp.celltype import annotate, diagonal_heatmap
from .pp.markers import get_markers, find_gene_program, find_consensus_program
from .pp.enrichment import enrich_and_plot, filter_go_genes

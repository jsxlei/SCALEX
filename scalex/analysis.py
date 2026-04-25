"""
Backward-compatibility shim. All logic has moved to scalex/pp/ submodules.
"""
from scalex.pp._markers_db import (
    cell_type_markers_human,
    cell_type_markers_mouse,
    macrophage_markers,
    selected_GO_terms,
    gene_modules,
)
from scalex.pp.markers import (
    get_markers,
    flatten_dict,
    flatten_list,
    filter_marker_dict,
    rename_marker_dict,
    cluster_program,
    find_gene_program,
    find_peak_program,
    _process_group,
    find_consensus_program,
    find_consensus_links,
    get_rank_dict,
)
from scalex.pp.enrichment import (
    check_is_numeric,
    parse_go_file,
    enrich_analysis,
    enrich_and_plot,
    plot_go_enrich,
    filter_go_genes,
    find_go_term_gene,
    format_dict_of_list,
    parse_go_results,
    merge_all_go_results,
    enrich_module,
    plot_enrich_module,
    plot_radar_module,
    aucell_scores,
)
from scalex.pp.celltype import (
    annotate,
    reorder,
    reorder_marker_dict_diagonal,
    diagonal_heatmap,
)

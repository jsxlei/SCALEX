import numpy as np
import pandas as pd
import scanpy as sc

from scalex.pp._markers_db import cell_type_markers_human, cell_type_markers_mouse
from scalex.pp.markers import get_markers
from scalex.pp.enrichment import enrich_and_plot


def annotate(
    adata,
    cell_type='leiden',
    color=['cell_type', 'leiden', 'tissue', 'donor'],
    cell_type_markers='human',
    show_markers=False,
    gene_sets='GO_Biological_Process_2023',
    additional={},
    go=True,
    out_dir=None,
    cutoff=0.05,
    processed=False,
    top_n=300,
    filter_pseudo=True,
):
    color = [i for i in color if i in adata.obs.columns]
    color = color + [cell_type] if cell_type not in color else color
    sc.pl.umap(adata, color=color, legend_loc='on data', legend_fontsize=10)

    var_names = adata.raw.var_names if adata.raw is not None else adata.var_names
    if cell_type_markers is not None:
        if isinstance(cell_type_markers, str):
            if cell_type_markers == 'human':
                cell_type_markers = cell_type_markers_human
            elif cell_type_markers == 'mouse':
                cell_type_markers = cell_type_markers_mouse
                filter_pseudo = False
        cell_type_markers_ = {k: [i for i in v if i in var_names] for k, v in cell_type_markers.items()}
        sc.pl.dotplot(adata, cell_type_markers_, groupby=cell_type, standard_scale='var', cmap='coolwarm')

    if 'rank_genes_groups' not in adata.uns:
        if not processed:
            sc.pp.normalize_total(adata, target_sum=10000)
            sc.pp.log1p(adata)
        sc.tl.rank_genes_groups(adata, groupby=cell_type, method='t-test')
    sc.pl.rank_genes_groups_dotplot(adata, groupby=cell_type, n_genes=10)

    marker_genes = get_markers(adata, groupby=cell_type, processed=processed, top_n=top_n, filter_pseudo=filter_pseudo)

    try:
        go_df = enrich_and_plot(marker_genes, gene_sets=gene_sets, cutoff=cutoff, out_dir=out_dir)
        return go_df
    except Exception as e:
        print(e)
        return None


def reorder(adata, groupby: str = 'cell_type', order=None):
    """Reorder the categories of an obs column to a specified order.

    Parameters
    ----------
    adata : AnnData
        Annotated data.
    groupby : str, default 'cell_type'
        Column in ``obs`` to reorder.
    order : list
        Desired category order.

    Returns
    -------
    AnnData
        Same ``adata`` with categories reordered in place.
    """
    adata.obs[groupby] = adata.obs[groupby].astype(str).astype('category').cat.reorder_categories(order)
    return adata


def reorder_marker_dict_diagonal(marker_dict: dict, avg_adata, groupby: str = 'cell_type') -> dict:
    """
    Merge clusters in marker_dict by their dominant cell type, producing a diagonal
    pattern in the heatmap. Clusters are merged into cell-type-keyed buckets.

    Returns
    -------
    dict keyed by cell type name with merged features from all matching clusters
    """
    cell_types = avg_adata.obs[groupby].tolist()

    ct_features = {ct: [] for ct in cell_types}
    for cluster, features in marker_dict.items():
        present = [f for f in features if f in avg_adata.var_names]
        if not present:
            continue
        X = avg_adata[:, present].X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        dominant_ct = cell_types[int(np.argmax(X.mean(axis=1)))]
        ct_features[dominant_ct].extend(present)

    return {ct: ct_features[ct] for ct in cell_types if ct_features[ct]}


def diagonal_heatmap(adata, groupby='cell_type', set_type='gene', order=None, dataset=None,
                     n_clusters=25, top_n=300, processed=False, filter_pseudo=None,
                     cluster_method='kmeans', method='wilcoxon', force=True,
                     pval_cutoff=None, logfc_cutoff=None, min_pct=None, min_cells=10, min_cell_per_batch=100,
                     cmap='RdBu_r', limits=(-2, 2), figsize=None, col_width=0.09, height=4,
                     save=None, **kwargs):
    """
    Find peak or gene programs and plot a diagonal heatmap where clusters are
    automatically ordered to match their dominant cell type.

    Parameters
    ----------
    set_type
        'peak' or 'gene'
    order
        Explicit cell-type order for the heatmap columns.

    Returns
    -------
    ordered_marker_dict, avg_adata
    """
    from scalex.pp.markers import find_gene_program, find_peak_program, find_consensus_program
    from scalex.pl._heatmap import plot_heatmap

    # Peaks rarely show the large fold changes typical of marker genes, so use
    # gentler cutoffs for peaks unless the caller overrides them explicitly.
    if pval_cutoff is None:
        pval_cutoff = 0.05 if set_type == 'peak' else 0.01
    if logfc_cutoff is None:
        logfc_cutoff = 0.25 if set_type == 'peak' else 1.5
    # Detection-fraction (prevalence) filter: peaks are sparser, so use a gentler
    # default than genes.
    if min_pct is None:
        min_pct = 0.05 if set_type == 'peak' else 0.10
    # Pseudogene filtering only applies to gene names; peaks ("chr:start-end")
    # must not be passed through format_rna.
    if filter_pseudo is None:
        filter_pseudo = set_type != 'peak'

    marker_kw = dict(pval_cutoff=pval_cutoff, logfc_cutoff=logfc_cutoff, min_pct=min_pct,
                     min_cells=min_cells, min_cell_per_batch=min_cell_per_batch)
    if set_type == 'peak':
        if dataset is None:
            marker_dict, avg_adata = find_peak_program(adata, groupby=groupby, processed=processed,
                                                       n_clusters=n_clusters, top_n=top_n,
                                                       filter_pseudo=filter_pseudo, method=method, force=force,
                                                       **marker_kw)
        else:
            marker_dict, avg_adata = find_consensus_program(adata, groupby=groupby, across=dataset, set_type='peak',
                                                            processed=processed, top_n=top_n, method=method, **marker_kw)
    elif set_type == 'gene':
        if dataset is None:
            marker_dict, avg_adata = find_gene_program(adata, groupby=groupby, processed=processed,
                                                       n_clusters=n_clusters, top_n=top_n,
                                                       filter_pseudo=filter_pseudo, cluster_method=cluster_method,
                                                       method=method, force=force, **marker_kw)
        else:
            marker_dict, avg_adata = find_consensus_program(adata, groupby=groupby, across=dataset, set_type='gene',
                                                            processed=processed, top_n=top_n, method=method, **marker_kw)
    else:
        raise ValueError("set_type must be 'peak' or 'gene'")

    # With multiple datasets, render one aligned panel per dataset (each with its
    # own cell-type columns) instead of a single merged heatmap. The first dataset
    # anchors the shared gene-program row order.
    if dataset is not None:
        datasets   = list(pd.unique(avg_adata.obs[dataset]))
        panel_list = [avg_adata[avg_adata.obs[dataset] == d].copy() for d in datasets]
        anchor_adata = panel_list[0]
    else:
        datasets, panel_list = None, None
        anchor_adata = reorder(avg_adata, groupby=groupby, order=order) if order is not None else avg_adata

    ordered_dict = reorder_marker_dict_diagonal(marker_dict, anchor_adata, groupby=groupby)
    if order is not None:
        ordered_dict = {ct: ordered_dict[ct] for ct in order if ct in ordered_dict}
    ct_order = order if order is not None else anchor_adata.obs[groupby].tolist()

    hm_kw = dict(
        order_rows=False,
        order=ct_order,
        limits=limits,
        cmaps=cmap,
        figsize=figsize,
        col_width=col_width,
        height=height,
        save=save,
        groupby=groupby,
        labels=datasets if dataset is not None else [set_type],
        show_gene_labels='bar',
        per_panel_columns=dataset is not None,
    )
    hm_kw.update(kwargs)

    data_arg = panel_list if dataset is not None else avg_adata
    if set_type == 'gene':
        plot_heatmap(rna=data_arg, genes=ordered_dict, **hm_kw)
    else:
        plot_heatmap(atac=data_arg, peaks=ordered_dict, **hm_kw)

    return ordered_dict, avg_adata

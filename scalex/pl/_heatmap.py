import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools

import scanpy as sc
import anndata

from scalex.pl._utils import _sort_key


def plot_agg_heatmap(adata, genes, cell_type: str = 'cell_type', **kwargs):
    """Heatmap of per-cell-type mean expression for a gene set.

    Aggregates ``adata`` (or ``adata.raw``) by ``cell_type`` (mean), wraps
    the result back into an AnnData, and dispatches to ``sc.pl.heatmap``.
    """
    adata_agg = adata.raw.to_adata() if adata.raw is not None else adata
    df = adata_agg.to_df()
    df[cell_type] = adata_agg.obs[cell_type].values
    df_agg = df.groupby(cell_type).mean()
    df_ann = anndata.AnnData(df_agg)
    df_ann.obs['groupby'] = df_agg.index
    return sc.pl.heatmap(df_ann, genes, groupby='groupby',
                         show_gene_labels=True, **kwargs)


def get_module_series(topgenes): # topgenes are DataFrame with ranks as index, GEPs as columns
    long_df = topgenes.stack().reset_index()
    long_df.columns = ['Rank', 'GEP', 'Gene']
    long_df = long_df.sort_values('Rank', ascending=True)
    gene_series = long_df.drop_duplicates(subset='Gene', keep='first').set_index('Gene')['GEP']
    return gene_series

def _build_color_lookup(modules_series, mod_cmap='tab10', color_map=None):
    if color_map is not None:
        return color_map
    unique_modules = sorted(modules_series.unique())
    palette_cycle = itertools.cycle(sns.color_palette(mod_cmap))
    lookup = {}
    for mod in unique_modules:
        if mod == -1:
            lookup[mod] = '#ffffff'
        else:
            lookup[mod] = next(palette_cycle)
    return lookup


def _add_row_labels(cm, modules_aligned):
    if cm.dendrogram_row is not None:
        reordered_indices = cm.dendrogram_row.reordered_ind
    else:
        reordered_indices = np.arange(len(modules_aligned))
    mod_reordered = modules_aligned.iloc[reordered_indices]
    y = np.arange(modules_aligned.size)
    mod_map = {x: y[mod_reordered == x].mean() for x in mod_reordered.unique() if x != -1}
    plt.sca(cm.ax_row_colors)
    for mod, mod_y in mod_map.items():
        plt.text(-0.5, y=mod_y, s="{}".format(mod),
                 horizontalalignment='right', verticalalignment='center')
    plt.xticks([])


def _add_col_labels(cm, col_modules_aligned):
    if cm.dendrogram_col is not None:
        reordered_indices = cm.dendrogram_col.reordered_ind
    else:
        reordered_indices = np.arange(len(col_modules_aligned))
    mod_reordered = col_modules_aligned.iloc[reordered_indices]
    x = np.arange(col_modules_aligned.size)
    mod_map = {m: x[mod_reordered == m].mean() for m in mod_reordered.unique() if m != -1}
    ax_cc = cm.ax_col_colors
    from matplotlib.transforms import blended_transform_factory
    transform = blended_transform_factory(ax_cc.transData, ax_cc.transAxes)
    for mod, mod_x in mod_map.items():
        ax_cc.text(x=mod_x, y=1.05, s="{}".format(mod),
                   horizontalalignment='left', verticalalignment='bottom',
                   rotation=45, transform=transform, clip_on=False)
    ax_cc.set_yticks([])


def _to_flat_series(s, index):
    """Return a plain object-dtype Series with a positional RangeIndex, stripping Categorical/MultiIndex."""
    return pd.Series(list(s), index=index, dtype=object)


def _sort_order(series):
    """Return argsort indices using the shared _sort_key for numeric-aware sorting."""
    keys = series.map(_sort_key)
    return keys.argsort().values


def _proportional_figsize(n_rows, n_cols, size, max_ratio=1.5):
    ratio = min(max_ratio, max(1 / max_ratio, n_cols / n_rows))
    if ratio >= 1:
        return (size, size / ratio)
    return (size * ratio, size)


def local_correlation_plot(
            local_correlation_z, modules, linkage=None,
            col_modules=None, order=None,
            mod_cmap='tab10', vmin=-1, vmax=1, color_map=None,
            z_cmap='RdBu_r', yticklabels=False, save=False,
            cluster=True, size=10, square=True,
):
    # Normalise to positional RangeIndex throughout to avoid MultiIndex/Categorical issues
    n_rows = len(local_correlation_z)
    n_cols = local_correlation_z.shape[1]
    row_idx = np.arange(n_rows)
    col_idx = np.arange(n_cols)
    local_correlation_z = pd.DataFrame(local_correlation_z.values, index=row_idx, columns=col_idx)
    modules_aligned = _to_flat_series(modules, row_idx)

    if not cluster:
        if order is not None:
            rank = {ct: i for i, ct in enumerate(order)}
            sorted_order = sorted(range(n_rows), key=lambda i: rank.get(modules_aligned.iloc[i], len(order)))
        else:
            sorted_order = _sort_order(modules_aligned)

        if col_modules is None:
            local_correlation_z = pd.DataFrame(
                local_correlation_z.values[np.ix_(sorted_order, sorted_order)],
                index=row_idx, columns=col_idx,
            )
            modules_aligned = _to_flat_series(modules_aligned.iloc[sorted_order], row_idx)
        else:
            local_correlation_z = pd.DataFrame(
                local_correlation_z.values[sorted_order, :],
                index=row_idx, columns=col_idx,
            )
            modules_aligned = _to_flat_series(modules_aligned.iloc[sorted_order], row_idx)

    module_color_lookup = _build_color_lookup(modules_aligned, mod_cmap, color_map)
    row_colors_series = pd.Series([module_color_lookup[v] for v in modules_aligned], index=modules_aligned.index)
    row_colors_series.name = "Rows"

    clustermap_kwargs = dict(
        vmin=vmin, vmax=vmax, cmap=z_cmap,
        xticklabels=False, yticklabels=yticklabels,
        row_colors=row_colors_series.to_frame(),
        rasterized=True,
        figsize=(size, size) if square else _proportional_figsize(n_rows, n_cols, size),
    )

    if col_modules is not None:
        # Two-adata mode: annotate columns independently
        col_modules_aligned = _to_flat_series(col_modules, col_idx)
        if not cluster:
            if order is not None:
                rank = {ct: i for i, ct in enumerate(order)}
                col_sorted_order = sorted(range(n_cols), key=lambda i: rank.get(col_modules_aligned.iloc[i], len(order)))
            else:
                col_sorted_order = _sort_order(col_modules_aligned)
            local_correlation_z = pd.DataFrame(
                local_correlation_z.values[:, col_sorted_order],
                index=row_idx, columns=col_idx,
            )
            col_modules_aligned = _to_flat_series(col_modules_aligned.iloc[col_sorted_order], col_idx)
        col_color_lookup = _build_color_lookup(col_modules_aligned, mod_cmap, color_map)
        col_colors_series = pd.Series([col_color_lookup[v] for v in col_modules_aligned], index=col_modules_aligned.index)
        col_colors_series.name = "Cols"
        clustermap_kwargs['col_colors'] = col_colors_series.to_frame()
        if not cluster:
            clustermap_kwargs.update(row_cluster=False, col_cluster=False)
        # linkage does not apply symmetrically for non-square; let seaborn compute independently
    else:
        if cluster:
            clustermap_kwargs.update(row_linkage=linkage, col_linkage=linkage)
        else:
            clustermap_kwargs.update(row_cluster=False, col_cluster=False)

    cm = sns.clustermap(local_correlation_z, **clustermap_kwargs)

    fig = cm.fig
    plt.sca(cm.ax_heatmap)
    plt.ylabel("")
    plt.xlabel("")

    if cm.ax_row_dendrogram is not None:
        cm.ax_row_dendrogram.remove()
    if cm.ax_col_dendrogram is not None:
        cm.ax_col_dendrogram.remove()

    _add_row_labels(cm, modules_aligned)

    if col_modules is not None:
        _add_col_labels(cm, col_modules_aligned)

    if cm.cax:
        cm.cax.grid(False)
        # Reposition colorbar to bottom-right corner of the figure
        heatmap_pos = cm.ax_heatmap.get_position()
        cbar_width = 0.02
        cbar_height = 0.15
        cm.cax.set_position([
            heatmap_pos.x1 + 0.01,
            heatmap_pos.y0,
            cbar_width,
            cbar_height,
        ])
        cm.cax.set_ylabel('Correlation', fontsize=12)
        cm.cax.yaxis.set_label_position("right")

    if save:
        import os
        os.makedirs(os.path.dirname(save), exist_ok=True) if os.path.dirname(save) else None
        fig.savefig(save, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return cm


def _subsample_by_group(adata, groupby, n=50):
    def _n_for_group(g):
        k = max(1, round(len(g) * n)) if isinstance(n, float) else n
        return g.sample(min(len(g), k))

    idx = (
        adata.obs.groupby(groupby, observed=True)
        .apply(_n_for_group)
        .index.get_level_values(-1)
    )
    return adata[idx]


def _pearson_corr(X):
    """Pearson correlation matrix between rows of X (float32, normalized matmul)."""
    X = X.astype(np.float32)
    X_c = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_n = X_c / norms
    return X_n @ X_n.T


def _pearson_cross(A, B):
    """Pearson cross-correlation between rows of A and rows of B (float32)."""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    A_c = A - A.mean(axis=1, keepdims=True)
    B_c = B - B.mean(axis=1, keepdims=True)
    A_n = A_c / (np.linalg.norm(A_c, axis=1, keepdims=True) + 1e-10)
    B_n = B_c / (np.linalg.norm(B_c, axis=1, keepdims=True) + 1e-10)
    return A_n @ B_n.T


def plot_corr(
        adata, adata2=None, groupby='cell_type', obsm_key='latent',
        batch=None, subsample=False, subsample_n=50, **kwargs
):
    if batch is not None:
        batches = adata.obs[batch].unique()
        if len(batches) != 2:
            raise ValueError(f"batch column '{batch}' must have exactly 2 unique values, got {list(batches)}")
        adata, adata2 = adata[adata.obs[batch] == batches[0]], adata[adata.obs[batch] == batches[1]]

    if subsample:
        adata = _subsample_by_group(adata, groupby, subsample_n)
        if adata2 is not None:
            adata2 = _subsample_by_group(adata2, groupby, subsample_n)

    # Use positional integer index to avoid any MultiIndex issues from AnnData views
    if adata2 is None:
        corr = _pearson_corr(adata.obsm[obsm_key])
        idx1 = np.arange(len(adata))
        corr_df = pd.DataFrame(corr, index=idx1, columns=idx1)
        modules = pd.Series(adata.obs[groupby].values, index=idx1)
        return local_correlation_plot(corr_df, modules=modules, **kwargs)

    # Cross-correlation: N x M (adata rows vs adata2 rows)
    idx1 = np.arange(len(adata))
    idx2 = np.arange(len(adata2))
    corr = _pearson_cross(adata.obsm[obsm_key], adata2.obsm[obsm_key])
    transpose = kwargs.pop('transpose', False)
    if transpose:
        corr = corr.T
        corr_df = pd.DataFrame(corr, index=idx2, columns=idx1)
        modules = pd.Series(adata2.obs[groupby].values, index=idx2)
        col_modules = pd.Series(adata.obs[groupby].values, index=idx1)
    else:
        corr_df = pd.DataFrame(corr, index=idx1, columns=idx2)
        modules = pd.Series(adata.obs[groupby].values, index=idx1)
        col_modules = pd.Series(adata2.obs[groupby].values, index=idx2)
    return local_correlation_plot(corr_df, modules=modules, col_modules=col_modules, **kwargs)


def _subsample_by_group(adata, groupby, n=50):
    def _n_for_group(g):
        k = max(1, round(len(g) * n)) if isinstance(n, float) else n
        return g.sample(min(len(g), k))

    idx = (
        adata.obs.groupby(groupby, observed=True)
        .apply(_n_for_group)
        .index.get_level_values(-1)
    )
    return adata[idx]


def _pearson_corr(X):
    """Pearson correlation matrix between rows of X (float32, normalized matmul)."""
    X = X.astype(np.float32)
    X_c = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_n = X_c / norms
    return X_n @ X_n.T


def _pearson_cross(A, B):
    """Pearson cross-correlation between rows of A and rows of B (float32)."""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    A_c = A - A.mean(axis=1, keepdims=True)
    B_c = B - B.mean(axis=1, keepdims=True)
    A_n = A_c / (np.linalg.norm(A_c, axis=1, keepdims=True) + 1e-10)
    B_n = B_c / (np.linalg.norm(B_c, axis=1, keepdims=True) + 1e-10)
    return A_n @ B_n.T


def plot_corr(
        adata, adata2=None, groupby='cell_type', obsm_key='latent',
        batch=None, subsample=False, subsample_n=50, **kwargs
):
    if batch is not None:
        batches = adata.obs[batch].unique()
        if len(batches) != 2:
            raise ValueError(f"batch column '{batch}' must have exactly 2 unique values, got {list(batches)}")
        adata, adata2 = adata[adata.obs[batch] == batches[0]], adata[adata.obs[batch] == batches[1]]

    if subsample:
        adata = _subsample_by_group(adata, groupby, subsample_n)
        if adata2 is not None:
            adata2 = _subsample_by_group(adata2, groupby, subsample_n)

    if adata2 is None:
        corr = _pearson_corr(adata.obsm[obsm_key])
        idx1 = np.arange(len(adata))
        corr_df = pd.DataFrame(corr, index=idx1, columns=idx1)
        modules = pd.Series(adata.obs[groupby].values, index=idx1)
        return local_correlation_plot(corr_df, modules=modules, **kwargs)

    idx1 = np.arange(len(adata))
    idx2 = np.arange(len(adata2))
    corr = _pearson_cross(adata.obsm[obsm_key], adata2.obsm[obsm_key])
    transpose = kwargs.pop('transpose', False)
    if transpose:
        corr = corr.T
        corr_df = pd.DataFrame(corr, index=idx2, columns=idx1)
        modules = pd.Series(adata2.obs[groupby].values, index=idx2)
        col_modules = pd.Series(adata.obs[groupby].values, index=idx1)
    else:
        corr_df = pd.DataFrame(corr, index=idx1, columns=idx2)
        modules = pd.Series(adata.obs[groupby].values, index=idx1)
        col_modules = pd.Series(adata2.obs[groupby].values, index=idx2)
    return local_correlation_plot(corr_df, modules=modules, col_modules=col_modules, **kwargs)


def _row_zscore(X):
    """Z-score each row; rows with zero variance are left as zeros."""
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True)
    sigma[sigma == 0] = 1
    return (X - mu) / sigma


    
def _parse_peak(p):
    chrom, rest = p.split(':')
    s, e = rest.split('-')
    return chrom, (int(s) + int(e)) // 2


def plot_peak2gene_heatmap(
    atac_avg,
    rna_avg,
    links,
    groupby='cell_type',
    gene_col=None,
    k=None,
    n_plot=25000,
    limits_atac=(-2, 2),
    limits_rna=(-2, 2),
    cmap_atac='RdBu_r',
    cmap_rna='viridis',
    order=None,
    figsize=None,
    col_width=0.09,
    height=4,
    save=None,
    return_matrices=False,
    seed=1,
    order_links=True,
    links_order=None,
    transpose=False,
):
    """
    Cell-type-averaged peak-to-gene heatmap (publication quality).

    Parameters
    ----------
    atac_avg, rna_avg : AnnData   Cell-type-averaged matrices.
    links : pd.DataFrame | dict   Either a flat DataFrame with columns
                                  ['peak', 'gene'] (and optionally 'cluster'),
                                  as returned by compute_peak2gene_links, or a
                                  legacy {cluster: DataFrame} dict.
    groupby : str                 Cell-type column in obs.
    gene_col : str|None           Auto-detected ('gene' or 'TargetGene').
    k : int|None                  K-means clusters. Default: len(links dict).
    n_plot : int                  Max links (rows).
    limits_atac, limits_rna : (float,float)  Color-scale clipping.
    cmap_atac : str               ATAC colormap (default 'RdBu_r').
    cmap_rna  : str               RNA colormap  (default 'viridis').
    order : list|None             Explicit cell-type column order.
    figsize : (float,float)|None  Override auto-sized figure.
    col_width : float             Inches per cell-type column (default 0.25).
    height : float                Figure height in inches (default 7).
    save : str|None               Output path (dpi=300, bbox_inches='tight').
    return_matrices : bool        Return matrices instead of plotting.
    seed : int                    Random seed.
    order_links : bool            Whether to reorder links via k-means (default True).
                                  Set False to preserve input order.
    links_order : DataFrame|None  links_df from a previous call; replays that exact
                                  peak/gene row order on new data (implies order_links=False).

    Example
    -------
    links_df = compute_peak2gene_links(atac_avg, rna_avg, peak_list=marker_peaks)
    plot_peak2gene_heatmap(atac_avg, rna_avg, links_df)
    """
    from sklearn.cluster import KMeans
    from scipy.cluster.hierarchy import linkage, leaves_list
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    np.random.seed(seed)

    # Normalise links to an internal dict
    if isinstance(links, pd.DataFrame):
        if 'cluster' in links.columns:
            _link_dict = {
                c: links[links['cluster'] == c]
                for c in links['cluster'].unique()
            }
        else:
            _link_dict = {'_all': links}
    else:
        _link_dict = links   # existing dict path unchanged

    # Auto-detect gene column
    if gene_col is None:
        first_df = next(iter(_link_dict.values()))
        gene_col = next((c for c in ['gene', 'TargetGene'] if c in first_df.columns),
                        first_df.columns[-1])

    cell_types = atac_avg.obs[groupby].tolist()
    n_ct = len(cell_types)

    # ── 1. Aligned (peak, gene) pairs ────────────────────────────────────
    use_ordering = order_links and links_order is None

    if links_order is not None:
        valid = (
            links_order['peak'].isin(set(atac_avg.var_names)) &
            links_order['gene'].isin(set(rna_avg.var_names))
        )
        lo = links_order[valid].reset_index(drop=True)
        peak_list = lo['peak'].tolist()
        gene_list = lo['gene'].tolist()
        ct_list    = lo['cell_type'].tolist() if 'cell_type' in lo.columns else [''] * len(lo)
        tss_list   = lo['tss'].tolist() if 'tss' in lo.columns else [None] * len(lo)
        score_list = lo['correlation'].tolist() if 'correlation' in lo.columns else [None] * len(lo)
    else:
        peak_list, gene_list, ct_list, tss_list, score_list = [], [], [], [], []
        for ct, links_df in _link_dict.items():
            for _, row in links_df.iterrows():
                peak, gene = row['peak'], row[gene_col]
                if peak in atac_avg.var_names and gene in rna_avg.var_names:
                    peak_list.append(peak)
                    gene_list.append(gene)
                    ct_list.append(ct)
                    tss_list.append(row.get('tss', None))
                    score_list.append(row.get('correlation', None))
    if not peak_list:
        raise ValueError("No valid peak-gene pairs found.")

    def _extract(adata, features):
        X = adata[:, features].X
        return X.toarray().T if hasattr(X, 'toarray') else np.array(X).T

    M_atac = _extract(atac_avg, peak_list)   # (n_pairs, n_ct)
    M_rna  = _extract(rna_avg,  gene_list)

    # ── 2. Row Z-score ────────────────────────────────────────────────────
    M_atac = _row_zscore(M_atac)
    M_rna  = _row_zscore(M_rna)

    # ── 3. K-means on ATAC rows ───────────────────────────────────────────
    if use_ordering:
        if k is None:
            k = len(_link_dict)
        k = min(k, len(peak_list))
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km_labels = km.fit_predict(M_atac)
    else:
        k = 1
        km_labels = np.zeros(len(peak_list), dtype=int)

    # ── 4. Downsample ─────────────────────────────────────────────────────
    if links_order is None and len(peak_list) > n_plot:
        if use_ordering:
            keep = []
            for cid in range(k):
                idxs   = np.where(km_labels == cid)[0]
                n_keep = max(1, int(n_plot * len(idxs) / len(peak_list)))
                keep.extend(np.random.choice(idxs, min(n_keep, len(idxs)), replace=False))
            keep = np.array(sorted(keep))
        else:
            keep = np.sort(np.random.choice(len(peak_list), n_plot, replace=False))
        M_atac, M_rna, km_labels = M_atac[keep], M_rna[keep], km_labels[keep]
        peak_list  = [peak_list[i]  for i in keep]
        gene_list  = [gene_list[i]  for i in keep]
        ct_list    = [ct_list[i]    for i in keep]
        tss_list   = [tss_list[i]   for i in keep]
        score_list = [score_list[i] for i in keep]

    # ── 5. Column order ───────────────────────────────────────────────────
    cluster_means = np.array([
        M_atac[km_labels == cid].mean(axis=0) if (km_labels == cid).any()
        else np.zeros(n_ct)
        for cid in range(k)
    ])

    if order is not None:
        ct_index  = {ct: i for i, ct in enumerate(cell_types)}
        col_order = np.array([ct_index[ct] for ct in order if ct in ct_index])
    else:
        col_order = leaves_list(linkage(cluster_means.T, method='ward')) \
                    if n_ct > 1 else np.arange(n_ct)

    # ── 6. Row order: diagonal ────────────────────────────────────────────
    if use_ordering:
        cluster_peak_pos = np.argmax(cluster_means[:, col_order], axis=1)
        cluster_order    = np.argsort(cluster_peak_pos)

        row_order, split_pos = [], []
        for old_cid in cluster_order:
            idxs       = np.where(km_labels == old_cid)[0]
            dominant   = col_order[int(cluster_peak_pos[old_cid])]
            within     = np.argsort(-M_atac[idxs, dominant])
            row_order.extend(idxs[within])
            split_pos.append(len(row_order))
        row_order = np.array(row_order)
    else:
        row_order = np.arange(len(peak_list))
        split_pos = [len(peak_list)]

    M_atac_ord = M_atac[row_order][:, col_order]
    M_rna_ord  = M_rna[row_order][:, col_order]
    ct_ord     = [cell_types[i] for i in col_order]
    n_ct       = len(ct_ord)   # update to displayed count (subset when order is given)

    links_df = pd.DataFrame({
        'Chromosome': [_parse_peak(peak_list[i])[0] for i in row_order],
        'Start':      [_parse_peak(peak_list[i])[1] for i in row_order],
        'End':        [tss_list[i]  for i in row_order],
        'peak':       [peak_list[i] for i in row_order],
        'gene':       [gene_list[i] for i in row_order],
        'score':      [score_list[i] for i in row_order],
    })

    if return_matrices:
        return dict(ATAC=M_atac_ord, RNA=M_rna_ord, cell_types=ct_ord,
                    km_labels=km_labels[row_order],
                    split_positions=split_pos[:-1],
                    links=links_df)

    # ── 7. Layout ─────────────────────────────────────────────────────────
    gap    = 0.1   # small gap between ATAC and RNA panels (ratio units)
    margin = 0.4
    cb_left = 0.91
    cb_w    = 0.025

    if transpose:
        gs_top, gs_bottom = 0.95, 0.05
        hm_height  = n_ct * col_width
        fig_height = 2 * hm_height + gap + 1.5
        fig_width  = height
        if figsize is not None:
            fig_width, fig_height = figsize

        fig = plt.figure(figsize=(fig_width, fig_height))

        gs = gridspec.GridSpec(
            3, 1,
            height_ratios=[hm_height, gap, hm_height],
            hspace=0,
            left=0.12, right=0.88, top=gs_top, bottom=gs_bottom,
        )
        ax_atac = fig.add_subplot(gs[0, 0])
        ax_rna  = fig.add_subplot(gs[2, 0])

        # Each colorbar = 1/4 of its heatmap height in figure fractions
        total_frac = gs_top - gs_bottom
        hm_frac    = hm_height / (2 * hm_height + gap) * total_frac
        cb_h       = hm_frac / 4
        ax_cb_atac = fig.add_axes([cb_left, gs_top - cb_h,  cb_w, cb_h])
        ax_cb_rna  = fig.add_axes([cb_left, gs_bottom,       cb_w, cb_h])
    else:
        gs_top, gs_bottom = 0.88, 0.22
        hm_width   = n_ct * col_width
        fig_width  = 2 * hm_width + gap + margin + 1.0
        fig_height = height
        if figsize is not None:
            fig_width, fig_height = figsize

        fig = plt.figure(figsize=(fig_width, fig_height))

        gs = gridspec.GridSpec(
            1, 3,
            width_ratios=[hm_width, gap, hm_width],
            wspace=0,
            left=0.08, right=0.88, top=gs_top, bottom=gs_bottom,
        )
        ax_atac = fig.add_subplot(gs[0, 0])
        ax_rna  = fig.add_subplot(gs[0, 2])

        # Each colorbar = 1/4 of heatmap height in figure fractions
        hm_frac    = gs_top - gs_bottom
        cb_h       = hm_frac / 4
        ax_cb_atac = fig.add_axes([cb_left, gs_top - cb_h,  cb_w, cb_h])
        ax_cb_rna  = fig.add_axes([cb_left, gs_bottom,       cb_w, cb_h])

    # ── 8. Heatmaps ───────────────────────────────────────────────────────
    for ax, mat, limits, cmap, title in [
        (ax_atac, M_atac_ord, limits_atac, cmap_atac, f'ATAC  ({len(peak_list)} links)'),
        (ax_rna,  M_rna_ord,  limits_rna,  cmap_rna,  f'RNA   ({len(gene_list)} links)'),
    ]:
        if transpose:
            mat = mat.T
        ax.imshow(mat, aspect='auto', cmap=cmap,
                  vmin=limits[0], vmax=limits[1], interpolation='none')
        ax.set_title(title, fontsize=7, pad=3)
        if transpose:
            ax.set_yticks(range(n_ct))
            ax.set_yticklabels(ct_ord, fontsize=6)
            ax.set_xticks([])
        else:
            ax.set_xticks(range(n_ct))
            ax.set_xticklabels(ct_ord, rotation=45, ha='right', fontsize=6)
            ax.set_yticks([])
        ax.set_frame_on(False)
        ax.grid(False)

    # ── 9. Colorbars (stacked, right corner) ──────────────────────────────
    for ax_cb, limits, cmap, label in [
        (ax_cb_atac, limits_atac, cmap_atac, 'Z-score ATAC'),
        (ax_cb_rna,  limits_rna,  cmap_rna,  'Z-score RNA'),
    ]:
        sm = ScalarMappable(cmap=cmap, norm=Normalize(*limits))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=ax_cb, orientation='vertical')
        cb.set_label(label, fontsize=5, labelpad=2)
        cb.set_ticks([limits[0], 0, limits[1]])
        ax_cb.tick_params(labelsize=4, length=2, pad=1)
        for spine in ax_cb.spines.values():
            spine.set_visible(False)

    if save is not None:
        fig.savefig(save, bbox_inches='tight', dpi=300)

    return links_df


def plot_heatmap(
    rna=None,
    atac=None,
    genes=None,
    peaks=None,
    links=None,
    groupby='cell_type',
    k=None,
    limits=(-2, 2),
    cmaps=None,
    labels=None,
    order=None,
    figsize=None,
    col_width=0.09,
    height=4,
    save=None,
    return_matrices=False,
    seed=1,
    order_rows=True,
    row_order=None,
    transpose=False,
    show_gene_labels=False,
    color_map=None,
    gene_ticks=None,
    gene_tick_fontsize=4,
    title=None,
):
    """
    Unified multi-panel heatmap (publication quality).

    Plots any combination of RNA and ATAC/gene-activity panels side-by-side
    (or stacked) with a shared row ordering derived from an anchor panel.

    Parameters
    ----------
    rna : AnnData | list[AnnData]
        RNA expression panel(s).
    atac : AnnData | list[AnnData]
        ATAC accessibility or gene-activity panel(s).
    genes : list[str] | dict[int, list[str]]
        Features for RNA panels (and ATAC panels when *peaks* is None).
        A single list is shared across all panels; a dict maps panel index
        to a per-panel list.
    peaks : list[str] | dict[int, list[str]] | None
        Features for ATAC panels. Falls back to *genes* when None.
    links : pd.DataFrame | dict | None
        Peak-to-gene links as returned by ``compute_peak2gene_links``.
        When provided, *genes*/*peaks* are ignored and link rows define
        the features for each panel.
    groupby : str
        Column in ``.obs`` containing cell-type labels.
    k : int | None
        K-means clusters for row grouping. Defaults to number of cell types.
    limits : (float, float) | list[(float, float)]
        Colour-scale clipping, shared or per-panel.
    cmaps : str | list[str] | None
        Colourmap(s) per panel. Defaults: RNA -> 'viridis', ATAC -> 'RdBu_r'.
    labels : list[str] | None
        Custom panel labels (one per panel, in order: atac panels then rna panels
        when links provided, otherwise rna panels then atac panels).
    order : list[str] | None
        Explicit cell-type column order.
    figsize : (float, float) | None
        Override auto-sized figure dimensions.
    col_width : float
        Inches per cell-type column.
    height : float
        Figure height in inches.
    save : str | None
        Output file path (dpi=300).
    return_matrices : bool
        Return dict of ordered matrices instead of plotting.
    seed : int
        Random seed.
    order_rows : bool
        Apply diagonal k-means ordering (default True).
    row_order : pd.DataFrame | None
        Return value from a previous call; replays that exact row order.
    transpose : bool
        Stack panels vertically instead of side-by-side.

    Returns
    -------
    pd.DataFrame
        Ordered feature table. Contains 'peak'+'gene' columns in links mode,
        or 'gene'/'peak' + 'cluster' columns otherwise.

    Examples
    --------
    # RNA + gene activity (same gene space)
    plot_heatmap(rna=rna_avg, atac=act_avg_archr, genes=marker_genes, order=ct_order)

    # ATAC + RNA with peak-to-gene links
    plot_heatmap(rna=rna_avg, atac=atac_avg, links=link_dict, order=ct_order)

    # Multiple RNA panels
    plot_heatmap(rna=[rna_avg_1, rna_avg_2], genes=marker_genes)
    """
    from sklearn.cluster import KMeans
    from scipy.cluster.hierarchy import linkage as hc_linkage, leaves_list
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    np.random.seed(seed)

    # ── 0. Normalise inputs ───────────────────────────────────────────────
    rna_list  = [rna]  if (rna  is not None and not isinstance(rna,  list)) else (rna  or [])
    atac_list = [atac] if (atac is not None and not isinstance(atac, list)) else (atac or [])

    if not rna_list and not atac_list:
        raise ValueError("Provide at least one of 'rna' or 'atac'.")

    # ── 1. Build panels list ──────────────────────────────────────────────
    # Each panel: dict(adata, features, label, modality)
    panels = []
    use_links    = links is not None
    replay_order = row_order is not None
    _link_dict   = {}  # populated only in links mode
    _gene_groups       = None  # gene -> group_name, set when genes is a string-keyed dict
    _gene_groups_order = []    # groups in user-specified order

    if use_links:
        # Parse links dict (same logic as plot_peak2gene_heatmap)
        if isinstance(links, pd.DataFrame):
            _link_dict = (
                {c: links[links['cluster'] == c] for c in links['cluster'].unique()}
                if 'cluster' in links.columns
                else {'_all': links}
            )
        else:
            _link_dict = links

        first_df = next(iter(_link_dict.values()))
        gene_col = next(
            (c for c in ['gene', 'TargetGene'] if c in first_df.columns),
            first_df.columns[-1],
        )

        if replay_order:
            lo    = row_order
            valid = pd.Series([True] * len(lo), index=lo.index)
            if atac_list:
                valid &= lo['peak'].isin(set(atac_list[0].var_names))
            if rna_list:
                valid &= lo['gene'].isin(set(rna_list[0].var_names))
            lo = lo[valid].reset_index(drop=True)
            _lp  = lo['peak'].tolist()
            _lg  = lo['gene'].tolist()
            _lct = lo['cell_type'].tolist()  if 'cell_type'   in lo.columns else [''] * len(lo)
            _lts = lo['tss'].tolist()        if 'tss'         in lo.columns else [None] * len(lo)
            _lsc = lo['correlation'].tolist() if 'correlation' in lo.columns else [None] * len(lo)
        else:
            _lp, _lg, _lct, _lts, _lsc = [], [], [], [], []
            atac_vars = set(atac_list[0].var_names) if atac_list else None
            rna_vars  = set(rna_list[0].var_names)  if rna_list  else None
            for ct, ldf in _link_dict.items():
                for _, row in ldf.iterrows():
                    pv, gv = row['peak'], row[gene_col]
                    if (atac_vars is None or pv in atac_vars) and \
                       (rna_vars  is None or gv in rna_vars):
                        _lp.append(pv);  _lg.append(gv)
                        _lct.append(ct); _lts.append(row.get('tss', None))
                        _lsc.append(row.get('correlation', None))
            if not _lp and not _lg:
                raise ValueError("No valid peak-gene pairs found in links.")

        for i, adata in enumerate(atac_list):
            panels.append(dict(
                adata=adata, features=_lp,
                label=(labels[len(panels)] if labels is not None
                       else ('ATAC' if len(atac_list) == 1 else f'ATAC {i+1}')),
                modality='atac',
            ))
        for i, adata in enumerate(rna_list):
            panels.append(dict(
                adata=adata, features=_lg,
                label=(labels[len(panels)] if labels is not None
                       else ('RNA' if len(rna_list) == 1 else f'RNA {i+1}')),
                modality='rna',
            ))

    else:
        from scalex.pp.markers import filter_marker_dict

        def _prefilter(feat_spec, adata_list):
            """Filter feat_spec to features shared across all adatas."""
            if feat_spec is None or not adata_list:
                return feat_spec
            is_list = not isinstance(feat_spec, dict)
            d = {'_all': feat_spec} if is_list else dict(feat_spec)
            for adata in adata_list:
                d = filter_marker_dict(d, adata.var_names)
            return [g for gs in d.values() for g in gs] if is_list else d

        genes = _prefilter(genes, rna_list)
        peaks = _prefilter(peaks, atac_list)

        _feat_dict = genes if isinstance(genes, dict) else peaks
        if isinstance(_feat_dict, dict) and not all(isinstance(k, int) for k in _feat_dict):
            _gene_groups       = {g: grp for grp, gs in _feat_dict.items() for g in gs}
            _gene_groups_order = list(_feat_dict.keys())

        def _resolve_features(feat_spec, i, adata):
            if feat_spec is None:
                return None
            if isinstance(feat_spec, dict):
                if all(isinstance(k, int) for k in feat_spec):
                    feat = feat_spec[i]   # per-panel spec: {0: [...], 1: [...]}
                else:
                    feat = [g for gs in feat_spec.values() for g in gs]  # scanpy marker dict
            else:
                feat = feat_spec
            return [f for f in feat if f in adata.var_names]

        for i, adata in enumerate(rna_list):
            feat = _resolve_features(genes, i, adata)
            if feat is None:
                raise ValueError("'genes' must be provided for RNA panels.")
            panels.append(dict(
                adata=adata, features=feat,
                label=(labels[len(panels)] if labels is not None
                       else ('RNA' if len(rna_list) == 1 else f'RNA {i+1}')),
                modality='rna',
            ))
        for i, adata in enumerate(atac_list):
            # ATAC falls back to genes when peaks not provided
            feat = _resolve_features(peaks, i, adata) or _resolve_features(genes, i, adata)
            if feat is None:
                raise ValueError("Provide 'peaks' or 'genes' for ATAC panels.")
            panels.append(dict(
                adata=adata, features=feat,
                label=(labels[len(panels)] if labels is not None
                       else ('ATAC' if len(atac_list) == 1 else f'ATAC {i+1}')),
                modality='atac',
            ))

        if replay_order:
            feat_col = 'gene' if 'gene' in row_order.columns else 'peak'
            for p in panels:
                p_vars = set(p['adata'].var_names)
                p['features'] = [f for f in row_order[feat_col] if f in p_vars]

    # ── 2. Extract & z-score matrices ─────────────────────────────────────
    def _extract(adata, features):
        X = adata[:, features].X
        return X.toarray().T if hasattr(X, 'toarray') else np.array(X).T

    cell_types = panels[0]['adata'].obs[groupby].tolist()
    n_ct = len(cell_types)

    for p in panels:
        p['M'] = _row_zscore(_extract(p['adata'], p['features']))

    M_anchor = panels[0]['M']
    n_rows   = M_anchor.shape[0]

    # ── 3. K-means on anchor ──────────────────────────────────────────────
    do_order = order_rows and not replay_order
    if show_gene_labels == 'bracket' or show_gene_labels is True:
        do_order = False  # preserve original group order so brackets are contiguous
    if do_order:
        _k = k if k is not None else (len(_link_dict) if use_links else min(n_ct, n_rows))
        _k = min(_k, n_rows)
        km_labels = KMeans(n_clusters=_k, random_state=seed, n_init=10).fit_predict(M_anchor)
    else:
        _k, km_labels = 1, np.zeros(n_rows, dtype=int)

    # ── 4. Column order ───────────────────────────────────────────────────
    cluster_means = np.array([
        M_anchor[km_labels == cid].mean(axis=0) if (km_labels == cid).any()
        else np.zeros(n_ct)
        for cid in range(_k)
    ])

    if order is not None:
        ct_index  = {ct: i for i, ct in enumerate(cell_types)}
        col_order = np.array([ct_index[ct] for ct in order if ct in ct_index])
    else:
        col_order = (
            leaves_list(hc_linkage(cluster_means.T, method='ward'))
            if n_ct > 1 else np.arange(n_ct)
        )

    # ── 5. Row order: diagonal ────────────────────────────────────────────
    if do_order:
        peak_pos      = np.argmax(cluster_means[:, col_order], axis=1)
        cluster_order = np.argsort(peak_pos)
        row_idx = []
        for old_cid in cluster_order:
            idxs     = np.where(km_labels == old_cid)[0]
            dominant = col_order[int(peak_pos[old_cid])]
            within   = np.argsort(-M_anchor[idxs, dominant])
            row_idx.extend(idxs[within])
        row_idx = np.array(row_idx)
    else:
        row_idx = np.arange(n_rows)

    ct_ord = [cell_types[i] for i in col_order]
    n_ct   = len(ct_ord)

    for p in panels:
        p['M_ord'] = p['M'][row_idx][:, col_order]

    # ── 6. Build return DataFrame ─────────────────────────────────────────
    if use_links:
        ret_df = pd.DataFrame({
            'Chromosome': [_parse_peak(_lp[i])[0] for i in row_idx],
            'Start':      [_parse_peak(_lp[i])[1] for i in row_idx],
            'End':        [_lts[i]  for i in row_idx],
            'peak':       [_lp[i]   for i in row_idx],
            'gene':       [_lg[i]   for i in row_idx],
            'score':      [_lsc[i]  for i in row_idx],
        })
    else:
        anchor = panels[0]
        feat_col = 'gene' if anchor['modality'] == 'rna' else 'peak'
        ret_df = pd.DataFrame({
            feat_col:  [anchor['features'][i] for i in row_idx],
            'cluster': [km_labels[i]          for i in row_idx],
        })

    if return_matrices:
        return {p['label']: p['M_ord'] for p in panels} | {'cell_types': ct_ord, 'df': ret_df}

    # ── 7. Per-panel colours ──────────────────────────────────────────────
    _default_cmap = {'atac': 'RdBu_r', 'rna': 'viridis'}
    if cmaps is None:
        panel_cmaps = [_default_cmap[p['modality']] for p in panels]
    elif isinstance(cmaps, str):
        panel_cmaps = [cmaps] * len(panels)
    else:
        panel_cmaps = list(cmaps)

    panel_limits = [limits] * len(panels) if isinstance(limits, tuple) else list(limits)
    n_panels = len(panels)

    # ── 8. Figure layout ──────────────────────────────────────────────────
    # Fixed-size colorbars, centered in the right strip
    cb_left    = 0.895
    cb_w       = 0.018
    cb_h_fixed = 0.08
    cb_gap     = 0.02

    if transpose:
        gs_top, gs_bottom = 0.95, 0.05
        hm_h    = n_ct * col_width
        total_h = n_panels * hm_h + (n_panels - 1) * 0.1
        fig_w   = height
        fig_h   = total_h + 1.5
        if figsize is not None:
            fig_w, fig_h = figsize

        fig = plt.figure(figsize=(fig_w, fig_h))
        ratios = []
        for pi in range(n_panels):
            ratios.append(hm_h)
            if pi < n_panels - 1:
                ratios.append(0.1)
        gs_left = 0.20 if (show_gene_labels and _gene_groups) else 0.12
        gs = gridspec.GridSpec(
            len(ratios), 1, height_ratios=ratios, hspace=0,
            left=gs_left, right=0.87, top=gs_top, bottom=gs_bottom,
        )
        axes = [fig.add_subplot(gs[2 * pi, 0]) for pi in range(n_panels)]

    else:
        gs_top, gs_bottom = 0.88, 0.22
        hm_w  = n_ct * col_width
        fig_w = n_panels * hm_w + (n_panels - 1) * 0.1 + 0.4 + 1.0
        fig_h = height
        if figsize is not None:
            fig_w, fig_h = figsize

        fig = plt.figure(figsize=(fig_w, fig_h))
        ratios = []
        for pi in range(n_panels):
            ratios.append(hm_w)
            if pi < n_panels - 1:
                ratios.append(0.1)
        gs_left = 0.22 if ((show_gene_labels and _gene_groups) or gene_ticks) else 0.08
        gs = gridspec.GridSpec(
            1, len(ratios), width_ratios=ratios, wspace=0,
            left=gs_left, right=0.87, top=gs_top, bottom=gs_bottom,
        )
        axes = [fig.add_subplot(gs[0, 2 * pi]) for pi in range(n_panels)]

    # Shared colorbar placement: fixed-height stack, top-aligned with heatmap
    total_cb_h = n_panels * cb_h_fixed + (n_panels - 1) * cb_gap
    cb_top     = gs_top
    cb_axes    = [
        fig.add_axes([cb_left,
                      cb_top - (pi + 1) * cb_h_fixed - pi * cb_gap,
                      cb_w, cb_h_fixed])
        for pi in range(n_panels)
    ]

    # ── 9. Draw heatmaps ──────────────────────────────────────────────────
    for p, ax, cmap, lim in zip(panels, axes, panel_cmaps, panel_limits):
        mat    = p['M_ord'].T if transpose else p['M_ord']
        n_feat = p['M_ord'].shape[0]
        ax.imshow(mat, aspect='auto', cmap=cmap,
                  vmin=lim[0], vmax=lim[1], interpolation='none')
        if title is not None:
            ax.set_title(title, fontsize=5, pad=2)
        if transpose:
            ax.set_yticks(range(n_ct))
            ax.set_yticklabels(ct_ord, fontsize=5)
            ax.set_xticks([])
        else:
            ax.set_xticks(range(n_ct))
            ax.set_xticklabels(ct_ord, rotation=45, ha='right', fontsize=5)
            ax.set_yticks([])
        ax.set_frame_on(False)
        ax.grid(False)

    # ── 9b. Gene group annotations ────────────────────────────────────────
    ax_bar = None
    if show_gene_labels and _gene_groups:
        anchor_features = panels[0]['features']
        ordered_groups  = [_gene_groups.get(anchor_features[i], '') for i in row_idx]

        group_spans = []
        for ri, grp in enumerate(ordered_groups):
            if not group_spans or group_spans[-1][0] != grp:
                group_spans.append([grp, ri, ri])
            else:
                group_spans[-1][2] = ri

        style   = 'bracket' if show_gene_labels is True else show_gene_labels
        ax0     = axes[0]
        n_total = len(ordered_groups)

        if style == 'bar':
            unique_groups = list(dict.fromkeys(ordered_groups))
            if color_map is None:
                palette   = plt.cm.get_cmap('tab20', max(len(unique_groups), 1))
                grp_color = {g: palette(i) for i, g in enumerate(unique_groups)}
            elif isinstance(color_map, dict):
                grp_color = color_map
            else:
                grp_color = dict(zip(unique_groups, color_map))
            from matplotlib.colors import to_rgba
            color_arr = np.array([to_rgba(grp_color[g]) for g in ordered_groups])  # (n, 4)

            ax0_pos = ax0.get_position()
            bar_w, bar_gap = 0.012, 0.0
            if transpose:
                ax_bar = fig.add_axes([ax0_pos.x0,
                                       ax0_pos.y0 - bar_w - bar_gap,
                                       ax0_pos.width, bar_w])
                ax_bar.imshow(color_arr[np.newaxis, :, :], aspect='auto')
            else:
                ax_bar = fig.add_axes([ax0_pos.x0 - bar_w - bar_gap,
                                       ax0_pos.y0,
                                       bar_w, ax0_pos.height])
                ax_bar.imshow(color_arr[:, np.newaxis, :], aspect='auto')
            ax_bar.set_xticks([]); ax_bar.set_yticks([])
            ax_bar.set_frame_on(False)

            from matplotlib.patches import Patch
            legend_order = [g for g in _gene_groups_order if g in grp_color]
            handles = [Patch(facecolor=grp_color[g], label=g) for g in legend_order]
            fig.legend(handles=handles, fontsize=5, frameon=False, ncol=1,
                       loc='lower left',
                       bbox_to_anchor=(0.87, gs_bottom),
                       bbox_transform=fig.transFigure)

        else:  # bracket
            pad = 0.3
            for grp_name, start, end in group_spans:
                center = (start + end) / 2
                if transpose:
                    ax0.annotate(
                        '', xy=(end + pad, -0.6), xytext=(start - pad, -0.6),
                        xycoords='data', textcoords='data', annotation_clip=False,
                        arrowprops=dict(arrowstyle='|-|,widthA=0.3,widthB=0.3',
                                        color='black', lw=0.8))
                    ax0.text(center, -1.4, grp_name, ha='center', va='top',
                             fontsize=5, clip_on=False, rotation=45)
                else:
                    ax0.annotate(
                        '', xy=(-0.6, end + pad), xytext=(-0.6, start - pad),
                        xycoords='data', textcoords='data', annotation_clip=False,
                        arrowprops=dict(arrowstyle='|-|,widthA=0.3,widthB=0.3',
                                        color='black', lw=0.8))
                    ax0.text(-1.0, center, grp_name, ha='right', va='center',
                             fontsize=5, clip_on=False)

    # ── 9c. Gene tick annotations ─────────────────────────────────────────
    if gene_ticks:
        anchor_features  = panels[0]['features']
        ordered_features = [anchor_features[i] for i in row_idx]
        feat_to_row      = {f: ri for ri, f in enumerate(ordered_features)}

        _tick_list    = (
            [g for gs in gene_ticks.values() for g in gs]
            if isinstance(gene_ticks, dict) else list(gene_ticks)
        )
        ticks_to_show = [g for g in _tick_list if g in feat_to_row]
        if ticks_to_show:
            n_rows_total = len(ordered_features)

            # Sort by heatmap row order → no crossings
            ticks_to_show.sort(key=lambda g: feat_to_row[g])

            # Pack labels close to gene rows; push apart only enough to prevent font overlap
            min_gap   = max(1.0, n_rows_total / 35)
            label_pos = np.array([feat_to_row[g] for g in ticks_to_show], dtype=float)
            for i in range(1, len(label_pos)):
                label_pos[i] = max(label_pos[i], label_pos[i - 1] + min_gap)
            for i in range(len(label_pos) - 2, -1, -1):
                label_pos[i] = min(label_pos[i], label_pos[i + 1] - min_gap)
            label_pos = np.clip(label_pos, 0, n_rows_total - 1)

            # Place ax_ticks to the LEFT of ax_bar (or heatmap if no bar)
            ref_pos     = ax_bar.get_position() if ax_bar is not None else axes[0].get_position()
            ticks_w_fig = 0.07
            ax_ticks = fig.add_axes([
                ref_pos.x0 - ticks_w_fig, ref_pos.y0, ticks_w_fig, ref_pos.height
            ])
            ax_ticks.set_xlim(0, 1)
            ax_ticks.set_ylim(n_rows_total - 0.5, -0.5)  # inverted, matches heatmap
            ax_ticks.axis('off')

            lw = 0.5
            kw = dict(color='black', lw=lw, clip_on=False)

            for g, ly in zip(ticks_to_show, label_pos):
                gy = feat_to_row[g]
                ax_ticks.plot([0.00, 0.20], [ly, ly], **kw)   # left horizontal
                ax_ticks.plot([0.20, 0.90], [ly, gy], **kw)   # angled middle
                ax_ticks.plot([0.90, 1.00], [gy, gy], **kw)   # short right stub → bar
                ax_ticks.text(-0.02, ly, g, ha='right', va='center',
                              fontsize=gene_tick_fontsize, clip_on=False)

    # ── 10. Colorbars ─────────────────────────────────────────────────────
    for p, ax_cb, cmap, lim in zip(panels, cb_axes, panel_cmaps, panel_limits):
        sm = ScalarMappable(cmap=cmap, norm=Normalize(*lim))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=ax_cb, orientation='vertical')
        cb.set_label(p['label'] + ' z-score', fontsize=5, labelpad=3, rotation=270, va='bottom')
        cb.set_ticks([lim[0], lim[1]])
        ax_cb.tick_params(labelsize=4, length=2, pad=1)
        for spine in ax_cb.spines.values():
            spine.set_visible(False)

    if save is not None:
        fig.savefig(save, bbox_inches='tight', dpi=300)

    return ret_df

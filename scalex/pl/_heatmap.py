import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools


def get_module_series(topgenes): # topgenes are DataFrame with ranks as index, GEPs as columns
    long_df = topgenes.stack().reset_index()
    long_df.columns = ['Rank', 'GEP', 'Gene']
    long_df = long_df.sort_values('Rank', ascending=True)
    gene_series = long_df.drop_duplicates(subset='Gene', keep='first').set_index('Gene')['GEP']
    return gene_series

def local_correlation_plot(
            local_correlation_z, modules, linkage=None,
            mod_cmap='tab10', vmin=-1, vmax=1, color_map=None,
            z_cmap='RdBu_r', yticklabels=False, save=False,
):
    # 1. ALIGNMENT: Ensure modules Series matches the matrix row order exactly
    # This prevents errors if 'modules' and 'local_correlation_z' are sorted differently
    modules_aligned = modules.reindex(local_correlation_z.index)

    if color_map is None:
        unique_modules = sorted(modules_aligned.unique())
        
        # itertools.cycle ensures we don't run out of colors if we have many modules
        palette_cycle = itertools.cycle(sns.color_palette(mod_cmap))
        
        module_color_lookup = {}
        for mod in unique_modules:
            if mod == -1:
                module_color_lookup[mod] = '#ffffff' # White for noise
            else:
                module_color_lookup[mod] = next(palette_cycle)
    else:
        module_color_lookup = color_map

    # Map the aligned modules to their colors using the lookup dictionary
    row_colors_series = modules_aligned.map(module_color_lookup)
    row_colors_series.name = "Modules" # This becomes the header in the plot

    # 3. PLOTTING
    cm = sns.clustermap(
        local_correlation_z,
        row_linkage=linkage,
        col_linkage=linkage,
        vmin=vmin,
        vmax=vmax,
        cmap=z_cmap,
        xticklabels=False,
        yticklabels=yticklabels,
        row_colors=row_colors_series.to_frame(), # Convert Series to DataFrame
        rasterized=True,
    )

    fig = plt.gcf()
    plt.sca(cm.ax_heatmap)
    plt.ylabel("")
    plt.xlabel("")
    
    # Hide the dendrogram tree
    cm.ax_row_dendrogram.remove()

    # 4. REORDERING LOGIC
    # Get the row indices used by the clustermap (calculated or provided)
    if cm.dendrogram_row is not None:
        reordered_indices = cm.dendrogram_row.reordered_ind
    else:
        reordered_indices = np.arange(len(modules_aligned))

    # Apply these indices to the ALIGNED modules series
    mod_reordered = modules_aligned.iloc[reordered_indices]

    # Calculate positions for the text labels
    mod_map = {}
    y = np.arange(modules_aligned.size)

    for x in mod_reordered.unique():
        if x == -1:
            continue
        # Find the center position of this module in the reordered list
        mod_map[x] = y[mod_reordered == x].mean()

    # Add text labels
    plt.sca(cm.ax_row_colors)
    for mod, mod_y in mod_map.items():
        plt.text(-0.5, y=mod_y, s="{}".format(mod),
                 horizontalalignment='right',
                 verticalalignment='center')
    plt.xticks([])

    # 5. COLORBAR OPTIMIZATION
    # Access the colorbar axes directly via cm.cax
    if cm.cax:
        cm.cax.set_ylabel('Correlation')
        cm.cax.yaxis.set_label_position("left")

    cbar = cm.ax_heatmap.collections[0].colorbar
    cbar.solids.set_rasterized(True)

    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
 
    return cm



def _row_zscore(X):
    """Z-score each row; rows with zero variance are left as zeros."""
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True)
    sigma[sigma == 0] = 1
    return (X - mu) / sigma


    
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
    col_width=0.25,
    height=7,
    save=None,
    return_matrices=False,
    seed=1,
    order_links=True,
    links_order=None,
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
        ct_list   = lo['cell_type'].tolist() if 'cell_type' in lo.columns else [''] * len(lo)
    else:
        peak_list, gene_list, ct_list = [], [], []
        for ct, links_df in _link_dict.items():
            for _, row in links_df.iterrows():
                peak, gene = row['peak'], row[gene_col]
                if peak in atac_avg.var_names and gene in rna_avg.var_names:
                    peak_list.append(peak)
                    gene_list.append(gene)
                    ct_list.append(ct)
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
        peak_list = [peak_list[i] for i in keep]
        gene_list = [gene_list[i] for i in keep]
        ct_list   = [ct_list[i]   for i in keep]

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

    links_df = pd.DataFrame({
        'peak':      [peak_list[i] for i in row_order],
        'gene':      [gene_list[i] for i in row_order],
        'cell_type': [ct_list[i]   for i in row_order],
        'cluster':   km_labels[row_order],
    })

    if return_matrices:
        return dict(ATAC=M_atac_ord, RNA=M_rna_ord, cell_types=ct_ord,
                    km_labels=km_labels[row_order],
                    split_positions=split_pos[:-1],
                    links=links_df)

    # ── 7. Layout ─────────────────────────────────────────────────────────
    cb_width  = 0.15          # colorbar column (inches)
    gap       = 0.3           # white space between ATAC and RNA heatmaps
    cb_pad    = 0.2           # padding left of colorbar column
    margin    = 0.4
    hm_width  = n_ct * col_width
    fig_width = 2 * hm_width + cb_width + gap + cb_pad + margin
    if figsize is not None:
        fig_width, height = figsize

    fig = plt.figure(figsize=(fig_width, height))

    # Main grid: ATAC | gap | RNA | cb_pad | colorbar-column
    gs = gridspec.GridSpec(
        1, 5,
        width_ratios=[hm_width, gap, hm_width, cb_pad, cb_width],
        wspace=0,
        left=0.08, right=0.97, top=0.88, bottom=0.22,
    )
    ax_atac = fig.add_subplot(gs[0, 0])
    ax_rna  = fig.add_subplot(gs[0, 2])

    # Stacked colorbars: padding rows shrink bars to ~30% of column height each
    gs_cb = gridspec.GridSpecFromSubplotSpec(
        5, 1,
        subplot_spec=gs[0, 4],
        height_ratios=[1, 2, 1, 2, 1],
        hspace=0,
    )
    ax_cb_atac = fig.add_subplot(gs_cb[1])
    ax_cb_rna  = fig.add_subplot(gs_cb[3])

    # ── 8. Heatmaps ───────────────────────────────────────────────────────
    for ax, mat, limits, cmap, title in [
        (ax_atac, M_atac_ord, limits_atac, cmap_atac, f'ATAC  ({len(peak_list)} links)'),
        (ax_rna,  M_rna_ord,  limits_rna,  cmap_rna,  f'RNA   ({len(gene_list)} links)'),
    ]:
        ax.imshow(mat, aspect='auto', cmap=cmap,
                  vmin=limits[0], vmax=limits[1], interpolation='none')
        ax.set_title(title, fontsize=7, pad=3)
        ax.set_xticks(range(n_ct))
        ax.set_xticklabels(ct_ord, rotation=45, ha='right', fontsize=6)
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.grid(False)

    # ── 9. Colorbars (stacked, right corner) ──────────────────────────────
    for ax_cb, limits, cmap, label in [
        (ax_cb_atac, limits_atac, cmap_atac, 'ATAC Z'),
        (ax_cb_rna,  limits_rna,  cmap_rna,  'RNA Z'),
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

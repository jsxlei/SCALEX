import numpy as np
import scanpy as sc
import pandas as pd
from collections import Counter
from anndata import concat

from scalex.io import aggregate_data


def _max_group_detection(adata, groupby) -> np.ndarray:
    """Max over groups of the per-feature detection fraction (cells with X>0).

    Returns a 1-D array of length ``n_vars``: for each feature, the highest
    fraction of cells in which it is detected across all groups.
    """
    import scipy.sparse as sp
    binary = adata.X > 0
    best = np.zeros(adata.n_vars, dtype=float)
    for g in adata.obs[groupby].cat.categories:
        mask = (adata.obs[groupby] == g).values
        n = int(mask.sum())
        if n == 0:
            continue
        sub = binary[mask]
        s = np.asarray(sub.sum(axis=0)).ravel() if sp.issparse(sub) else np.asarray(sub).sum(axis=0).ravel()
        best = np.maximum(best, s / n)
    return best


def get_markers(
        adata,
        groupby='cell_type',
        pval_cutoff=0.01,
        logfc_cutoff=1.5,
        min_cells=10,
        top_n=300,
        processed=False,
        filter_pseudo=True,
        min_cell_per_batch=100,
        method='wilcoxon',
        force=False,
        pval_key='pvals_adj',
        min_pct=0.0,
    ):
    """
    Get markers filtered by detection fraction, p-value and log fold change.

    Parameters
    ----------
    logfc_cutoff
        0.58 ≈ 1.5 fold change (log2(1.5)), 1.0 ≈ 2 fold change
    min_cell_per_batch
        Minimum number of cells required per batch
    pval_key
        Column used for the p-value filter: 'pvals_adj' (BH-adjusted, default) or
        'pvals' (raw).
    min_pct
        Minimum detection fraction (fraction of cells with non-zero signal). A
        feature must be detected in ≥ ``min_pct`` of the cells of its own group
        to be called a marker. The same threshold also pre-filters features
        before testing (keeping those detected in ≥ ``min_pct`` of at least one
        group), which removes ultra-sparse noise and shrinks the multiple-testing
        burden so that BH-adjusted p-values stay meaningful even for hundreds of
        thousands of peaks. ``0`` disables both (legacy behaviour).
    """
    from scalex.pp.annotation import format_rna

    adata = adata.copy()
    if filter_pseudo:
        adata = format_rna(adata)

    adata.obs[groupby] = adata.obs[groupby].astype('category')

    if not processed:
        sc.pp.normalize_total(adata, target_sum=10000)
        sc.pp.log1p(adata)

    # Prevalence pre-filter — keep features detected in ≥ min_pct of cells in at
    # least one group. Shrinks the BH denominator so adjusted p-values remain
    # usable for very large feature sets (e.g. peaks).
    recompute = force or 'rank_genes_groups' not in adata.uns
    if min_pct and min_pct > 0:
        keep = _max_group_detection(adata, groupby) >= min_pct
        if keep.any() and not keep.all():
            adata = adata[:, keep].copy()
            recompute = True

    if recompute:
        sc.tl.rank_genes_groups(adata, groupby=groupby, method=method, pts=True)

    markers_dict = {}
    for cluster in adata.obs[groupby].cat.categories:
        df = sc.get.rank_genes_groups_df(adata, group=cluster)
        mask = (df[pval_key] < pval_cutoff) & (df['logfoldchanges'] > logfc_cutoff)
        if min_pct and min_pct > 0 and 'pct_nz_group' in df.columns:
            mask &= df['pct_nz_group'] >= min_pct
        filtered = df[mask]
        markers_dict[cluster] = filtered.sort_values('scores', ascending=False).head(top_n)['names'].values

    return markers_dict


# Substrings that mark an obs column as a sequencing-depth/coverage variable, which is
# log10-transformed before bias matching (ArchR uses log10(nFrags)). TSS-like columns
# are used as-is.
_DEPTH_HINTS = ('nfrag', 'ncount', 'fragment', 'depth', 'reads', 'cutsite', 'n_counts')


def _resolve_bias_cols(adata, bias):
    """Resolve the bias obs columns. ``bias=True`` auto-detects an ArchR-like pair:
    a TSS-enrichment column (if present) plus one sequencing-depth column."""
    if bias is True:
        cols = []
        for c in adata.obs.columns:
            if 'tss' in c.lower() and 'enrich' in c.lower():
                cols.append(c)
                break
        for cand in ('nFrags', 'nCount_ATAC', 'atac_fragments', 'atac_peak_region_fragments', 'n_counts'):
            if cand in adata.obs.columns:
                cols.append(cand)
                break
        if not cols:
            raise ValueError("bias=True but no TSS-enrichment or depth column found in adata.obs; "
                             "pass an explicit list of obs columns to `bias`.")
        return cols
    return list(bias)


def _bias_matrix(adata, bias_cols):
    """Build a z-scored cell x bias matrix; depth-like columns are log10-transformed."""
    mats = []
    for c in bias_cols:
        v = np.asarray(adata.obs[c]).astype(float)
        if any(h in c.lower() for h in _DEPTH_HINTS):
            v = np.log10(v + 1.0)
        s = v.std()
        v = (v - v.mean()) / s if s > 0 else v * 0.0
        mats.append(v)
    return np.vstack(mats).T


def find_markers_archr(adata, groupby='cell_type', bias=True, set_type='peak',
                       pval_cutoff=0.05, logfc_cutoff=None, top_n=300,
                       pval_key='pvals', min_pct=0.05, method='wilcoxon',
                       bg_ratio=1, processed=False, filter_pseudo=None, seed=0):
    """ArchR ``getMarkerFeatures``-style marker finding (standalone).

    Self-contained: does its own normalization + prevalence pre-filter and does NOT
    use ``get_markers``/``diagonal_heatmap``. For each group it tests that group's
    cells against a **bias-matched background** of non-group cells (nearest neighbours
    in standardized bias space; ArchR bias = TSS enrichment + log10 depth) with a
    Wilcoxon test, then keeps features passing ``pval_key < pval_cutoff`` and
    ``Log2FC > logfc_cutoff`` (top ``top_n`` by score). Returns ``{group: np.array(features)}``.

    Parameters
    ----------
    bias
        ``True`` -> auto-detect an ArchR-like pair from ``.obs`` (a TSS-enrichment column
        if present, plus a sequencing-depth column), or pass an explicit list of obs
        column names. Depth-like columns are log10-transformed; all are z-scored.
    set_type
        Only sets the default ``logfc_cutoff`` ('peak' -> 0.25, else 1.25) and whether
        ``format_rna`` pseudogene filtering is applied (genes only). ArchR's Log2FC>=1.25
        is defined on ArchR's own normalization and does not transfer to this log1p
        matrix, hence the gentler peak default.

    Approximates ArchR (bias-matched background + Wilcoxon + FDR cutoff); not bit-identical.
    """
    from sklearn.neighbors import NearestNeighbors

    if logfc_cutoff is None:
        logfc_cutoff = 0.25 if set_type == 'peak' else 1.25
    if filter_pseudo is None:
        filter_pseudo = set_type != 'peak'

    adata = adata.copy()
    if filter_pseudo:
        from scalex.pp.annotation import format_rna
        adata = format_rna(adata)
    adata.obs[groupby] = adata.obs[groupby].astype('category')
    if not processed:
        sc.pp.normalize_total(adata, target_sum=10000)
        sc.pp.log1p(adata)
    # Prevalence pre-filter: keep features detected in >= min_pct of >=1 group.
    if min_pct and min_pct > 0:
        keep = _max_group_detection(adata, groupby) >= min_pct
        if keep.any() and not keep.all():
            adata = adata[:, keep].copy()

    bias_cols = _resolve_bias_cols(adata, bias)
    print(f"ArchR-style markers: bias-matched background on {bias_cols}")
    X = _bias_matrix(adata, bias_cols)
    labels = adata.obs[groupby].astype(str).values
    cats = [str(c) for c in adata.obs[groupby].cat.categories]

    markers = {}
    for g in cats:
        in_g = labels == g
        grp_idx = np.where(in_g)[0]
        other = np.where(~in_g)[0]
        if grp_idx.size < 1 or other.size < 1:
            markers[g] = np.array([])
            continue
        k = min(max(1, int(bg_ratio)), other.size)
        nn = NearestNeighbors(n_neighbors=k).fit(X[other])
        _, nbr = nn.kneighbors(X[grp_idx])
        bg_idx = np.unique(other[nbr.ravel()])

        sub = adata[np.concatenate([grp_idx, bg_idx])].copy()
        grp = np.array(['grp'] * grp_idx.size + ['bg'] * bg_idx.size)
        sub.obs['_archr_grp'] = pd.Categorical(grp, categories=['bg', 'grp'])
        sc.tl.rank_genes_groups(sub, '_archr_grp', groups=['grp'], reference='bg',
                                method=method, pts=True)
        df = sc.get.rank_genes_groups_df(sub, group='grp')
        mask = (df[pval_key] < pval_cutoff) & (df['logfoldchanges'] > logfc_cutoff)
        if min_pct and min_pct > 0 and 'pct_nz_group' in df.columns:
            mask &= df['pct_nz_group'] >= min_pct
        filtered = df[mask].sort_values('scores', ascending=False)
        if top_n and top_n > 0:
            filtered = filtered.head(top_n)
        markers[g] = filtered['names'].values
        print(g, len(markers[g]))
    return markers


def flatten_dict(markers: dict) -> np.ndarray:
    """Flatten a marker dict to a unique sorted array of all genes.

    Parameters
    ----------
    markers : dict
        Mapping of cluster label to list of gene names.

    Returns
    -------
    np.ndarray
        Unique sorted array of all gene names across all clusters.
    """
    return np.unique([item for sublist in markers.values() for item in sublist])


def flatten_list(lists: list) -> list:
    """Flatten a list of lists into a single list.

    Parameters
    ----------
    lists : list[list]
        Nested list to flatten.

    Returns
    -------
    list
        Concatenated flat list.
    """
    return [item for sublist in lists for item in sublist]


def filter_marker_dict(markers: dict, var_names) -> dict:
    """Filter each cluster's gene list to genes present in var_names.

    Parameters
    ----------
    markers : dict
        Mapping of cluster label to list of gene names.
    var_names : list[str] or Index
        Reference gene list (e.g. ``adata.var_names``).

    Returns
    -------
    dict
        Same structure as ``markers`` with genes not in ``var_names`` removed.
    """
    return {cluster: [i for i in genes if i in var_names] for cluster, genes in markers.items()}


def rename_marker_dict(markers, rename_dict):
    """
    Rename dictionary keys and merge values if multiple keys map to the same new key.
    """
    marker_dict = {}
    for cluster, genes in markers.items():
        if cluster not in rename_dict:
            marker_dict[cluster] = genes
            continue
        new_key = rename_dict[cluster]
        if new_key in marker_dict:
            marker_dict[new_key].extend(genes)
        else:
            marker_dict[new_key] = genes.copy()

    for key in marker_dict:
        marker_dict[key] = list(dict.fromkeys(marker_dict[key]))

    return marker_dict


def cluster_program(adata_avg, n_clusters: int = 25, method: str = 'hclust') -> dict:
    """Cluster genes into programs using k-means or hierarchical clustering.

    Parameters
    ----------
    adata_avg : AnnData
        Pseudobulk expression matrix (cell types × genes).
    n_clusters : int, default 25
        Number of gene clusters.
    method : {'hclust', 'kmeans'}, default 'hclust'
        Clustering method.

    Returns
    -------
    dict
        Mapping of cluster label (str) to list of gene names.
    """
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        adata_avg.var['cluster'] = np.array(kmeans.fit_predict(adata_avg.X.T)).astype(str)
        gene_cluster_dict = adata_avg.var.groupby('cluster').groups
        return {k: v.tolist() for k, v in gene_cluster_dict.items()}
    elif method == 'hclust':
        from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
        X = adata_avg.X.T
        Z = linkage(X, method='ward')
        labels = fcluster(Z, n_clusters, criterion='maxclust').astype(str)

        leaf_pos = np.empty(len(leaves_list(Z)), dtype=int)
        leaf_pos[leaves_list(Z)] = np.arange(len(leaf_pos))

        adata_avg.var['cluster'] = labels
        adata_avg.var['_dend_order'] = leaf_pos
        return {
            k: grp.sort_values('_dend_order').index.tolist()
            for k, grp in adata_avg.var.groupby('cluster')
        }


def find_gene_program(adata, groupby='cell_type', processed=False, n_clusters=25, top_n=300, filter_pseudo=True, cluster_method='kmeans', **kwargs):
    """Find gene program for each cell type."""
    adata = adata.copy()
    adata_avg = aggregate_data(adata, groupby=groupby, processed=processed, scale=True)
    markers = get_markers(adata, groupby=groupby, processed=processed, top_n=top_n, filter_pseudo=filter_pseudo, **kwargs)
    for cluster, genes in markers.items():
        print(cluster, len(genes))
    marker_list = flatten_dict(markers)
    adata_avg_ = adata_avg[:, marker_list].copy()
    gene_cluster_dict = cluster_program(adata_avg_, n_clusters=n_clusters, method=cluster_method)
    return gene_cluster_dict, adata_avg


def find_peak_program(adata, groupby='cell_type', processed=False, n_clusters=25, top_n=-1, pval_cutoff=0.05, logfc_cutoff=1., filter_pseudo=False, pval_key='pvals', min_pct=0.05, **kwargs):
    """Find peak program for each cell type."""
    return find_gene_program(adata, groupby=groupby, processed=processed, n_clusters=n_clusters, top_n=top_n, filter_pseudo=filter_pseudo,
                             pval_cutoff=pval_cutoff, logfc_cutoff=logfc_cutoff, pval_key=pval_key, min_pct=min_pct, **kwargs)


def _process_group(args):
    """Helper function for multiprocessing."""
    adata_, groupby, set_type, top_n, filter_pseudo, kwargs = args
    if set_type == 'gene':
        filter_pseudo = True
    elif set_type == 'peak':
        filter_pseudo = False
        if 'pval_cutoff' not in kwargs:
            kwargs['pval_cutoff'] = 0.05
        if 'logfc_cutoff' not in kwargs:
            kwargs['logfc_cutoff'] = 1.
        if 'pval_key' not in kwargs:
            kwargs['pval_key'] = 'pvals'
        if 'min_pct' not in kwargs:
            kwargs['min_pct'] = 0.05

    group_counts = adata_.obs[groupby].value_counts()
    valid_groups = group_counts[group_counts >= 2].index
    if len(valid_groups) < 2:
        print("Skipping as it has less than 2 groups with 2 or more samples")
        return None

    adata_ = adata_[adata_.obs[groupby].isin(valid_groups)].copy()
    markers = get_markers(adata_, groupby=groupby, top_n=top_n, filter_pseudo=filter_pseudo, **kwargs)
    return flatten_dict(markers)


def find_consensus_program(adata, groupby='cell_type', across=None, set_type='gene', processed=False, top_n=-1, occurance=None, min_samples=2, n_jobs=None, n_clusters=None, **kwargs):
    """
    Find consensus program for each cell type across multiple samples.

    Parameters
    ----------
    across
        Column name in adata.obs to split data across
    occurance
        Minimum number of occurrences across groups
    n_jobs
        Number of parallel jobs. None uses all available cores.
    """
    adata.obs[groupby] = adata.obs[groupby].astype('category')
    if n_clusters is None:
        n_clusters = len(adata.obs[groupby].cat.categories)
    occurance = occurance or max(2, len(np.unique(adata.obs[across])) // 2)
    filter_pseudo = set_type == 'gene'

    if across is not None:
        args_list = []
        adata_avg_list = []
        for c in np.unique(adata.obs[across]):
            adata_ = adata[adata.obs[across] == c].copy()
            adata_ = adata_[adata_.obs.dropna(subset=[groupby]).index].copy()
            args_list.append((adata_, groupby, set_type, top_n, filter_pseudo, kwargs))
            adata_avg_c = aggregate_data(adata_, groupby=groupby, processed=processed, scale=True)
            adata_avg_c.obs[across] = c
            adata_avg_list.append(adata_avg_c)

        adata_avg = concat(adata_avg_list)

        if n_jobs == 1:
            results = [_process_group(args) for args in args_list]
        else:
            from multiprocessing import Pool, cpu_count
            if n_jobs is None:
                n_jobs = min(cpu_count(), 32)
            with Pool(n_jobs) as pool:
                results = pool.map(_process_group, args_list)

        markers_list = [r for r in results if r is not None]
        if not markers_list:
            raise ValueError("No valid groups found with sufficient samples")

        markers_list = np.concatenate(markers_list)
        gene_counts = Counter(markers_list)
        markers_list = np.array([gene for gene, count in gene_counts.items() if count >= occurance])
        print('There are {} {set_type}s with at least {} occurrences'.format(len(markers_list), occurance, set_type=set_type))

    adata_avg.obs[groupby + '_' + across] = adata_avg.obs[groupby].astype(str) + '_' + adata_avg.obs[across].astype(str)
    adata_avg_ = adata_avg[:, markers_list].copy()
    gene_cluster_dict = cluster_program(adata_avg_, n_clusters=n_clusters)
    return gene_cluster_dict, adata_avg


def find_consensus_links(link_dict_a, link_dict_b):
    """
    Find consensus peak-to-gene links supported in both datasets.

    Parameters
    ----------
    link_dict_a, link_dict_b
        DataFrames with columns ['peak', 'gene'], or dicts of such DataFrames.

    Returns
    -------
    pd.DataFrame with columns: peak, gene.
    """
    def to_df(obj):
        if isinstance(obj, pd.DataFrame):
            return obj[['peak', 'gene']].drop_duplicates()
        return pd.concat(
            [df[['peak', 'gene']] for df in obj.values()],
            ignore_index=True,
        ).drop_duplicates()

    df_a = to_df(link_dict_a)
    df_b = to_df(link_dict_b)
    return df_a.merge(df_b, on=['peak', 'gene']).reset_index(drop=True)


def get_rank_dict(adata, cell_type: str = 'cell_type', n_top: int = 100, to_dict: bool = True):
    """Extract top ranked genes per cluster from ``adata.uns['rank_genes_groups']``.

    Parameters
    ----------
    adata : AnnData
        Data with differential expression results in ``uns``.
    cell_type : str, default 'cell_type'
        Key used when computing rank_genes_groups (if not already computed).
    n_top : int, default 100
        Number of top genes to retrieve per cluster.
    to_dict : bool, default True
        If True, return a dict mapping cluster → gene list.
        If False, return a DataFrame.

    Returns
    -------
    dict or pd.DataFrame
        Top ranked genes per cluster.
    """
    if 'rank_genes_groups' not in adata.uns:
        sc.tl.rank_genes_groups(adata, cell_type)
    df = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(n_top)
    if to_dict:
        return df.to_dict(orient='list')
    return df

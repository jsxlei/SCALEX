import numpy as np
import scanpy as sc
import pandas as pd
from collections import Counter
from anndata import concat

from scalex.io import aggregate_data


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
    ):
    """
    Get markers filtered by both p-value and log fold change.

    Parameters
    ----------
    logfc_cutoff
        0.58 ≈ 1.5 fold change (log2(1.5)), 1.0 ≈ 2 fold change
    min_cell_per_batch
        Minimum number of cells required per batch
    """
    from scalex.pp.annotation import format_rna

    adata = adata.copy()
    if filter_pseudo:
        adata = format_rna(adata)

    markers_dict = {}
    adata.obs[groupby] = adata.obs[groupby].astype('category')
    clusters = adata.obs[groupby].cat.categories

    if 'rank_genes_groups' not in adata.uns or force:
        if not processed:
            sc.pp.normalize_total(adata, target_sum=10000)
            sc.pp.log1p(adata)
        sc.tl.rank_genes_groups(adata, groupby=groupby, method=method)

    for cluster in clusters:
        df = sc.get.rank_genes_groups_df(adata, group=cluster)
        filtered = df[
            (df['pvals_adj'] < pval_cutoff) &
            (df['logfoldchanges'] > logfc_cutoff)
        ].copy()
        markers_dict[cluster] = filtered.sort_values('scores', ascending=False).head(top_n)['names'].values

    return markers_dict


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


def find_peak_program(adata, groupby='cell_type', processed=False, n_clusters=25, top_n=-1, pval_cutoff=0.05, logfc_cutoff=1., filter_pseudo=False, **kwargs):
    """Find peak program for each cell type."""
    return find_gene_program(adata, groupby=groupby, processed=processed, top_n=top_n, filter_pseudo=filter_pseudo,
                             pval_cutoff=pval_cutoff, logfc_cutoff=logfc_cutoff, **kwargs)


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

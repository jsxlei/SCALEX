import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

from scalex.atac.snapatac2._diff import aggregate_X
from scalex.data import aggregate_data

from scalex.atac.bedtools import intersect_bed, bed_to_df, df_to_bed
from scalex.pp.annotation import format_rna, format_atac
from scalex.analysis import get_markers, flatten_dict, flatten_list
# from scalex.linkage.utils import row_wise_correlation

class Linkage:
    def __init__(
            self, 
            rna=None, atac=None, 
            rna_avg=None, atac_avg=None, 
            groupby='cell_type', 
            order=None,
            gtf=None, up=100_000, down=100_000):
        self.rna = rna
        self.atac = atac
        self.gtf = gtf if gtf is not None else os.path.expanduser('~/.scalex/gencode.v38.annotation.gtf.gz')
        self.up = up
        self.down = down

        if rna_avg is None or atac_avg is None:
            self.format_data()
            self.aggregate_data(groupby=groupby)
        else:
            self.rna_avg = rna_avg
            self.atac_avg = atac_avg
        self.get_edges(up=up, down=down)
        self.get_peak_gene_corr()
        self.adj_dicts = {}

    # def reorder_cell_types(self, order):
    #     self.rna_avg = self.rna_avg[order, :]
    #     self.atac_avg = self.atac_avg[order, :]

    def format_data(self):
        self.rna = format_rna(self.rna.copy(), gtf=self.gtf)
        self.atac = format_atac(self.atac.copy())
        self.rna_var = self.rna.var
        self.atac_var = self.atac.var

    def aggregate_data(self, groupby="cell_type"):
        self.rna_avg = aggregate_data(self.rna, groupby=groupby)
        self.atac_avg = aggregate_data(self.atac, groupby=groupby)

        common = set(self.rna_avg.obs[groupby].values) & set(self.atac_avg.obs[groupby].values)
        self.rna_avg = self.rna_avg[self.rna_avg.obs[groupby].isin(common)]
        self.atac_avg = self.atac_avg[self.atac_avg.obs[groupby].isin(common)]
        print(common)

    def get_linked_peak_adj(self, corr_threshold=0.7):
        adj_dict = {}
        for i in range(len(self.peak_gene_corr)):
            gene = self.rna_var.index[self.edges[0, i]]
            peak = self.atac_var.index[self.edges[1, i]]
            if gene not in adj_dict:
                adj_dict[gene] = []
            if self.peak_gene_corr[i] > corr_threshold:
                adj_dict[gene].append(peak)
        self.adj_dicts[corr_threshold] = adj_dict
        # return adj_dict
        
    def get_linked_peaks(self, gene_dict, corr_threshold=0.7, return_unlinked_genes=False):
        if corr_threshold not in self.adj_dicts:
            self.get_linked_peak_adj(corr_threshold=corr_threshold)
        adj_dict = self.adj_dicts[corr_threshold]
        linked_peaks = {}
        unlinked_genes = {}
        for name, gene_list in gene_dict.items():
            linked_peaks[name] = flatten_list([adj_dict[gene] for gene in gene_list if gene in adj_dict])
            unlinked_genes[name] = [gene for gene in gene_list if gene not in adj_dict]
        if return_unlinked_genes:
            return linked_peaks, unlinked_genes
        return linked_peaks
    


    def get_cell_type_links(self, marker_genes, marker_peaks, groupby="cell_type"):
        links = {}
        shared_keys = set(marker_genes.keys()) & set(marker_peaks.keys())   
        for cell_type in shared_keys:
            rna_avg_ct = self.rna_avg[:, marker_genes[cell_type]].copy()
            atac_avg_ct = self.atac_avg[:, marker_peaks[cell_type]].copy()
            if rna_avg_ct.shape[0] == 0 or atac_avg_ct.shape[0] == 0:
                continue
            edge = intersect_bed(rna_avg_ct.var, atac_avg_ct.var, up=self.up, down=self.down, add_distance=True)
            # print(len(edge[0]), len(edge[1]))
            index_to_gene = {i: gene for i, gene in enumerate(rna_avg_ct.var.index)}
            index_to_peak = {i: peak for i, peak in enumerate(atac_avg_ct.var.index)}
            gene_list = [index_to_gene[i] for i in edge[0]]
            peak_list = [index_to_peak[i] for i in edge[1]]
            links[cell_type] = list(zip(gene_list, peak_list))
        return links
    
    def link_peak_to_gene(self, links):
        links = flatten_dict(links)
        peak_to_gene = dict(zip(links[1], links[0]))
        return peak_to_gene

    def get_edges(self, up=100_000, down=100_000):
        # assert self.rna_agg.obs[self.groupby] == self.atac_agg.obs[self.groupby], "groupby should have the same order"
        ## peak to gene linkage
        self.edges = intersect_bed(self.rna_var, self.atac_var, up=up, down=down, add_distance=True)
        self.gene_to_index = {gene: i for i, gene in enumerate(self.rna_var.index)}
        self.peak_to_index = {peak: i for i, peak in enumerate(self.atac_var.index)}    


    def get_peak_gene_corr(self):
        peak_vec = self.atac_avg.to_df().iloc[:, self.edges[1]].T.values
        genes_vec = self.rna_avg.to_df().iloc[:, self.edges[0]].T.values

        self.peak_gene_corr = row_wise_correlation(genes_vec, peak_vec) 

    def filter_peak_gene_linkage(self, threshold=0.7):
        indices = np.where(self.peak_gene_corr > threshold)[0]
        gene_indices = self.edges[0][indices]
        peak_indices = self.edges[1][indices]

        genes = self.rna_var.iloc[gene_indices].index.values
        peaks = self.atac_var.iloc[peak_indices].index.values

        return genes, peaks

    def find_overlapping_peak_gene_linkage(self, genes_deg, peaks_deg, threshold=0.7):
        genes, peaks = self.filter_peak_gene_linkage(threshold=threshold)
        if isinstance(genes_deg, dict):
            genes_deg = flatten_dict(genes_deg)
        if isinstance(peaks_deg, dict):
            peaks_deg = flatten_dict(peaks_deg)

        gene_peak_pair = [(genes[i], peaks[i]) for i, _ in enumerate(genes) if genes[i] in genes_deg and peaks[i] in peaks_deg]

        return gene_peak_pair

    def get_cell_type_peak_gene_links(self, marker_genes, marker_peaks, correlation_threshold=0.7):
        """
        Identify linked peaks and genes for each cell type.
        
        Parameters:
        -----------
        cell_type_genes : dict
            Dictionary mapping cell types to lists of cell type specific genes
        cell_type_peaks : dict
            Dictionary mapping cell types to lists of cell type specific peaks
        correlation_threshold : float, optional (default=0.7)
            Minimum correlation threshold for peak-gene pairs
            
        Returns:
        --------
        dict
            Dictionary mapping cell types to lists of (gene, peak) tuples
        """
        cell_type_links = {}
        
        # Get all potential peak-gene pairs based on genomic distance and correlation
        genes, peaks = self.filter_peak_gene_linkage(threshold=correlation_threshold)
        
        # Create lookup dictionaries for faster membership testing
        gene_to_peak = {gene: peak for gene, peak in zip(genes, peaks)}
        peak_to_gene = {peak: gene for gene, peak in zip(genes, peaks)}
        
        # Process each cell type
        for cell_type in marker_genes.keys():
            if cell_type not in marker_peaks:
                continue
                
            # Get cell type specific genes and peaks
            ct_genes = set(marker_genes[cell_type])
            ct_peaks = set(marker_peaks[cell_type])
            
            # Find overlapping genes and peaks that are linked
            linked_pairs = []
            for gene in ct_genes:
                if gene in gene_to_peak and gene_to_peak[gene] in ct_peaks:
                    linked_pairs.append((gene, gene_to_peak[gene]))
            
            cell_type_links[cell_type] = linked_pairs
            
        return cell_type_links

def row_wise_correlation(arr1, arr2, epsilon=1e-8):
    """
    Calculates the Pearson correlation coefficient between corresponding rows of two NumPy arrays,
    with robustness to small values and division by zero.

    Parameters:
    - arr1: NumPy array of shape (m, n)
    - arr2: NumPy array of shape (m, n)
    - epsilon: Small constant to avoid division by zero (default: 1e-8)

    Returns:
    - correlations: NumPy array of shape (m,)
    """
    assert arr1.shape == arr2.shape, "Arrays must have the same shape."

    # Compute means
    mean1 = np.mean(arr1, axis=1, keepdims=True)
    mean2 = np.mean(arr2, axis=1, keepdims=True)

    # Compute standard deviations
    std1 = np.std(arr1, axis=1, ddof=1, keepdims=True)
    std2 = np.std(arr2, axis=1, ddof=1, keepdims=True)

    # Avoid division by zero by adding epsilon
    safe_std1 = np.where(std1 < epsilon, np.nan, std1)
    safe_std2 = np.where(std2 < epsilon, np.nan, std2)

    # Standardize the data (z-scores)
    z1 = (arr1 - mean1) / safe_std1
    z2 = (arr2 - mean2) / safe_std2

    # Compute sum of products of z-scores
    sum_of_products = np.nansum(z1 * z2, axis=1)

    # Degrees of freedom
    n = arr1.shape[1]
    degrees_of_freedom = n - 1

    # Compute Pearson correlation coefficients
    correlations = sum_of_products / degrees_of_freedom

    return correlations


def _row_zscore(mat, epsilon=1e-8):
    """Z-score each row of a 2-D array. Rows with near-zero std become 0."""
    mean = mat.mean(axis=1, keepdims=True)
    std  = mat.std(axis=1, keepdims=True)
    return np.where(std < epsilon, 0.0, (mat - mean) / std)


def _filter_overlapping_knn(knn_idx, overlap_cutoff):
    """Greedily retain KNN groups whose pair-wise overlap < overlap_cutoff * k."""
    k = knn_idx.shape[1]
    threshold = int(np.floor(overlap_cutoff * k))
    kept = [0]
    kept_sets = [set(knn_idx[0])]
    for i in range(1, len(knn_idx)):
        s = set(knn_idx[i])
        if all(len(s & ks) <= threshold for ks in kept_sets):
            kept.append(i)
            kept_sets.append(s)
    return np.array(kept)


def add_peak2gene_links(
    atac,
    rna,
    groupby='cell_type',
    use_rep='X_lsi',
    k=100,
    knn_iteration=500,
    overlap_cutoff=0.8,
    cor_cutoff=0.45,
    fdr_cutoff=1e-4,
    var_cutoff_atac=0.25,
    var_cutoff_rna=0.25,
    window=250_000,
    scale_to=10_000,
    log2_norm=True,
    seed=1,
):
    """
    Mirror of ArchR's addPeak2GeneLinks.

    Builds KNN pseudo-bulk groups from the ATAC LSI embedding, aggregates
    ATAC and RNA counts per group (sum → library-size normalize → log2),
    then computes Pearson correlation + BH-FDR for all peak-gene pairs
    within `window` bp of each gene's TSS.

    atac and rna must share barcodes (multiome / paired data). For unpaired
    data, use compute_peak2gene_links with cell-type averages instead.

    Parameters
    ----------
    atac : AnnData   Single-cell ATAC with obsm[use_rep] and obs[groupby].
    rna  : AnnData   Single-cell RNA sharing barcodes with atac.
    groupby : str    Cell-type column for annotating KNN groups.
    use_rep : str    LSI/PCA embedding key in atac.obsm (default 'X_lsi').
    k : int          Cells per KNN group (ArchR default: 100).
    knn_iteration : int  Number of KNN groups to attempt (ArchR default: 500).
    overlap_cutoff : float  Max fraction of shared cells between groups (default 0.8).
    cor_cutoff : float   Minimum Pearson r to keep a link (default 0.45).
    fdr_cutoff : float   Maximum BH-adjusted p-value (default 1e-4).
    var_cutoff_atac, var_cutoff_rna : float
        Variance quantile cutoffs for feature pre-filtering (default 0.25).
    window : int     Genomic window around TSS (bp; default 250 000).
    scale_to : float Library-size normalization target (default 10 000).
    log2_norm : bool Log2(x+1) transform after normalization (default True).
    seed : int       Random seed.

    Returns
    -------
    dict with keys:
      'links'           : pd.DataFrame  [peak, gene, correlation, pval, fdr]
      'knn_atac'        : np.ndarray    (n_groups, n_peaks_filt)
      'knn_rna'         : np.ndarray    (n_groups, n_genes_filt)
      'group_celltypes' : list[str]     dominant cell type per KNN group
      'atac_var_names'  : pd.Index      peak names for knn_atac columns
      'rna_var_names'   : pd.Index      gene names for knn_rna columns

    Example
    -------
    p2g = add_peak2gene_links(atac, rna, groupby='cell_type')
    fig = plot_peak2gene_heatmap(p2g, order=['Foamy', 'Resident', 'Inflammatory'])
    """
    from scipy import sparse, stats
    from statsmodels.stats.multitest import multipletests
    from sklearn.neighbors import NearestNeighbors
    from scalex.pp.annotation import format_rna, format_atac
    from scalex.atac.bedtools import intersect_bed

    np.random.seed(seed)

    if use_rep not in atac.obsm:
        raise ValueError(f"'{use_rep}' not in atac.obsm. Available: {list(atac.obsm.keys())}")

    # ── 1. Build KNN pseudo-bulk groups ───────────────────────────────────
    emb    = atac.obsm[use_rep]
    n_cells = len(atac)
    n_query = min(knn_iteration, n_cells)
    query_idx = np.random.choice(n_cells, n_query, replace=False)

    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(emb)
    _, knn_idx = nn.kneighbors(emb[query_idx])   # (n_query, k)

    keep       = _filter_overlapping_knn(knn_idx, overlap_cutoff)
    knn_groups = knn_idx[keep]                    # (n_groups, k)
    n_groups   = len(knn_groups)
    print(f"  KNN groups: {n_groups}  (from {n_query} queries, k={k})")

    # ── 2. Align RNA to ATAC barcodes ─────────────────────────────────────
    rna_bc_idx = {bc: i for i, bc in enumerate(rna.obs_names)}
    atac_to_rna = np.array([rna_bc_idx.get(bc, -1) for bc in atac.obs_names])
    n_paired = (atac_to_rna >= 0).sum()
    if n_paired < 0.5 * n_cells:
        raise ValueError(
            f"Only {n_paired}/{n_cells} ATAC barcodes found in rna. "
            "For unpaired data use compute_peak2gene_links with cell-type averages."
        )

    def _to_dense(X):
        return X.toarray() if sparse.issparse(X) else np.array(X)

    atac_X = _to_dense(atac.X)                      # (n_cells, n_peaks)
    rna_X  = np.zeros((n_cells, rna.n_vars))
    valid  = atac_to_rna >= 0
    rna_X[valid] = _to_dense(rna.X)[atac_to_rna[valid]]

    # ── 3. Aggregate: sum → library-size normalize → log2 ─────────────────
    def _agg(X, groups):
        mat    = np.stack([X[g].sum(axis=0) for g in groups])  # (n_groups, n_feat)
        totals = mat.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1
        mat = mat / totals * scale_to
        return np.log2(mat + 1) if log2_norm else mat

    atac_mat = _agg(atac_X, knn_groups)   # (n_groups, n_peaks)
    rna_mat  = _agg(rna_X,  knn_groups)   # (n_groups, n_genes)

    # ── 4. Variance filter ────────────────────────────────────────────────
    atac_pass = atac_mat.var(axis=0) >= np.quantile(atac_mat.var(axis=0), var_cutoff_atac)
    rna_pass  = rna_mat.var(axis=0)  >= np.quantile(rna_mat.var(axis=0),  var_cutoff_rna)

    atac_mat_f = atac_mat[:, atac_pass]
    rna_mat_f  = rna_mat[:, rna_pass]
    atac_names = atac.var_names[atac_pass]
    rna_names  = rna.var_names[rna_pass]

    # ── 5. Genomic proximity edges ────────────────────────────────────────
    atac_var_f = format_atac(atac[:, atac_pass].copy()).var
    rna_var_f  = format_rna(rna[:, rna_pass].copy()).var

    edges = intersect_bed(rna_var_f, atac_var_f, up=window, down=window, add_distance=False)
    gene_idx, peak_idx = edges[0], edges[1]
    if len(gene_idx) == 0:
        raise ValueError("No peak-gene pairs found within the genomic window.")

    # ── 6. Pearson correlation + t-test + BH-FDR ─────────────────────────
    peak_vecs = atac_mat_f[:, peak_idx].T    # (n_pairs, n_groups)
    gene_vecs = rna_mat_f[:, gene_idx].T     # (n_pairs, n_groups)
    corr = row_wise_correlation(peak_vecs, gene_vecs)

    tstat = corr * np.sqrt((n_groups - 2) / np.maximum(1 - corr**2, 1e-16))
    pval  = 2 * stats.t.sf(np.abs(tstat), df=n_groups - 2)
    _, fdr, _, _ = multipletests(pval, method='fdr_bh')

    links = pd.DataFrame({
        'peak':        atac_names[peak_idx],
        'gene':        rna_names[gene_idx],
        'correlation': corr,
        'pval':        pval,
        'fdr':         fdr,
    })
    links = links[(links['correlation'] >= cor_cutoff) & (links['fdr'] <= fdr_cutoff)]
    links = links.reset_index(drop=True)
    print(f"  Links after filtering: {len(links)}")

    # ── 7. Dominant cell type per KNN group ───────────────────────────────
    ct_arr = np.array(atac.obs[groupby].tolist())
    group_celltypes = []
    for g in knn_groups:
        cts, counts = np.unique(ct_arr[g], return_counts=True)
        group_celltypes.append(cts[np.argmax(counts)])

    return {
        'links':           links,
        'knn_atac':        atac_mat_f,
        'knn_rna':         rna_mat_f,
        'group_celltypes': group_celltypes,
        'atac_var_names':  atac_names,
        'rna_var_names':   rna_names,
    }


def compute_peak2gene_links(
    atac_avg,
    rna_avg,
    peak_dict=None,
    gene_list=None,
    groupby='cell_type',
    cor_cutoff=0.45,
    var_cutoff_atac=0.25,
    var_cutoff_rna=0.25,
    window=250_000,
    gtf=None,
):
    """
    Compute peak-to-gene correlation links from cell-type averaged matrices.

    Mirrors ArchR's addPeak2GeneLinks logic but uses cell-type averages:
      1. Variance-filter peaks and genes to high-variance features.
      2. Find candidate pairs by genomic proximity (peaks within `window` of gene TSS).
      3. Compute Pearson correlation for each candidate pair across cell types.
      4. Filter by `cor_cutoff`.
      5. Organise output by the cluster in `peak_dict` (if provided).

    Parameters
    ----------
    atac_avg : AnnData
        Cell-type-averaged ATAC (obs = cell types, var = peaks).
        Peak names must be parseable as 'chr:start-end'.
    rna_avg : AnnData
        Cell-type-averaged RNA (obs = cell types, var = genes).
    peak_dict : dict or None
        {cluster: [peak, ...]} from find_peak_program.
        If provided, output is organised by cluster; peaks not in peak_dict
        are ignored. If None, all peaks in atac_avg are used and the output
        has a single 'all' key.
    gene_list : list[str] | dict[str, list[str]] | None
        If provided, only these genes are considered when computing links.
        If a dict, all values are flattened to a unique set. If None, all
        genes passing the variance filter are used.
    groupby : str
        Cell-type column in obs (only used to align obs between atac_avg and rna_avg).
    cor_cutoff : float
        Minimum Pearson r to retain a link (default 0.45, same as ArchR).
    var_cutoff_atac : float
        Variance quantile cutoff for peaks (0–1). Peaks below this quantile
        are dropped before correlation. Default 0.25 (same as ArchR).
    var_cutoff_rna : float
        Variance quantile cutoff for genes (0–1). Default 0.25.
    window : int
        Genomic window around gene TSS to search for peaks (bp). Default 250 000.
    gtf : pr.PyRanges | str | None
        GTF source for TSS annotation. Pass the result of
        ``read_gencode_transcripts()`` to use MANE_Select canonical TSS
        coordinates (recommended when loop anchor coordinates come from that
        source). If None, falls back to the default GTF path used by
        ``format_rna`` (~/.scalex/gencode.v38.annotation.gtf.gz).

    Returns
    -------
    link_dict : dict  {cluster: pd.DataFrame}
        Each DataFrame has columns ['peak', 'gene', 'correlation'].
        Pass directly to plot_peak2gene_heatmap(..., links=link_dict).

    Example
    -------
    link_dict = compute_peak2gene_links(atac_avg, rna_avg,
                                        peak_dict=marker_peaks)
    plot_peak2gene_heatmap(atac_avg, rna_avg, link_dict)
    """
    from scalex.pp.annotation import format_rna, format_atac
    from scalex.atac.bedtools import intersect_bed
    from scalex.linkage.linkage import row_wise_correlation

    # ── 1. Add genomic coordinates to .var ───────────────────────────────
    atac_avg = format_atac(atac_avg.copy())
    fmt_kwargs = {'gtf': gtf} if gtf is not None else {}
    rna_avg  = format_rna(rna_avg.copy(), **fmt_kwargs)

    # ── 2. Align cell-type order between ATAC and RNA ─────────────────────
    common_cts = list(dict.fromkeys(
        ct for ct in atac_avg.obs[groupby] if ct in rna_avg.obs[groupby].values
    ))
    atac_avg = atac_avg[atac_avg.obs[groupby].isin(common_cts)]
    rna_avg  = rna_avg[rna_avg.obs[groupby].isin(common_cts)]
    # Ensure same order
    atac_avg = atac_avg[atac_avg.obs[groupby].argsort()]
    rna_avg  = rna_avg[rna_avg.obs[groupby].argsort()]

    # ── 3. Variance filter ────────────────────────────────────────────────
    def _get_X(adata):
        X = adata.X
        return X.toarray() if hasattr(X, 'toarray') else np.array(X)

    atac_X = _get_X(atac_avg)   # (n_ct, n_peaks)
    rna_X  = _get_X(rna_avg)    # (n_ct, n_genes)

    peak_var = atac_X.var(axis=0)
    gene_var  = rna_X.var(axis=0)

    peak_pass = peak_var >= np.quantile(peak_var, var_cutoff_atac)
    gene_pass = gene_var  >= np.quantile(gene_var,  var_cutoff_rna)

    atac_filt = atac_avg[:, peak_pass]
    rna_filt  = rna_avg[:, gene_pass]

    # Further restrict to peaks in peak_dict if provided
    if peak_dict is not None:
        all_dict_peaks = set(p for peaks in peak_dict.values() for p in peaks)
        keep_peaks = [p for p in atac_filt.var_names if p in all_dict_peaks]
        atac_filt = atac_filt[:, keep_peaks]

    # Further restrict to genes in gene_list if provided
    if gene_list is not None:
        if isinstance(gene_list, dict):
            gene_set = set(g for genes in gene_list.values() for g in genes)
        else:
            gene_set = set(gene_list)
        keep_genes = [g for g in rna_filt.var_names if g in gene_set]
        rna_filt = rna_filt[:, keep_genes]

    atac_X_filt = _get_X(atac_filt)   # (n_ct, n_peaks_filt)
    rna_X_filt  = _get_X(rna_filt)    # (n_ct, n_genes_filt)

    # ── 4. Genomic proximity edges ────────────────────────────────────────
    # edges[0] = gene indices, edges[1] = peak indices
    edges = intersect_bed(
        rna_filt.var, atac_filt.var,
        up=window, down=window, add_distance=False,
    )
    gene_idx, peak_idx = edges[0], edges[1]

    if len(gene_idx) == 0:
        raise ValueError("No peak-gene pairs found within the genomic window.")

    # ── 5. Pearson correlation for each proximal pair ─────────────────────
    # row_wise_correlation expects (n_pairs, n_ct) arrays
    peak_vecs = atac_X_filt[:, peak_idx].T   # (n_pairs, n_ct)
    gene_vecs = rna_X_filt[:, gene_idx].T    # (n_pairs, n_ct)

    corr = row_wise_correlation(peak_vecs, gene_vecs)   # (n_pairs,)

    # ── 6. Filter by cor_cutoff ───────────────────────────────────────────
    keep = np.where(corr >= cor_cutoff)[0]
    if len(keep) == 0:
        raise ValueError(f"No pairs passed cor_cutoff={cor_cutoff}. Try lowering it.")

    peaks_kept = atac_filt.var_names[peak_idx[keep]]
    genes_kept = rna_filt.var_names[gene_idx[keep]]
    corr_kept  = corr[keep]

    tss_kept = rna_filt.var.iloc[gene_idx[keep]]['tss'].values
    links_df = pd.DataFrame({'peak': peaks_kept, 'gene': genes_kept, 'correlation': corr_kept, 'tss': tss_kept})

    # ── 7. Organise by peak_dict cluster ─────────────────────────────────
    if peak_dict is None:
        return {'all': links_df}

    peak_to_cluster = {p: c for c, peaks in peak_dict.items() for p in peaks}
    link_dict = {}
    for cluster in peak_dict:
        cluster_df = links_df[links_df['peak'].map(peak_to_cluster) == cluster].reset_index(drop=True)
        if not cluster_df.empty:
            link_dict[cluster] = cluster_df

    return link_dict


def plot_peak2gene_heatmap_archr(
    p2g,
    groupby=None,
    order=None,
    pal_group=None,
    k_kmeans=25,
    n_plot=25000,
    limits_atac=(-2, 2),
    limits_rna=(-2, 2),
    cmap_atac='YlOrRd',
    cmap_rna='RdYlBu_r',
    col_width=0.015,
    annot_height=0.25,
    height=6,
    save=None,
    return_matrices=False,
    seed=1,
):
    """
    Plot a peak-to-gene heatmap from add_peak2gene_links output.

    This is the direct Python equivalent of ArchR's plotPeak2GeneHeatmap.
    Columns are KNN pseudo-bulk groups (already aggregated and normalized),
    so no additional per-cell smoothing is needed.

    Parameters
    ----------
    p2g : dict          Output of add_peak2gene_links.
    groupby : str|None  Ignored (cell types come from p2g['group_celltypes']).
    order : list|None   Explicit cell-type order for columns.
    pal_group : dict|None  {cell_type: color}. Auto-generated from tab20 if None.
    k_kmeans : int      K-means clusters for row ordering (default 25).
    n_plot : int        Max links to display (default 25 000).
    limits_atac, limits_rna : (float, float)  Z-score color clipping.
    cmap_atac : str     ATAC colormap (default 'YlOrRd', like ArchR solarExtra).
    cmap_rna  : str     RNA colormap (default 'RdYlBu_r', like ArchR blueYellow).
    col_width : float   Figure width per KNN-group column (inches; default 0.015).
    annot_height : float  Height of annotation bar (inches; default 0.25).
    height : float      Total figure height (inches; default 6).
    save : str|None     Output path.
    return_matrices : bool  Return dict of matrices instead of plotting.
    seed : int          Random seed.

    Example
    -------
    p2g = add_peak2gene_links(atac, rna, groupby='cell_type')
    fig = plot_peak2gene_heatmap(p2g, order=['Foamy', 'Resident', 'Inflammatory'])
    """
    from sklearn.cluster import KMeans
    from scipy.cluster.hierarchy import linkage as hclust, leaves_list
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    np.random.seed(seed)

    links           = p2g['links']
    knn_atac        = p2g['knn_atac']          # (n_groups, n_peaks_filt)
    knn_rna         = p2g['knn_rna']           # (n_groups, n_genes_filt)
    group_celltypes = np.array(p2g['group_celltypes'])
    atac_var_names  = p2g['atac_var_names']
    rna_var_names   = p2g['rna_var_names']

    if links.empty:
        raise ValueError("No links in p2g. Lower cor_cutoff or fdr_cutoff in add_peak2gene_links.")

    # ── 1. Map links to column indices in knn_atac / knn_rna ─────────────
    atac_idx = {name: i for i, name in enumerate(atac_var_names)}
    rna_idx  = {name: i for i, name in enumerate(rna_var_names)}

    valid    = links['peak'].isin(atac_idx) & links['gene'].isin(rna_idx)
    links    = links[valid].reset_index(drop=True)
    peak_col = np.array([atac_idx[p] for p in links['peak']])
    gene_col = np.array([rna_idx[g]  for g in links['gene']])

    # ── 2. Extract and z-score per-link matrices ──────────────────────────
    # knn_atac[:, peak_col].T → (n_links, n_groups), then z-score rows
    M_atac_raw = knn_atac[:, peak_col].T    # (n_links, n_groups)
    M_rna_raw  = knn_rna[:, gene_col].T

    M_atac_z = _row_zscore(M_atac_raw)
    M_rna_z  = _row_zscore(M_rna_raw)

    n_links  = len(links)
    n_groups = knn_atac.shape[0]

    # ── 3. Column order: cell-type blocks, hierarchical within ────────────
    unique_cts = list(dict.fromkeys(group_celltypes))
    ct_seq     = [ct for ct in order if ct in unique_cts] \
                 if order is not None else unique_cts

    col_order, ct_boundaries = [], []
    for ct in ct_seq:
        idx = np.where(group_celltypes == ct)[0]
        if not idx.size:
            continue
        if len(idx) > 1:
            idx = idx[leaves_list(hclust(M_atac_z[idx], method='ward'))]
        col_order.extend(idx)
        ct_boundaries.append(len(col_order))
    col_order = np.array(col_order)

    # ── 4. Row order: K-means + diagonal ─────────────────────────────────
    if n_links > n_plot:
        keep     = np.random.choice(n_links, n_plot, replace=False)
        M_atac_z = M_atac_z[keep]
        M_rna_z  = M_rna_z[keep]
        links    = links.iloc[keep].reset_index(drop=True)
        n_links  = n_plot

    k_km = min(k_kmeans, n_links)
    km   = KMeans(n_clusters=k_km, random_state=seed, n_init=10)
    # cluster rows (links) using ordered columns
    km_labels = km.fit_predict(M_atac_z[:, col_order])

    cluster_means    = np.array([
        M_atac_z[km_labels == c][:, col_order].mean(axis=0)
        if (km_labels == c).any() else np.zeros(len(col_order))
        for c in range(k_km)
    ])
    cluster_peak_col = np.argmax(cluster_means, axis=1)
    cluster_diag_ord = np.argsort(cluster_peak_col)

    row_order, split_pos = [], []
    for old_c in cluster_diag_ord:
        idxs    = np.where(km_labels == old_c)[0]
        dom     = int(cluster_peak_col[old_c])
        dom_col = col_order[dom] if dom < len(col_order) else col_order[0]
        within  = np.argsort(-M_atac_z[idxs, dom_col])
        row_order.extend(idxs[within])
        split_pos.append(len(row_order))
    row_order = np.array(row_order)

    M_atac = M_atac_z[row_order][:, col_order]   # (n_links, n_groups)
    M_rna  = M_rna_z[row_order][:, col_order]

    if return_matrices:
        return dict(ATAC=M_atac, RNA=M_rna,
                    group_celltypes=group_celltypes[col_order].tolist(),
                    km_labels=km_labels[row_order],
                    split_positions=split_pos[:-1],
                    ct_boundaries=ct_boundaries[:-1],
                    peaks=links['peak'].iloc[row_order].tolist(),
                    genes=links['gene'].iloc[row_order].tolist())

    # ── 5. Cell-type color palette ────────────────────────────────────────
    if pal_group is None:
        tab20 = plt.get_cmap('tab20')
        pal_group = {ct: tab20(i % 20) for i, ct in enumerate(ct_seq)}

    ct_to_idx = {ct: i for i, ct in enumerate(ct_seq)}
    annot_row = np.array([ct_to_idx.get(group_celltypes[i], 0) for i in col_order])
    ct_cmap   = ListedColormap([pal_group[ct] for ct in ct_seq])
    ct_norm   = BoundaryNorm(np.arange(-0.5, len(ct_seq)), len(ct_seq))

    # ── 6. Figure layout ──────────────────────────────────────────────────
    n_col     = len(col_order)
    hm_width  = n_col * col_width
    cb_width  = 0.10
    gap       = 0.15
    fig_width = 2 * hm_width + 2 * cb_width + gap + 0.5

    annot_frac = annot_height / height
    fig = plt.figure(figsize=(fig_width, height))
    gs  = gridspec.GridSpec(
        2, 5,
        height_ratios=[annot_frac, 1 - annot_frac],
        width_ratios=[hm_width, cb_width, gap, hm_width, cb_width],
        hspace=0.02, wspace=0,
        left=0.08, right=0.97, top=0.92, bottom=0.12,
    )
    ax_ann_atac = fig.add_subplot(gs[0, 0])
    ax_ann_rna  = fig.add_subplot(gs[0, 3])
    ax_atac     = fig.add_subplot(gs[1, 0])
    ax_cb_atac  = fig.add_subplot(gs[1, 1])
    ax_rna      = fig.add_subplot(gs[1, 3])
    ax_cb_rna   = fig.add_subplot(gs[1, 4])

    # ── 7. Annotation bars ────────────────────────────────────────────────
    for ax_ann, title in [(ax_ann_atac, f'ATAC  ({n_links} links)'),
                          (ax_ann_rna,  f'RNA   ({n_links} links)')]:
        ax_ann.imshow(annot_row[np.newaxis, :], aspect='auto',
                      cmap=ct_cmap, norm=ct_norm, interpolation='none')
        ax_ann.set_xticks([])
        ax_ann.set_yticks([])
        ax_ann.set_title(title, fontsize=7, pad=3)
        for bnd in ct_boundaries[:-1]:
            ax_ann.axvline(bnd - 0.5, color='white', linewidth=0.6, alpha=0.8)
        for spine in ax_ann.spines.values():
            spine.set_visible(False)

    # ── 8. Heatmaps ───────────────────────────────────────────────────────
    for ax, mat, limits, cmap in [
        (ax_atac, M_atac, limits_atac, cmap_atac),
        (ax_rna,  M_rna,  limits_rna,  cmap_rna),
    ]:
        ax.imshow(mat, aspect='auto', cmap=cmap,
                  vmin=limits[0], vmax=limits[1], interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        for pos in split_pos[:-1]:
            ax.axhline(pos - 0.5, color='white', linewidth=0.5, alpha=0.8)
        for bnd in ct_boundaries[:-1]:
            ax.axvline(bnd - 0.5, color='white', linewidth=0.6, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ── 9. Colorbars ──────────────────────────────────────────────────────
    for ax_cb, limits, cmap, label in [
        (ax_cb_atac, limits_atac, cmap_atac, 'ATAC\nZ'),
        (ax_cb_rna,  limits_rna,  cmap_rna,  'RNA\nZ'),
    ]:
        sm = ScalarMappable(cmap=cmap, norm=Normalize(*limits))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=ax_cb, orientation='vertical',
                          fraction=1.0, shrink=0.4, aspect=8)
        cb.set_label(label, fontsize=6, labelpad=2)
        cb.set_ticks([limits[0], 0, limits[1]])
        ax_cb.tick_params(labelsize=5, length=2, pad=1)
        for spine in ax_cb.spines.values():
            spine.set_visible(False)

    # ── 10. Cell-type legend ──────────────────────────────────────────────
    handles = [Patch(facecolor=pal_group[ct], label=ct) for ct in ct_seq]
    fig.legend(handles=handles, loc='lower center',
               ncol=min(len(ct_seq), 6), fontsize=6,
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    if save is not None:
        fig.savefig(save, bbox_inches='tight', dpi=300)

    return fig

    return fig
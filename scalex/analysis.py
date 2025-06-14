import os
import scanpy as sc
import pandas as pd
import numpy as np
from gseapy import barplot, dotplot
import gseapy as gp
import matplotlib.pyplot as plt
from collections import Counter
from anndata import concat

from scalex.data import aggregate_data

macrophage_markers = {
    'B': ['CD79A',  'IGHM'], # 'CD37',
    'NK': ['GNLY', 'NKG7', 'PRF1'],
    'NKT': ['DCN', 'MGP','COL1A1'],
    'T': ['IL32', 'CCL5', 'CD3D', 'CD3E', 'CD3G', 'CD4', 'CD8A', 'CD8B',  'CD7', 'TRAC', 'CD3D', 'TRBC2'],
    'Treg': ['FOXP3', 'CD25'], #'UBC', 'DNAJB1'],
    # 'naive T': ['TPT1'],
    'mast': ['TPSB2', 'CPA3', 'MS4A2', 'KIT', 'GATA2', 'FOS2'],
    'pDC': ['CLEC4C', 'TCL1A'], #['IRF7', 'IRF8', 'PLD4', 'MPEG1'],
    'epithelial': ['KRT8'],
    'cancer cells': ['EPCAM'],
    'neutrophils': ['FCGR3B', 'CSF3R', 'CXCR2', 'IFITM2', 'BASP1', 'GOS2'],
    'DC': ['CLEC9A', 'XCR1', 'CD1C', 'CD1A', 'LILRA4'],
    'cDC1': ['CLEC9A', 'IRF8', 'SNX3', 'XCR1'],
    'cDC2': ['FCER1A', 'CD1C', 'CD1E', 'CLEC10A'],
    'migratoryDC': ['BIRC3', 'CCR7', 'LAMP3'],
    'follicular DC': ['FDCSP'],
    'CD207+ DC': ['CD1A', 'CD207'], # 'FCAR1A'],
    'Monocyte': ['FCGR3A', 'CD14', 'CD16', 'VCAN', 'SELL', 'CDKN1C', 'MTSS1'],
    'Macrophage': ['CSF1R', 'C1QA', 'APOE', 'TREM2', 'MARCO', 'MCR1', 'CD68', 'CD163', 'CD206', 'CCL2', 'CCL3', 'CCL4'],
    'Foamy': ['SPP1', 'GPNMB', 'LPL', 'MGLL', 'FN1', 'APOE', 'CAPG', 'CTSD', 'LGALS3', 'LGALS1'], # IL4I1# APOE covers SPP1, LPL is a subset of SPP1, while 'LGMN' is exclusive to SPP1+ macrophages
    'Resident': ['SEPP1', 'SELENOP', 'FOLR2', 'F13A1', 'LYVE1', 'C1QA', 'C1QB', 'C1QC'], # 'RNASE1',
    'Proliferating': ['MKI67', 'TOP2A', 'TUBB', 'SMC2'], #, 'CDK1', 'CCNB1', 'CCNB2', 'CCNA2', 'CCNE1', 'CDK2', 'CDK4', 'CDK6'],
    'MonoMac': ['FCN1', 'S100A9', 'S100A8', 'LYZ', 'S100A4'], # NLRP3, 'PLAC8', 'MSRB1'
    'Inflammatory': ['NFKBIA', 'IL1B', 'CXCL2', 'CXCL8', 'IER3', 'SOD2', ], 
    # 'Inflammatory_iMacs': ['SOD2', 'CXCL9', 'ACSL1', 'SLAMF7', 'CD44', 'NAMPT', 'CXCL10', 'GBP1', 'GBP2'],
    # 'Macro FABP4+': ['FABP4'],
}

def enrich_analysis(gene_names, organism='hsapiens', gene_sets='GO_Biological_Process_2023', cutoff=0.05, **kwargs): # gene_sets="GO_Biological_Process_2021"
    """
    Perform KEGG pathway analysis and plot the results as a clustermap.

    Parameters:
    - gene_names: A dictionary with group labels as keys and lists of gene names as values.
    - gene_sets: The gene set database to use for enrichment analysis (default is 'KEGG_2021_Human'). 'GO_Biological_Process_2021', could find in gp.get_library_name()
    - organism: Organism for KEGG analysis (default is 'hsapiens
    - top_terms: Number of top terms to consider for the clustermap.
    """
    import gseapy as gp
    from gseapy import Msigdb
    msig = Msigdb()
    if isinstance(gene_names, pd.DataFrame):
         gene_names = gene_names.to_dict(orient='list')
    if gene_sets in msig.list_category(): 
        # ['c1.all', 'c2.all', 'c2.cgp', 'c2.cp.biocarta', 'c2.cp.kegg_legacy', 'c2.cp.kegg_medicus', 'c2.cp.pid', 'c2.cp.reactome', 'c2.cp', 'c2.cp.wikipathways', 'c3.all', 'c3.mir.mir_legacy', 'c3.mir.mirdb', 'c3.mir', 'c3.tft.gtrd', 'c3.tft.tft_legacy', 'c3.tft', 
        # 'c4.3ca', 'c4.all', 'c4.cgn', 'c4.cm', 'c5.all', 'c5.go.bp', 'c5.go.cc', 'c5.go.mf', 'c5.go', 'c5.hpo', 'c6.all', 'c7.all', 'c7.immunesigdb', 'c7.vax', 'c8.all', 'h.all', 'msigdb']
        gene_sets = msig.get_gmt(category = gene_sets, dbver='2024.1.Hs')
         
    results = pd.DataFrame()
    for group, genes in gene_names.items():
        # print(group, genes)
        genes = list(genes)
        enr = gp.enrichr(genes, gene_sets=gene_sets, cutoff=cutoff).results
        enr['cell_type'] = group  # Add the group label to the results
        results = pd.concat([results, enr])

    results_filtered = results[results['Adjusted P-value'] < cutoff]
    # results_pivot = results_filtered.pivot_table(index='Term', columns='cell_type', values='Adjusted P-value', aggfunc='min')
    # results_pivot = results_pivot.sort_values(by=results_pivot.columns.tolist(), ascending=True)

    # return results_pivot, results_filtered
    return results_filtered


def check_is_numeric(df, col):
    col_values = df[col].astype(str)
    is_numeric = pd.to_numeric(col_values, errors='coerce').notna().all()
    return is_numeric

def enrich_and_plot(gene_names, organism='hsapiens', gene_sets='GO_Biological_Process_2023', cutoff=0.05, add='', out_dir=None, **kwargs):
    go_results = enrich_analysis(gene_names, organism=organism, gene_sets=gene_sets, cutoff=cutoff, **kwargs)
    if add:
        go_results['cell_type'] = add + go_results['cell_type'].astype(str)
    if check_is_numeric(go_results, 'cell_type'):
        go_results['cell_type'] = 'cluster_'+go_results['cell_type'].astype(str)

    n = go_results['cell_type'].nunique()
    ax = dotplot(go_results,
            column="Adjusted P-value",
            x='cell_type', # set x axis, so you could do a multi-sample/library comparsion
            # size=10,
            top_term=10,
            figsize=(0.7*n, 2*n),
            title = f"GO_BP",  
            xticklabels_rot=45, # rotate xtick labels
            show_ring=False, # set to False to revmove outer ring
            marker='o',
            cutoff=cutoff,
            cmap='viridis'
            )
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        go_results = go_results.sort_values('Adjusted P-value', ascending=True).groupby('cell_type').head(20)
        go_results[['Gene_set','Term','Overlap', 'Adjusted P-value', 'Genes', 'cell_type']].to_csv(os.path.join(out_dir, f'go_results.csv'))
        plt.savefig(os.path.join(out_dir, f'go_results.pdf'))
    plt.show()

    return go_results



def get_markers(
        adata, 
        groupby='cell_type',
        pval_cutoff=0.01, 
        logfc_cutoff=1.5,  # ~1.5 in linear scale
        min_cells=10,
        top_n=300,
        processed=False,
        filter_pseudo=True,
        min_cell_per_batch=100,
    ):
    """
    Get markers filtered by both p-value and log fold change
    
    Parameters:
        logfc_cutoff: 0.58 ≈ 1.5 fold change (log2(1.5))
                     1.0 ≈ 2 fold change (log2(2))
        min_cell_per_batch: int, optional (default: 100)
            Minimum number of cells required per batch
    """
    from scalex.pp.annotation import format_rna

    adata = adata.copy()
    if filter_pseudo:
        adata = format_rna(adata)
    
    markers_dict = {}
    adata.obs[groupby] = adata.obs[groupby].astype('category')
    clusters = adata.obs[groupby].cat.categories

    if 'rank_genes_groups' not in adata.uns:
        if not processed:
            sc.pp.normalize_total(adata, target_sum=10000)
            sc.pp.log1p(adata)
        sc.tl.rank_genes_groups(adata, groupby=groupby, method='t-test')
    
    for cluster in clusters:
        # Get all results for this cluster
        df = sc.get.rank_genes_groups_df(adata, group=cluster)
        
        # Apply filters
        filtered = df[
            (df['pvals_adj'] < pval_cutoff) & 
            (df['logfoldchanges'] > logfc_cutoff)
        ].copy()
        
        markers_dict[cluster] = filtered.sort_values('scores', ascending=False).head(top_n)['names'].values
    
    return markers_dict


def flatten_dict(markers):
    flatten_markers = np.unique([item for sublist in markers.values() for item in sublist])
    return flatten_markers


def filter_marker_dict(markers, var_names):
    marker_dict = {}
    for cluster, genes in markers.items():
        marker_dict[cluster] = [i for i in genes if i in var_names]
    return marker_dict

def rename_marker_dict(markers, rename_dict):
    """
    Rename dictionary keys and merge values if multiple keys map to the same new key.
    
    Parameters:
    -----------
    markers : dict
        Dictionary mapping original keys to lists of values
    rename_dict : dict
        Dictionary mapping original keys to new keys
        
    Returns:
    --------
    dict
        Dictionary with renamed keys and merged values
    """
    marker_dict = {}
    for cluster, genes in markers.items():
        new_key = rename_dict[cluster]
        if new_key in marker_dict:
            # If key exists, extend the list with new values
            marker_dict[new_key].extend(genes)
        else:
            # If key doesn't exist, create new list
            marker_dict[new_key] = genes.copy()
    
    # Remove duplicates while preserving order
    for key in marker_dict:
        marker_dict[key] = list(dict.fromkeys(marker_dict[key]))
        
    return marker_dict


def cluster_program(adata_avg, n_clusters=None):
    if n_clusters is None:
        n_clusters = adata_avg.shape[0] #+ 2

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    adata_avg.var['cluster'] = np.array(kmeans.fit_predict(adata_avg.X.T)).astype(str)
    # print(adata_avg_.var)

    gene_cluster_dict = adata_avg.var.groupby('cluster').groups
    gene_cluster_dict = {k: v.tolist() for k, v in gene_cluster_dict.items()}

    return gene_cluster_dict


def find_gene_program(adata, groupby='cell_type', processed=False, n_clusters=None, top_n=300, filter_pseudo=True, **kwargs):
    """
    Find gene program for each cell type
    """
    adata = adata.copy()
    
    adata_avg = aggregate_data(adata, groupby=groupby, processed=processed, scale=True)

    markers = get_markers(adata, groupby=groupby, processed=processed, top_n=top_n, filter_pseudo=filter_pseudo, **kwargs)
    for cluster, genes in markers.items():
        print(cluster, len(genes))

    marker_list = flatten_dict(markers)

    # sc.pp.scale(adata_avg, zero_center=True)
    adata_avg_ = adata_avg[:, marker_list].copy()

    gene_cluster_dict = cluster_program(adata_avg_, n_clusters=n_clusters)

    return gene_cluster_dict, adata_avg


def find_peak_program(adata, groupby='cell_type', processed=False, n_clusters=None, top_n=-1, pval_cutoff=0.05, logfc_cutoff=1., filter_pseudo=False, **kwargs):
    """
    Find peak program for each cell type
    """
    return find_gene_program(adata, groupby=groupby, processed=processed, top_n=top_n, filter_pseudo=filter_pseudo, pval_cutoff=pval_cutoff, logfc_cutoff=logfc_cutoff, **kwargs)


def _process_group(args):
    """Helper function for multiprocessing"""
    adata_, groupby, set_type, top_n, filter_pseudo, kwargs = args
    if set_type == 'gene':
        filter_pseudo = True
    elif set_type == 'peak':
        filter_pseudo = False
        if not 'pval_cutoff' in kwargs:
            kwargs['pval_cutoff'] = 0.05
        if not 'logfc_cutoff' in kwargs:
            kwargs['logfc_cutoff'] = 1.
        
    # Filter groups with less than min_samples
    group_counts = adata_.obs[groupby].value_counts()
    valid_groups = group_counts[group_counts >= 2].index
    if len(valid_groups) < 2:
        print(f"Skipping as it has less than 2 groups with 2 or more samples")
        return None
        
    adata_ = adata_[adata_.obs[groupby].isin(valid_groups)].copy()
    markers = get_markers(adata_, groupby=groupby, top_n=top_n, filter_pseudo=filter_pseudo, **kwargs)
    # print(len(flatten_dict(markers)))
    return flatten_dict(markers)


def find_consensus_program(adata, groupby='cell_type', across=None, set_type='gene', processed=False, top_n=-1, occurance=None, min_samples=2, n_jobs=None, **kwargs):
    """
    Find consensus program for each cell type
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    groupby : str, optional (default: 'cell_type')
        Column name in adata.obs to group by
    across : str, optional (default: None)
        Column name in adata.obs to split data across
    set_type : str, optional (default: 'gene')
        Type of features to analyze ('gene' or 'peak')
    processed : bool, optional (default: False)
        Whether the data is already processed
    top_n : int, optional (default: -1)
        Number of top markers to select per group
    occurance : int, optional (default: None)
        Minimum number of occurrences across groups
    min_samples : int, optional (default: 2)
        Minimum number of samples required per group
    n_jobs : int, optional (default: None)
        Number of jobs to run in parallel. If None, uses all available cores.
    """
    adata.obs[groupby] = adata.obs[groupby].astype('category')
    n_clusters = len(adata.obs[groupby].cat.categories)
    occurance = occurance or max(2, len(np.unique(adata.obs[across])) // 2)
    filter_pseudo = True if set_type == 'gene' else False

    if across is not None:
        # Prepare arguments for parallel processing
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

        # Process groups in parallel or sequentially based on n_jobs
        if n_jobs == 1:
            results = [_process_group(args) for args in args_list]
        else:
            from multiprocessing import Pool, cpu_count
            if n_jobs is None:
                n_jobs = min(cpu_count(), 32)
            with Pool(n_jobs) as pool:
                results = pool.map(_process_group, args_list)
            
        # Filter out None results and combine markers
        markers_list = [r for r in results if r is not None]
        if not markers_list:
            raise ValueError("No valid groups found with sufficient samples")
            
        markers_list = np.concatenate(markers_list)
        gene_counts = Counter(markers_list)
        markers_list = np.array([gene for gene, count in gene_counts.items() if count >= occurance])
        print('There are {} genes with at least {} occurrences'.format(len(markers_list), occurance))

        # adata_avg_list = []
        # # gene_cluster_dict = {}
        # for c in np.unique(adata.obs[across]):
        #     adata_ = adata[adata.obs[across] == c].copy()
        #     adata_avg = aggregate_data(adata_, groupby=groupby, processed=processed, scale=True)
        #     adata_avg_list.append(adata_avg)
        #     adata_avg.obs[across] = c

        # adata_avg = concat(adata_avg_list)

    # adata_avg = aggregate_data(adata, groupby=groupby, processed=processed, scale=True)
    adata_avg.obs[groupby+'_'+across] = adata_avg.obs[groupby].astype(str) + '_' + adata_avg.obs[across].astype(str)
    # groupby = groupby+'_'+across


    # adata_avg = aggregate_data(adata, groupby=groupby, processed=processed, scale=True)
    adata_avg_ = adata_avg[:, markers_list].copy()

    gene_cluster_dict = cluster_program(adata_avg_, n_clusters=n_clusters)

    return gene_cluster_dict, adata_avg


def annotate(
    adata, 
    cell_type='leiden',
    color = ['cell_type', 'leiden', 'tissue', 'donor'],
    cell_type_markers='macrophage', #None, 
    show_markers=False,
    gene_sets='GO_Biological_Process_2023',
    additional={},
    go=True,
    out_dir = None, 
    cutoff = 0.05,
    processed=False,
    top_n=300,
):
    color = [i for i in color if i in adata.obs.columns]
    color = color + [cell_type] if cell_type not in color else color
    sc.pl.umap(adata, color=color, legend_loc='on data', legend_fontsize=10)

    var_names = adata.raw.var_names if adata.raw is not None else adata.var_names
    if cell_type_markers is not None:
        if isinstance(cell_type_markers, str):
            if cell_type_markers == 'macrophage':
                cell_type_markers = macrophage_markers
        cell_type_markers_ = {k: [i for i in v if i in var_names] for k,v in cell_type_markers.items() }
        sc.pl.dotplot(adata, cell_type_markers_, groupby=cell_type, standard_scale='var', cmap='coolwarm')
        sc.pl.heatmap(adata, cell_type_markers_, groupby=cell_type,  show_gene_labels=True, vmax=6)

    marker_genes = get_markers(adata, groupby=cell_type, processed=processed, top_n=top_n)
    # print(marker_genes)
    return enrich_and_plot(marker_genes, gene_sets=gene_sets, cutoff=cutoff, out_dir=out_dir)


def find_go_term_gene(df, term):
    """
    df: df = pd.read_csv(go_results, index_col=0)
    term: either Term full name or Go number: GO:xxxx
    """
    if term.startswith('GO'):
        df['GO'] = df['Term'].str.split('(').str[1].str.replace(')', '')
        select = df[df['GO'] == term].copy()
    else:
        select = df[df['Term'] == term].copy()
    gene_set = set(gene for sublist in select['Genes'].str.split(';') for gene in sublist)
    # gene_set = set(select['Genes'].str.split(';'))
    # print(select['Term'].head(1).values[0])
    # print('\n'.join(gene_set))
    return gene_set

def format_dict_of_list(d, out='table'):
    if out == 'matrix':
        data = []
        for k, lt in d.items():
            for v in lt:
                data.append({'Gene': v, 'Pathway': k})

        # Step 2: Create a DataFrame from the list
        df = pd.DataFrame(data)

        # Step 3: Use crosstab to pivot the DataFrame
        df = pd.crosstab(df['Gene'], df['Pathway'])
    elif out == 'table':
        df = pd.DataFrame.from_dict(d, orient='index').transpose()

    return df


def parse_go_results(df, cell_type='cell_type', top=20, out='table', tag='', dataset=''):
    """
    Return:
        a term gene dataframe: each column is a term
        a term cluster dataframe: each column is a term
    """
    term_genes = {}
    term_clusters = {}
    for c in np.unique(df[cell_type]):
        terms = df[df[cell_type]==c]['Term'].values
        for term in terms[:top]:
            if term not in term_clusters:
                term_clusters[term] = []
            
            term_clusters[term].append(c)

            if term not in term_genes:
                term_genes[term] = find_go_term_gene(df, term)

    tag = tag + ':' if tag else ''

    if out == 'dict':
        return term_genes, term_clusters
    else:
        term_genes = format_dict_of_list(term_genes, out=out)
        index = [(k, dataset, tag+';'.join(v)) for k, v in term_clusters.items()]
        term_genes.columns = pd.MultiIndex.from_tuples(index, names=['Pathway', 'Dataset', 'Cluster'])
        return term_genes


def merge_all_go_results(path, datasets=None, top=20, out_dir=None, add_ref=False, union=True, reference='GO_Biological_Process_2023', organism='human'):
    """
    The go results should organized by path/datasets/go_results.csv
    Args: 
        path is the input to store all the go results
        datasets are selected to merge
    """
    df_list = []
    if datasets is None:
        datasets = [i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]
    for dataset in datasets:
        path2 = os.path.join(path, dataset)
        for filename in os.listdir(path2):
            if 'go_results' in filename:
                name = filename.replace('.csv', '')
                path3 = os.path.join(path2, filename)
                df = pd.read_csv(path3, index_col=0)
                term_genes = parse_go_results(df, dataset=dataset, tag=name, top=top)
                df_list.append(term_genes)
    concat_df = pd.concat(df_list, axis=1)

    if add_ref and not union: 
        go_ref = gp.get_library(name=reference, organism=organism)
        go_ref = format_dict_of_list(go_ref)
        pathways = [i for i in concat_df.columns.get_level_values('Pathway').unique() if i in go_ref.columns]
        go_ref = go_ref.loc[:, pathways]
        index_tuples = [ (i, 'GO_Biological_Process_2023', 'reference') for i in go_ref.columns ] 
        go_ref.columns = pd.MultiIndex.from_tuples(index_tuples, names=['Pathway', 'Dataset', 'Cluster'])
        concat_df = pd.concat([concat_df, go_ref], axis=1)

    concat_df = concat_df.sort_index(axis=1, level='Pathway')

    if union:
        concat_df = concat_df.groupby(level=["Pathway"], axis=1)
        concat_dict = {name:  [i for i in set(group.values.flatten()) if pd.notnull(i)] for name, group in concat_df}
        concat_df = pd.DataFrame.from_dict(concat_dict, orient='index').transpose()

    if out_dir is not None:
        dirname = os.path.dirname(out_dir)
        os.makedirs(dirname, exist_ok=True)

        if not union:
            if not out_dir.endswith('xlsx'):
                out_dir = out_dir + '.xlsx'
            with pd.ExcelWriter(out_dir, engine='openpyxl') as writer:
                concat_df.to_excel(writer, sheet_name='Sheet1')
        else:
            concat_df.to_csv(out_dir, index=False)
    return concat_df
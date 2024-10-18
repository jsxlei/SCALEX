import os
import scanpy as sc
import pandas as pd
import numpy as np
from gseapy import barplot, dotplot
import matplotlib.pyplot as plt

macrophage_markers = {
    'B': ['CD79A', 'CD37', 'IGHM'],
    'NK': ['GNLY', 'NKG7', 'PRF1'],
    'NKT': ['DCN', 'MGP','COL1A1'],
    'T': ['CD3D', 'CD3E', 'CD3G', 'CD4', 'CD8A', 'CD8B'],
    'Treg': ['FOXP3', 'CD25'], #'UBC', 'DNAJB1'],
    'naive T': ['TPT1'],
    'mast': ['TPSB2', 'CPA3', 'MS4A2', 'KIT', 'GATA2', 'FOS2'],
    'pDC': ['IRF7', 'CLEC4C', 'TCL1A'], #['IRF8', 'PLD4', 'MPEG1'],
    'epithelial': ['KRT8'],
    'cancer cells': ['EPCAM'],
    # 'neutrophils': ['SOD2',  'GOS2'],
    'cDC1': ['CLEC9A'],
    'cDC2': ['FCER1A', 'CD1C', 'CD1E', 'CLEC10A'],
    'migratoryDC': ['BIRC3', 'CCR7', 'LAMP3'],
    'follicular DC': ['FDCSP'],
    'CD207+ DC': ['CD1A', 'CD207'], # 'FCAR1A'],
    'Monocyte': ['FCGR3A', 'VCAN'],
    'Macrophage': ['C1QA', 'APOE', 'TREM2', 'MARCO', 'CD68'],
    'Macro SPP1+': ['SPP1', 'LPL', 'MGLL', 'FN1'], # IL4I1
    'Macro SELENOP': ['SEPP1', 'SELENOP', 'FOLR2'], # 'RNASE1',
    'Macro FCN1+': ['FCN1', 'S100A9', 'S100A8'], # NLRP3
    'Macro IL32+': ['IL32', 'CCL5', 'CD7', 'TRAC', 'CD3D', 'TRBC2', 'IGHG1', 'IGKC'], 
    'Macro APOE+': ['C1QC', 'APOE', 'GPNMB', 'LGMN'], # C1QC > APOE
    'Macro NFKBIA+': ['NFKBIA', 'CXCL8', 'IER3', 'SOD2', 'IL1B'], 
    'Macro FABP4+': ['FABP4'],
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
    if isinstance(gene_names, pd.DataFrame):
         gene_names = gene_names.to_dict(orient='list')
    results = pd.DataFrame()
    for group, genes in gene_names.items():
        enr = gp.enrichr(genes, gene_sets=gene_sets, cutoff=cutoff).results
        enr['cell_type'] = group  # Add the group label to the results
        results = pd.concat([results, enr])

    results_filtered = results[results['Adjusted P-value'] < cutoff]
    # results_pivot = results_filtered.pivot_table(index='Term', columns='cell_type', values='Adjusted P-value', aggfunc='min')
    # results_pivot = results_pivot.sort_values(by=results_pivot.columns.tolist(), ascending=True)

    # return results_pivot, results_filtered
    return results_filtered


def annotate(
    adata, 
    cell_type='leiden',
    color = ['cell_type', 'leiden', 'tissue', 'donor'],
    cell_type_markers='macrophage', #None, 
    show_markers=False,
    n_tops = [100],
    additional={},
    go=True,
    out_dir = None, #'../../results/go_and_pathway/NSCLC_macrophage/'
):
    
    color = [i for i in color if i in adata.obs.columns]
    sc.pl.umap(adata, color=color, legend_loc='on data', legend_fontsize=10)

    if cell_type_markers is not None:
        if isinstance(cell_type_markers, str):
            if cell_type_markers == 'macrophage':
                cell_type_markers = macrophage_markers
        cell_type_markers_ = {k: [i for i in v if i in adata.raw.var_names] for k,v in cell_type_markers.items() }
        sc.pl.dotplot(adata, cell_type_markers_, groupby=cell_type, standard_scale='var', cmap='coolwarm')
    
    sc.tl.rank_genes_groups(adata, groupby=cell_type, key_added=cell_type, dendrogram=False)
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, cmap='coolwarm', key=cell_type, standard_scale='var', figsize=(22, 5), dendrogram=False)
    marker = pd.DataFrame(adata.uns[cell_type]['names'])
    marker_dict = marker.head(5).to_dict(orient='list')
    plt.show()

    if show_markers:
        for k, v in marker_dict.items():
            print(k)
            sc.pl.umap(adata, color=v, ncols=5)
        
    for n_top in n_tops:
        print('-'*20+'\n', n_top, '\n'+'-'*20)

        if go:
            go_results = enrich_analysis(marker.head(n_top))
            go_results['cell_type'] = 'leiden_' + go_results['cell_type']
            n = go_results['cell_type'].nunique()
            ax = dotplot(go_results,
                     column="Adjusted P-value",
                      x='cell_type', # set x axis, so you could do a multi-sample/library comparsion
                      # size=10,
                      top_term=10,
                      figsize=(0.7*n, 2*n),
                      title = "GO_BP",
                      xticklabels_rot=45, # rotate xtick labels
                      show_ring=False, # set to False to revmove outer ring
                      marker='o',
                     cutoff=0.05,
                     cmap='viridis'
                     )
            if out_dir is not None:
                os.makedirs(out_dir, exist_ok=True)
                go_results[['Gene_set','Term','Overlap', 'Adjusted P-value', 'Genes', 'cell_type']].to_csv(out_dir + f'/go_results_{n_top}.csv')
            plt.show()
        
        for pathway_name, pathways in additional.items():
            try:
                pathway_results = enrich_analysis(marker.head(n_top), gene_sets=pathways)
            except:
                continue
            ax = dotplot(pathway_results,
                     column="Adjusted P-value",
                      x='cell_type', # set x axis, so you could do a multi-sample/library comparsion
                      # size=10,
                      top_term=10,
                      figsize=(8,10),
                      title = pathway_name,
                      xticklabels_rot=45, # rotate xtick labels
                      show_ring=False, # set to False to revmove outer ring
                      marker='o',
                     cutoff=0.05,
                     cmap='viridis'
                     )
        
            plt.show()
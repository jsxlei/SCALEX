import os
import scanpy as sc
import pandas as pd
import numpy as np
from gseapy import barplot, dotplot
import gseapy as gp
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
    'neutrophils': ['FCGR3B', 'CSF3R', 'CXCR2', 'SOD2',  'GOS2'],
    'cDC1': ['CLEC9A'],
    'cDC2': ['FCER1A', 'CD1C', 'CD1E', 'CLEC10A'],
    'migratoryDC': ['BIRC3', 'CCR7', 'LAMP3'],
    'follicular DC': ['FDCSP'],
    'DC': ['CLEC9A', 'XCR1', 'CD1C', 'CD1A', 'LILRA4'],
    'CD207+ DC': ['CD1A', 'CD207'], # 'FCAR1A'],
    'Monocyte': ['FCGR3A', 'VCAN', 'SELL', 'CDKN1C', 'MTSS1'],
    'Macrophage': ['C1QA', 'APOE', 'TREM2', 'MARCO', 'MCR1', 'CD68', 'CD163', 'CD206', 'CCL2', 'CCL3', 'CCL4'],
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
    gene_sets='GO_Biological_Process_2021',
    n_tops = [100],
    additional={},
    go=True,
    out_dir = None, #'../../results/go_and_pathway/NSCLC_macrophage/'
):
    
    color = [i for i in color if i in adata.obs.columns]
    sc.pl.umap(adata, color=color, legend_loc='on data', legend_fontsize=10)

    var_names = adata.raw.var_names if adata.raw is not None else adata.var_names
    if cell_type_markers is not None:
        if isinstance(cell_type_markers, str):
            if cell_type_markers == 'macrophage':
                cell_type_markers = macrophage_markers
        cell_type_markers_ = {k: [i for i in v if i in var_names] for k,v in cell_type_markers.items() }
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
            for option in ['pos', 'neg']:
                if option == 'pos':
                    go_results = enrich_analysis(marker.head(n_top), gene_sets=gene_sets)
                else:
                    go_results = enrich_analysis(marker.tail(n_top), gene_sets=gene_sets)
            
                go_results['cell_type'] = 'leiden_' + go_results['cell_type']
                n = go_results['cell_type'].nunique()
                ax = dotplot(go_results,
                        column="Adjusted P-value",
                        x='cell_type', # set x axis, so you could do a multi-sample/library comparsion
                        # size=10,
                        top_term=10,
                        figsize=(0.7*n, 2*n),
                        title = f"{option}_GO_BP_{n_top}",  
                        xticklabels_rot=45, # rotate xtick labels
                        show_ring=False, # set to False to revmove outer ring
                        marker='o',
                        cutoff=0.05,
                        cmap='viridis'
                        )
                if out_dir is not None:
                    os.makedirs(out_dir, exist_ok=True)
                    go_results = go_results.sort_values('Adjusted P-value', ascending=False).groupby('cell_type').head(10)
                    go_results[['Gene_set','Term','Overlap', 'Adjusted P-value', 'Genes', 'cell_type']].to_csv(out_dir + f'/{option}_go_results_{n_top}.csv')
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
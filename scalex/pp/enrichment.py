import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import gseapy as gp
import matplotlib.pyplot as plt
from sklearn.metrics import auc

from scalex.pl._dotplot import dotplot


def check_is_numeric(df, col):
    col_values = df[col].astype(str)
    return pd.to_numeric(col_values, errors='coerce').notna().all()


def parse_go_file(filepath):
    go_dict = {}
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            key = parts[0].strip()
            genes = [g.strip() for g in parts[1:] if g.strip()]
            go_dict[key] = genes
    return go_dict


def enrich_analysis(gene_names, organism='hsapiens', gene_sets='GO_Biological_Process_2023', cutoff=0.05, **kwargs):
    """
    Perform enrichment analysis.

    Parameters
    ----------
    gene_names
        Dict with group labels as keys and lists of gene names as values.
    gene_sets
        Gene set database to use for enrichment analysis.
    organism
        Organism for analysis (default 'hsapiens').
    """
    from gseapy import Msigdb
    msig = Msigdb()
    if isinstance(gene_names, pd.DataFrame):
        gene_names = gene_names.to_dict(orient='list')
    if gene_sets in msig.list_category():
        gene_sets = msig.get_gmt(category=gene_sets, dbver='2024.1.Hs')

    results = pd.DataFrame()
    for group, genes in gene_names.items():
        genes = list(genes)
        enr = gp.enrichr(genes, gene_sets=gene_sets, cutoff=cutoff).results
        enr['cell_type'] = group
        results = pd.concat([results, enr])

    return results[results['Adjusted P-value'] < cutoff]


def enrich_and_plot(gene_names, organism='hsapiens', gene_sets='GO_Biological_Process_2023', cutoff=0.05, cmap='Reds', add='', save=None, figsize=None, vmin=None, vmax=None, **kwargs):
    go_results = enrich_analysis(gene_names, organism=organism, gene_sets=gene_sets, cutoff=cutoff)
    if vmin is not None or vmax is not None:
        go_results = go_results.copy()
        go_results['Adjusted P-value'] = go_results['Adjusted P-value'].clip(
            lower=10**(-vmax) if vmax is not None else None,
            upper=10**(-vmin) if vmin is not None else None,
        )
    plot_go_enrich(go_results, add=add, save=save, cutoff=cutoff, figsize=figsize, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)


def plot_go_enrich(go_results, add='', save=None, cutoff=0.05, top_term=10, figsize=None, vmin=None, vmax=None, **kwargs):
    if add:
        go_results['cell_type'] = add + go_results['cell_type'].astype(str)
    if check_is_numeric(go_results, 'cell_type'):
        go_results['cell_type'] = 'cluster_' + go_results['cell_type'].astype(str)

    n = go_results['cell_type'].nunique()
    if figsize is None:
        figsize = (0.7 * n, 2 * n)
    dotplot(
        go_results,
        column="Adjusted P-value",
        x='cell_type',
        top_term=top_term,
        figsize=figsize,
        title="GO_BP",
        xticklabels_rot=45,
        show_ring=False,
        marker='o',
        cutoff=cutoff,
        vmin=vmin,
        vmax=vmax,
        **kwargs
    )
    if save is not None:
        out_dir = os.path.dirname(save)
        os.makedirs(out_dir, exist_ok=True)
        go_results = go_results.sort_values('Adjusted P-value', ascending=True).groupby('cell_type').head(20)
        go_results[['Gene_set', 'Term', 'Overlap', 'Adjusted P-value', 'Genes', 'cell_type']].to_csv(
            os.path.join(out_dir, 'go_results.csv')
        )
        plt.savefig(save, bbox_inches='tight')
    plt.show()
    return go_results


def filter_go_genes(
    adata,
    go_dict,
    min_pct: float = 0.10,
    min_mean: float = 0.1,
    library: str = 'GO_Biological_Process_2023',
    organism: str = 'Human',
) -> dict:
    """Return filtered GO term gene lists based on expression in adata.

    For each GO term in go_dict, retrieves the full gene list from the
    gseapy library and keeps only genes that:
      - are present in adata.var_names
      - are expressed in >= min_pct fraction of cells
      - have mean expression >= min_mean

    Returns
    -------
    dict : {go_term: [filtered_genes]}
    """
    bp = gp.get_library(library, organism=organism)

    def _to_list(v):
        return [v] if isinstance(v, str) else list(v)

    all_go_terms = {go for terms in go_dict.values() for go in _to_list(terms)}
    var_names = set(adata.var_names)

    result = {}
    for go in all_go_terms:
        if go not in bp:
            result[go] = []
            continue

        genes = [g for g in bp[go] if g in var_names]
        if not genes:
            result[go] = []
            continue

        X_sub = adata[:, genes].X
        if sp.issparse(X_sub):
            X_sub = X_sub.toarray()

        pct_cells = (X_sub > 0).mean(axis=0)
        mean_expr = X_sub.mean(axis=0)

        result[go] = [
            g for g, pct, mean in zip(genes, pct_cells, mean_expr)
            if pct >= min_pct and mean >= min_mean
        ]

    return result


def find_go_term_gene(df, term):
    """
    Extract gene set for a given GO term.

    Parameters
    ----------
    df
        GO results DataFrame
    term
        Full term name or GO number (e.g. 'GO:0006954')
    """
    if term.startswith('GO'):
        df['GO'] = df['Term'].str.split('(').str[1].str.replace(')', '')
        select = df[df['GO'] == term].copy()
    else:
        select = df[df['Term'] == term].copy()
    return set(gene for sublist in select['Genes'].str.split(';') for gene in sublist)


def format_dict_of_list(d, out='table'):
    if out == 'matrix':
        data = [{'Gene': v, 'Pathway': k} for k, lt in d.items() for v in lt]
        df = pd.DataFrame(data)
        return pd.crosstab(df['Gene'], df['Pathway'])
    return pd.DataFrame.from_dict(d, orient='index').transpose()


def parse_go_results(df, cell_type='cell_type', top=20, out='table', tag='', dataset=''):
    """
    Parse GO results into a term-gene or term-cluster table.
    """
    term_genes = {}
    term_clusters = {}
    for c in np.unique(df[cell_type]):
        terms = df[df[cell_type] == c]['Term'].values
        for term in terms[:top]:
            if term not in term_clusters:
                term_clusters[term] = []
            term_clusters[term].append(c)
            if term not in term_genes:
                term_genes[term] = find_go_term_gene(df, term)

    tag = tag + ':' if tag else ''

    if out == 'dict':
        return term_genes, term_clusters
    term_genes = format_dict_of_list(term_genes, out=out)
    index = [(k, dataset, tag + ';'.join(v)) for k, v in term_clusters.items()]
    term_genes.columns = pd.MultiIndex.from_tuples(index, names=['Pathway', 'Dataset', 'Cluster'])
    return term_genes


def merge_all_go_results(path, datasets=None, top=20, out_dir=None, add_ref=False, union=True, reference='GO_Biological_Process_2023', organism='human'):
    """
    Merge GO results from multiple datasets.

    The go results should be organized as path/datasets/go_results.csv.
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
        index_tuples = [(i, 'GO_Biological_Process_2023', 'reference') for i in go_ref.columns]
        go_ref.columns = pd.MultiIndex.from_tuples(index_tuples, names=['Pathway', 'Dataset', 'Cluster'])
        concat_df = pd.concat([concat_df, go_ref], axis=1)

    concat_df = concat_df.sort_index(axis=1, level='Pathway')

    if union:
        concat_df = concat_df.groupby(level=["Pathway"], axis=1)
        concat_dict = {name: [i for i in set(group.values.flatten()) if pd.notnull(i)] for name, group in concat_df}
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


def enrich_module(adata, gene_sets):
    import omicverse as ov
    for k, v in gene_sets.items():
        ov.single.geneset_aucell(adata, geneset_name=k, geneset=v)
    return adata


def plot_enrich_module(adata, gene_sets):
    import scanpy as sc
    sc.pl.embedding(adata, basis='umap', color=["{}_aucell".format(i) for i in gene_sets.keys()])


def plot_radar_module(adata, columns='cell_type', cols=None, save=None):
    from scipy.stats import zscore
    from scalex.plot import plot_radar

    if cols is None:
        cols = [i for i in adata.obs.columns if i.endswith('aucell')]
    else:
        cols = [i + '_aucell' for i in cols if not i.endswith('aucell')]
    avg_score = adata.obs.groupby(columns)[cols].mean()
    avg_score = zscore(avg_score)
    scaled = (avg_score - avg_score.min(axis=0)) / (avg_score.max(axis=0) - avg_score.min(axis=0))
    plot_radar(scaled)
    return scaled


def aucell_scores(expr: pd.DataFrame, gene_sets: dict, top=0.05):
    """
    AUCell implementation in Python.

    Parameters
    ----------
    expr
        cells x genes expression matrix (DataFrame).
    gene_sets
        Dict of {program_name: [genes]}.
    top
        Fraction of top-ranked genes to consider.
    """
    n_top = int(expr.shape[1] * top)
    scores = {}
    for prog, genes in gene_sets.items():
        prog_scores = []
        valid_genes = [g for g in genes if g in expr.columns]
        if len(valid_genes) == 0:
            prog_scores = [np.nan] * expr.shape[0]
        else:
            for _, row in expr.iterrows():
                ranked_genes = row.sort_values(ascending=False).index[:n_top]
                hits = [1 if g in ranked_genes else 0 for g in valid_genes]
                x = np.linspace(0, 1, len(hits))
                y = np.cumsum(hits) / max(1, sum(hits))
                score = auc(x, y)
                prog_scores.append(score)
        scores[prog] = prog_scores
    return pd.DataFrame(scores, index=expr.index)

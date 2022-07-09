import numpy as np
import pandas as pd
import scanpy as sc
import argparse
import anndata
import bbknn
import os
import time
import matplotlib.pyplot as plt


sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=150)  # low dpi (dots per inch) yields small inline figures

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrate multi single cell datasets by BBKNN')
    parser.add_argument('--h5ad', type=str, default=None)
    parser.add_argument('--outdir', '-o', type=str, default='./')
    parser.add_argument('--min_genes', type=int, default=600)
    parser.add_argument('--min_cells', type=int, default=3)
    parser.add_argument('--num_pcs', type=int, default=20)
    parser.add_argument('--n_top_features', type=int, default=2000)

    args = parser.parse_args()
    
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    
    adata = sc.read_h5ad(args.h5ad)
    time1 = time.time()
    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sc.pp.filter_genes(adata, min_cells=args.min_cells)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata.copy()
    
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", batch_key='batch', n_top_genes=args.n_top_features, subset=True)
    
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_pcs=args.num_pcs, n_neighbors=20)
    adata_bbknn = bbknn.bbknn(adata, neighbors_within_batch=5, n_pcs=args.num_pcs, trim=0, copy=True)
    sc.tl.umap(adata_bbknn, min_dist=0.1)
    
    # UMAP
    sc.settings.figdir = outdir
    plt.rcParams['figure.figsize'] = (6, 8)
    cols = ['celltype',  'batch']
    
    color = [c for c in cols if c in adata_bbknn.obs]
    sc.pl.umap(adata_bbknn, color=color, frameon=False, save='.png', wspace=0.4, show=False, ncols=1)
    time2 = time.time()
    print("--- %s seconds ---" % (time2 - time1))

    # pickle data
    adata_bbknn.write(outdir+'/adata.h5ad', compression='gzip')  
   
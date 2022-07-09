import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import anndata
import scvi
import scanpy as sc

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=150)  # low dpi (dots per inch) yields small inline figures



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrate multi single cell datasets by scVI')
    parser.add_argument('--h5ad', type=str, default=None)
    parser.add_argument('--outdir', '-o', type=str, default='./')
    parser.add_argument('--min_genes', type=int, default=600)
    parser.add_argument('--min_cells', type=int, default=3)
    parser.add_argument('-g','--gpu', type=int, default=0)
    parser.add_argument('--n_top_features', type=int, default=2000)
    args = parser.parse_args()
    
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    
    adata = sc.read_h5ad(args.h5ad)    
    time1 = time.time()
    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    
    sc.pp.highly_variable_genes(adata,
                                flavor="seurat_v3",
                                n_top_genes=args.n_top_features,
                                layer="counts",
                                batch_key="batch",
                                subset=True)
    
    scvi.model.SCVI.setup_anndata(adata, 
                                  layer="counts", 
#                                   categorical_covariate_keys=["batch", "donor"],
#                                   continuous_covariate_keys=["percent_mito", "percent_ribo"],
                                  batch_key="batch"
                                 )
    vae = scvi.model.SCVI(adata)
#     vae = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
    vae.train(use_gpu = args.gpu)
    
    adata.obsm["X_scVI"] = vae.get_latent_representation()
    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.umap(adata, min_dist=0.1)
    time2 = time.time()
    
    # UMAP
    sc.settings.figdir = outdir
    plt.rcParams['figure.figsize'] = (6, 8)
    cols = ['celltype',  'batch']
    
    color = [c for c in cols if c in adata.obs]
    sc.pl.umap(adata, color=color, frameon=False, save='.png', wspace=0.4, show=False, ncols=1)
    
    # pickle data
    adata.write(outdir+'/adata.h5ad', compression='gzip')  
    print("--- %s seconds ---" % (time2 - time1))


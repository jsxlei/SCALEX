import pandas as pd
import numpy as np
import argparse
import os
import time
import matplotlib.pyplot as plt

import anndata
import desc
import scanpy as sc

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=150)  # low dpi (dots per inch) yields small inline figures

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrate multi single cell datasets by DESC')
    parser.add_argument('--h5ad', type=str, default=None)
    parser.add_argument('--outdir', '-o', type=str, default='./')
    parser.add_argument('--min_genes', type=int, default=600)
    parser.add_argument('--min_cells', type=int, default=3)
    parser.add_argument('-g','--gpu', type=int, default=0)
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    
    adata = sc.read_h5ad(args.h5ad)    
    time1 = time.time()
    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw=adata
#     sc.pp.scale(adata, zero_center=True, max_value=3)

    sc.pp.highly_variable_genes(adata,
                                flavor="seurat_v3",
                                n_top_genes=2000,
                                batch_key="batch",
                                subset=True)


    adata = desc.scale_bygroup(adata, groupby='batch', max_value=10)

    adata = desc.train(adata, 
                       dims=[adata.shape[1], 128, 32], 
                       tol=0.001, 
                       n_neighbors=10,
                       batch_size=256, 
                       louvain_resolution=[0.8],
                       save_dir=outdir, 
                       do_tsne=False, 
                       learning_rate=300,
                       use_GPU=True,
                       num_Cores=1,
                       do_umap=False, 
                       num_Cores_tsne=4,
                       use_ae_weights=False,
                       save_encoder_weights=False)
    
    sc.pp.neighbors(adata, use_rep="X_Embeded_z0.8")
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
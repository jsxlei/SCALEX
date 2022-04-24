#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Thu 31 May 2018 08:45:33 PM CST

# File Name: __init__.py
# Description:

"""

__author__ = "Lei Xiong"
__email__ = "jsxlei@gmail.com"

from .layer import *
from .model import *
from .loss import *
from .dataset import load_dataset
from .utils import estimate_k, binarization


import time
import torch

import numpy as np
import pandas as pd
import os
import scanpy as sc


def SCALE(
        data_list, 
        batch_categories=None, 
        join='inner', 
        batch_key='batch', 
        batch_name='batch',
        min_features=200, 
        min_cells=3, 
        n_top_features=30000, 
        batch_size=64, 
        lr=2e-4, 
        max_iteration=30000,
        seed=124, 
        gpu=0, 
        outdir='output/', 
        projection=None,
        repeat=False,
        impute=None, 
        chunk_size=20000,
        ignore_umap=False,
        verbose=False,
        n_centroids=30,
        pretrain=None,
        embed='UMAP',
    ):
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(gpu)
    else:
        device='cpu'
    
    print("\n**********************************************************************")
    print("  SCALE: Single-Cell ATAC-seq Analysis via Latent feature Extraction")
    print("**********************************************************************\n")
    
    if min_features==None: min_features=200
    if n_top_features==None: n_top_features=30000

    adata, trainloader, testloader = load_dataset(
        data_list,
        batch_categories=None, 
        join='inner', 
        batch_key='batch', 
        batch_name='batch',
        min_features=min_features,
        min_cells=min_cells,
        batch_size=batch_size, 
        log=None,
    )

    cell_num = adata.shape[0] 
    input_dim = adata.shape[1] 	
    
#     if  n_centroids is None:
#         k = estimate_k(adata.X.T)
#         print('Estimate k = {}'.format(k))
#     else:
#         k =  n_centroids
    k = n_centroids
    
    
    outdir =  outdir+'/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    print("\n======== Parameters ========")
    print('Cell number: {}\nPeak number: {}\nn_centroids: {}\nmax_iter: {}\nbatch_size: {}\ncell filter by peaks: {}\npeak filter by cells: {}'.format(
        cell_num, input_dim, k,  max_iteration, batch_size,  min_features,  min_cells))
    print("============================")

    latent = 10
    encode_dim = [1024, 128]
    decode_dim = []
    dims = [input_dim,  latent,  encode_dim,  decode_dim]
    model = SCALE(dims, n_centroids=k)
    print(model)

    if not pretrain:
        print('\n## Training Model ##')
        model.init_gmm_params(testloader)
        model.fit(trainloader,
                  lr=lr, 
                  verbose= verbose,
                  device = device,
                  max_iter= max_iteration,
                  outdir=outdir
                   )
        torch.save(model.state_dict(), os.path.join(outdir, 'model.pt')) # save model
    else:
        print('\n## Loading Model: {}\n'.format(pretrain))
        model.load_model(pretrain)
        model.to(device)
    
    ### output ###
    print('outdir: {}'.format(outdir))
    # 1. latent feature
    adata.obsm['latent'] = model.encodeBatch(testloader, device=device, out='z')

    # 2. cluster
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent')
    sc.tl.leiden(adata)
#     if  cluster_method == 'leiden':
#         sc.tl.leiden(adata)
#     elif  .cluster_method == 'kmeans':
#         kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
#         adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['latent']).astype(str)

#     if  .reference in adata.obs:
#         cluster_report(adata.obs[ .reference].cat.codes, adata.obs[ .cluster_method].astype(int))

    sc.settings.figdir = outdir
    sc.set_figure_params(dpi=80, figsize=(6,6), fontsize=10)
    if  embed == 'UMAP':
        sc.tl.umap(adata, min_dist=0.1)
        color = [c for c in ['celltype',  'leiden'] if c in adata.obs]
        sc.pl.umap(adata, color=color, save='.pdf', wspace=0.4, ncols=4)
    elif  embed == 'tSNE':
        sc.tl.tsne(adata, use_rep='latent')
        color = [c for c in ['celltype',  'leiden'] if c in adata.obs]
        sc.pl.tsne(adata, color=color, save='.pdf', wspace=0.4, ncols=4)
    
    if  impute:
        adata.obsm['impute'] = model.encodeBatch(testloader, device=device, out='x')
        adata.obsm['binary'] = binarization(adata.obsm['impute'], adata.X)
    
    adata.write(outdir+'adata.h5ad', compression='gzip')

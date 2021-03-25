#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Tue 29 Sep 2020 01:41:23 PM CST

# File Name: function.py
# Description:

"""

import torch
import numpy as np
import os
import scanpy as sc
from anndata import AnnData

from .data import load_data
from .net.vae import VAE
from .net.utils import EarlyStopping
from .metrics import batch_entropy_mixing_score, silhouette_score
from .logger import create_logger
from .plot import embedding


def SCALE(
        data_list, 
        batch_categories=None,
        profile='RNA',
        join='inner', 
        batch_key='batch', 
        batch_name='batch',
        min_features=600, 
        min_cells=3, 
        n_top_features=2000, 
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
        assess=False,
        show=True,
    ):
    """
    Single-Cell integrative Analysis via Latent feature Extraction
    
    Parameters
    ----------
    data_list
        A path list of AnnData matrices to concatenate with. Each matrix is referred to as a 'batch'.
    batch_categories
        Categories for the batch annotation. By default, use increasing numbers.
    profile
        Specify the single-cell profile, RNA or ATAC. Default: RNA.
    join
        Use intersection ('inner') or union ('outer') of variables of different batches. 
    batch_key
        Add the batch annotation to obs using this key. By default, batch_key='batch'.
    batch_name
        Use this annotation in obs as batches for training model. Default: 'batch'.
    min_features
        Filtered out cells that are detected in less than min_features. Default: 600.
    min_cells
        Filtered out genes that are detected in less than min_cells. Default: 3.
    n_top_features
        Number of highly-variable genes to keep. Default: 2000.
    batch_size
        Number of samples per batch to load. Default: 64.
    lr
        Learning rate. Default: 2e-4.
    max_iteration
        Max iterations for training. Training one batch_size samples is one iteration. Default: 30000.
    seed
        Random seed for torch and numpy. Default: 124.
    gpu
        Index of GPU to use if GPU is available. Default: 0.
    outdir
        Output directory. Default: 'output/'.
    projection
        Use for new dataset projection. Input the folder containing the pre-trained model. If None, don't do projection. Default: None. 
    repeat
        Use with projection. If False, concatenate the reference and projection datasets for downstream analysis. If True, only use projection datasets. Default: False.
    impute
        If True, calculate the imputed gene expression and store it at adata.layers['impute']. Default: False.
    chunk_size
        Number of samples from the same batch to transform. Default: 20000.
    ignore_umap
        If True, do not perform UMAP for visualization and leiden for clustering. Default: False.
    verbose
        Verbosity, True or False. Default: False.
    assess
        If True, calculate the entropy_batch_mixing score and silhouette score to evaluate integration results. Default: False.
    
    Returns
    -------
    The output folder contains:
    adata.h5ad
        The AnnData matrice after batch effects removal. The low-dimensional representation of the data is stored at adata.obsm['latent'].
    checkpoint
        model.pt contains the variables of the model and config.pt contains the parameters of the model.
    log.txt
        Records raw data information, filter conditions, model parameters etc.
    umap.pdf 
        UMAP plot for visualization.
    """
    
    np.random.seed(seed) # seed
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(gpu)
    else:
        device='cpu'
    
    outdir = outdir+'/'
    os.makedirs(outdir+'/checkpoint', exist_ok=True)
    log = create_logger('', fh=outdir+'log.txt')
    if not projection:
        adata, trainloader, testloader = load_data(
            data_list, batch_categories, 
            join=join,
            profile=profile,
            n_top_features=n_top_features,
            batch_size=batch_size, 
            chunk_size=chunk_size,
            min_features=min_features, 
            min_cells=min_cells,
            batch_name=batch_name, 
            batch_key=batch_key,
            log=log
        )
        
        early_stopping = EarlyStopping(patience=10, checkpoint_file=outdir+'/checkpoint/model.pt')
        x_dim, n_domain = adata.shape[1], len(adata.obs['batch'].cat.categories)
        
        # model config
        enc = [['fc', 1024, 1, 'relu'],['fc', 10, '', '']]  # TO DO
        dec = [['fc', x_dim, n_domain, 'sigmoid']]

        model = VAE(enc, dec, n_domain=n_domain)
        
        log.info('model\n'+model.__repr__())
        model.fit(
            trainloader, 
            lr=lr, 
            max_iteration=max_iteration, 
            device=device, 
            early_stopping=early_stopping, 
            verbose=verbose,
        )
        torch.save({'n_top_features':adata.var.index, 'enc':enc, 'dec':dec, 'n_domain':n_domain}, outdir+'/checkpoint/config.pt')     
    else:
        state = torch.load(projection+'/checkpoint/config.pt')
        n_top_features, enc, dec, n_domain = state['n_top_features'], state['enc'], state['dec'], state['n_domain']
        model = VAE(enc, dec, n_domain=n_domain)
        model.load_model(projection+'/checkpoint/model.pt')
        model.to(device)
        
        adata, trainloader, testloader = load_data(
            data_list, batch_categories,  
            join='outer', 
            profile=profile,
            chunk_size=chunk_size,
            n_top_features=n_top_features, 
            min_cells=0,
            min_features=min_features,
            batch_name=batch_name,
            batch_key=batch_key,
            log = log
        )
#         log.info('Processed dataset shape: {}'.format(adata.shape))
        
    adata.obsm['latent'] = model.encodeBatch(testloader, device=device) # save latent rep
    if impute:
        adata.layers['impute'] = model.encodeBatch(testloader, out='impute', batch_id=impute, device=device)
    log.info('Output dir: {}'.format(outdir))
        
    if projection and (not repeat):
        ref = sc.read_h5ad(projection+'/adata.h5ad')
        adata = AnnData.concatenate(
            ref, adata, 
            batch_categories=['reference', 'query'], 
            batch_key='projection', 
            index_unique=None
        )
    adata.write(outdir+'adata.h5ad', compression='gzip')  
    if not ignore_umap: #and adata.shape[0]<1e6:
        log.info('Plot umap')
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent')
        sc.tl.umap(adata, min_dist=0.1)
        sc.tl.leiden(adata)
        
        # UMAP visualization
        sc.settings.figdir = outdir
        sc.set_figure_params(dpi=80, figsize=(10,10), fontsize=20)
        cols = ['batch', 'celltype', 'leiden']
        color = [c for c in cols if c in adata.obs]
        if len(color) > 0:
            if projection and (not repeat):
                embedding(adata, groupby='projection', save='.pdf', show=show)
            else:
                sc.pl.umap(adata, color=color, save='.pdf', wspace=0.4, ncols=4, show=show)
           
        if assess:
            if len(adata.obs['batch'].cat.categories) > 1:
                entropy_score = batch_entropy_mixing_score(adata.obsm['X_umap'], adata.obs['batch'])
                log.info('batch_entropy_mixing_score: {:.3f}'.format(entropy_score))

            if 'celltype' in adata.obs:
                sil_score = silhouette_score(adata.obsm['X_umap'], adata.obs['celltype'].cat.codes)
                log.info("silhouette_score: {:.3f}".format(sil_score))

    adata.write(outdir+'adata.h5ad', compression='gzip')
    
    return adata
        
        

def label_transfer(ref, query, rep='latent', label='celltype'):
    """
    Label transfer
    
    Parameters
    -----------
    ref
        reference containing the projected representations and labels
    query
        query data to transfer label
    rep
        representations to train the classifier. Default is `latent`
    label
        label name. Defautl is `celltype` stored in ref.obs
    
    Returns
    --------
    transfered label
    """

    from sklearn.neighbors import KNeighborsClassifier
    
    X_train = ref.obsm[rep]
    y_train = ref.obs[label]
    X_test = query.obsm[rep]
    
    knn = knn = KNeighborsClassifier().fit(X_train, y_train)
    y_test = knn.predict(X_test)
    
    return y_test


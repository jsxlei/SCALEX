#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Tue 29 Sep 2020 01:41:23 PM CST

# File Name: function.py
# Description:

"""

from anndata import AnnData, concat
# from typing import Union, List


def SCALEX(
        data_list, #Union[str, AnnData, List]=None, 
        batch_categories:list=None,
        profile:str='RNA',
        batch_name:str='batch',
        min_features:int=600, 
        min_cells:int=3, 
        target_sum:int=None,
        n_top_features:int=None,
        min_cell_per_batch:int=10,
        join:str='inner', 
        batch_key:str='batch',  
        processed:bool=False,
        fraction:float=None,
        n_obs:int=None,
        use_layer:str='X',
        keep_mt:bool=False,
        backed:bool=False,
        batch_size:int=64, 
        groupby:str=None,
        subsets:list=None,
        lr:float=2e-4, 
        max_iteration:int=30000,
        seed:int=124, 
        gpu:int=0, 
        outdir:str=None, 
        projection:str=None,
        repeat:bool=False,
        name:str=None,
        impute:str=None, 
        chunk_size:int=20000,
        ignore_umap:bool=False,
        verbose:bool=False,
        assess:bool=False,
        show:bool=True,
        eval:bool=False,
        num_workers:int=4,
        cell_type:str='cell_type',
    ): # -> AnnData:
    """
    Online single-cell data integration through projecting heterogeneous datasets into a common cell-embedding space
    
    Parameters
    ----------
    data_list
        A path list of AnnData matrices to concatenate with. Each matrix is referred to as a 'batch'.
    batch_categories
        Categories for the batch annotation. By default, use increasing numbers.
    profile
        Specify the single-cell profile, RNA or ATAC. Default: RNA.
    batch_name
        Use this annotation in obs as batches for training model. Default: 'batch'.
    min_features
        Filtered out cells that are detected in less than min_features. Default: 600.
    min_cells
        Filtered out genes that are detected in less than min_cells. Default: 3.
    n_top_features
        Number of highly-variable genes to keep. Default: 2000.
    join
        Use intersection ('inner') or union ('outer') of variables of different batches. 
    batch_key
        Add the batch annotation to obs using this key. By default, batch_key='batch'.
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
    import torch
    import numpy as np
    import pandas as pd
    import os
    import scanpy as sc


    from .data import load_data
    from .net.vae import VAE
    from .net.utils import EarlyStopping
    from .metrics import batch_entropy_mixing_score, silhouette_score
    from .logger import create_logger
    from .plot import embedding

    np.random.seed(seed) # seed
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(gpu)
    else:
        device='cpu'
    
    if outdir:
        if name is not None and projection is not None:
            outdir = os.path.join(projection, 'projection', name)
            os.makedirs(outdir, exist_ok=True)
        # outdir = outdir+'/'
        os.makedirs(os.path.join(outdir, 'checkpoint'), exist_ok=True)
        log = create_logger('SCALEX', fh=os.path.join(outdir, 'log.txt'), overwrite=True)
    else:
        log = create_logger('SCALEX')

    if not projection:
        adata, trainloader, testloader = load_data(
            data_list, batch_categories, 
            join=join,
            profile=profile,
            target_sum=target_sum,
            n_top_features=n_top_features,
            min_cell_per_batch=min_cell_per_batch,
            batch_size=batch_size, 
            groupby=groupby,
            subsets=subsets,
            chunk_size=chunk_size,
            min_features=min_features, 
            min_cells=min_cells,
            fraction=fraction,
            n_obs=n_obs,
            processed=processed,
            use_layer=use_layer,
            backed=backed,
            batch_name=batch_name, 
            batch_key=batch_key,
            keep_mt=keep_mt,
            log=log,
            num_workers=num_workers,
        )
        
        early_stopping = EarlyStopping(patience=10, checkpoint_file=os.path.join(outdir, 'checkpoint/model.pt') if outdir else None)
        x_dim = adata.shape[1] if (use_layer == 'X' or use_layer in adata.layers) else adata.obsm[use_layer].shape[1]
        n_domain = len(adata.obs['batch'].cat.categories)
        
        # model config
        enc = [['fc', 1024, 1, 'relu'],['fc', 10, '', '']]  # TO DO
        dec = [['fc', x_dim, n_domain, 'sigmoid']]

        model = VAE(enc, dec, n_domain=n_domain)
        
        # log.info('model\n'+model.__repr__())
        model.fit(
            trainloader, 
            lr=lr, 
            max_iteration=max_iteration, 
            device=device, 
            early_stopping=early_stopping, 
            verbose=verbose,
        )
        if outdir:
            torch.save({'n_top_features':adata.var.index, 'enc':enc, 'dec':dec, 'n_domain':n_domain}, os.path.join(outdir, 'checkpoint/config.pt'))
    else:
        state = torch.load(os.path.join(projection, 'checkpoint/config.pt'), weights_only=False)
        n_top_features, enc, dec, n_domain = state['n_top_features'], state['enc'], state['dec'], state['n_domain']
        model = VAE(enc, dec, n_domain=n_domain)
        model.load_model(os.path.join(projection, 'checkpoint/model.pt'))
        model.to(device)
        
        adata, trainloader, testloader = load_data(
            data_list, batch_categories,  
            join='outer', 
            profile=profile,
            target_sum=target_sum,
            chunk_size=chunk_size,
            n_top_features=n_top_features, 
            min_cells=0,
            min_features=min_features,
            min_cell_per_batch=min_cell_per_batch,
            processed=processed,
            batch_name=batch_name,
            batch_key=batch_key,
            # keep_mt=keep_mt,
            log = log,
            num_workers=num_workers,
        )
#         log.info('Processed dataset shape: {}'.format(adata.shape))
        
    adata.obsm['latent'] = model.encodeBatch(testloader, device=device, eval=eval) # save latent rep
    if impute:
        adata.layers['impute'] = model.encodeBatch(testloader, out='impute', batch_id=impute, device=device, eval=eval)
    # log.info('Output dir: {}'.format(outdir))
    
    model.to('cpu')
    del model
    if projection and (not repeat):
        ref = sc.read_h5ad(os.path.join(projection, 'adata.h5ad'))
        adata = concat(
            [ref, adata],
            join='outer',
            label='projection',
            keys=['reference', 'query'],
            index_unique=None,
        )
        # adata.raw = concat([ref.raw.to_adata(), adata.raw.to_adata()], join='outer', label='projection', keys=['reference', 'query'])
        if 'leiden' in adata.obs:
            del adata.obs['leiden']
        for col in adata.obs.columns:
            if not pd.api.types.is_string_dtype(adata.obs[col]):
                adata.obs[col] = adata.obs[col].astype(str)

    # if outdir is not None:
    #     adata.write(os.path.join(outdir, 'adata.h5ad'), compression='gzip')  

    if not ignore_umap: #and adata.shape[0]<1e6:
        log.info('Plot umap')
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent')
        sc.tl.umap(adata, min_dist=0.1)
        sc.tl.leiden(adata)
        adata.obsm['X_scalex_umap'] = adata.obsm['X_umap']
        
        # UMAP visualization
        sc.set_figure_params(dpi=80, figsize=(3,3))
        cols = [cell_type, 'leiden'] 
        cols += ['batch'] if n_domain > 1 else []
        color = [c for c in cols if c in adata.obs]
        if outdir:
            sc.settings.figdir = outdir
            save = '.png'
        else:
            save = None

        if len(color) > 0:
            if projection and (not repeat):
                cell_type = cell_type if cell_type in adata.obs else 'leiden'
                save = os.path.join(outdir, 'umap.png')
                embedding(adata, color=cell_type, groupby='projection', save=save, show=show)
            else:
                sc.pl.umap(adata, color=color, save=save, wspace=0.4, ncols=4, show=show)  

    if outdir is not None:
        adata.write(os.path.join(outdir, 'adata.h5ad'), compression='gzip')

    if assess:
        if adata.shape[0] > 5e4:
            log.info('Subsample cell numbers to 50,000 for entropy_batch_mixing_score calculation.')
            sc.pp.subsample(adata, n_obs=int(5e4))
        if len(adata.obs['batch'].cat.categories) > 1:
            entropy_score = batch_entropy_mixing_score(adata.obsm['X_umap'], adata.obs['batch'])
            log.info('batch_entropy_mixing_score: {:.3f}'.format(entropy_score))
            adata.uns['batch_entropy_mixing_score'] = entropy_score

        if cell_type in adata.obs:
            sil_score = silhouette_score(adata.obsm['X_umap'], adata.obs[cell_type].cat.codes)
            log.info("silhouette_score: {:.3f}".format(sil_score))
            adata.uns['silhouette_score'] = sil_score
    
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

import argparse



def main():
    parser = argparse.ArgumentParser(description='Online single-cell data integration through projecting heterogeneous datasets into a common cell-embedding space')
    
    parser.add_argument('--data_list', '-d', type=str, nargs='+', default=[])
    parser.add_argument('--batch_categories', '-b', type=str, nargs='+', default=None)
    parser.add_argument('--join', type=str, default='inner')
    parser.add_argument('--batch_key', type=str, default='batch')
    parser.add_argument('--batch_name', type=str, default='batch')
    parser.add_argument('--profile', type=str, default='RNA')
    parser.add_argument('--subsets', type=str, nargs='+', default=None)
    parser.add_argument('--groupby', type=str, default=None)
    
    parser.add_argument('--min_features', type=int, default=None)
    parser.add_argument('--min_cells', type=int, default=3)
    parser.add_argument('--n_top_features', default=None)
    parser.add_argument('--target_sum', type=int, default=None)
    parser.add_argument('--min_cell_per_batch', type=int, default=10)
    parser.add_argument('--processed', action='store_true', default=False)
    parser.add_argument('--fraction', type=float, default=None)
    parser.add_argument('--n_obs', type=int, default=None)
    parser.add_argument('--use_layer', type=str, default='X')
    parser.add_argument('--backed', action='store_true', default=False)

    parser.add_argument('--projection', '-p', default=None)
    parser.add_argument('--impute', type=str, default=None)
    parser.add_argument('--outdir', '-o', type=str, default='output/')
    
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('-g','--gpu', type=int, default=0)
    parser.add_argument('--max_iteration', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=124)
    parser.add_argument('--chunk_size', type=int, default=20000)
    parser.add_argument('--ignore_umap', action='store_true')
    parser.add_argument('--repeat', action='store_true')
    parser.add_argument('--assess', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--keep_mt', action='store_true')
    parser.add_argument('--name', type=str, default=None)
    # parser.add_argument('--version', type=int, default=2)
    # parser.add_argument('--k', type=str, default=30)
    # parser.add_argument('--embed', type=str, default='UMAP')
#     parser.add_argument('--beta', type=float, default=0.5)
#     parser.add_argument('--hid_dim', type=int, default=1024)


    args = parser.parse_args()
    
    from scalex import SCALEX
    adata = SCALEX(
        args.data_list, 
        batch_categories=args.batch_categories,
        profile=args.profile,
        join=args.join, 
        batch_key=args.batch_key, 
        groupby=args.groupby,
        subsets=args.subsets,
        min_features=args.min_features, 
        min_cells=args.min_cells, 
        target_sum=args.target_sum,
        n_top_features=args.n_top_features, 
        min_cell_per_batch=args.min_cell_per_batch,
        fraction=args.fraction,
        n_obs=args.n_obs,
        processed=args.processed,
        keep_mt=args.keep_mt,
        use_layer=args.use_layer,
        backed=args.backed,
        batch_size=args.batch_size, 
        lr=args.lr, 
        max_iteration=args.max_iteration, 
        impute=args.impute,
        batch_name=args.batch_name, 
        seed=args.seed, 
        gpu=args.gpu, 
        outdir=args.outdir, 
        projection=args.projection, 
        chunk_size=args.chunk_size,
        ignore_umap=args.ignore_umap,
        repeat=args.repeat,
        name=args.name,
        verbose=True,
        assess=args.assess,
        eval=args.eval,
        num_workers=args.num_workers,
        show=False,
    )
    
        
if __name__ == '__main__':
    main()

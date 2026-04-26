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
        min_cell_per_batch:int=200,
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
            # join='outer',
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
import sys


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------

def _add_integrate_args(parser):
    """Register all integration arguments onto a parser (shared by both the
    flat legacy call and the 'integrate' subcommand)."""
    g = parser.add_argument_group('Input')
    g.add_argument('--data_list', '-d', type=str, nargs='+', default=[],
                   metavar='FILE', help='Input file(s): h5ad, h5mu, mtx, csv, txt')
    g.add_argument('--batch_categories', '-b', type=str, nargs='+', default=None,
                   metavar='NAME', help='Batch labels (one per file)')
    g.add_argument('--batch_key', type=str, default='batch',
                   help='obs column to use as batch key (default: batch)')
    g.add_argument('--batch_name', type=str, default='batch',
                   help='Name written to adata.obs for batch column (default: batch)')
    g.add_argument('--profile', type=str, default='RNA', choices=['RNA', 'ATAC'],
                   help='Data modality (default: RNA)')
    g.add_argument('--join', type=str, default='inner', choices=['inner', 'outer'],
                   help='How to join gene lists across batches (default: inner)')
    g.add_argument('--subsets', type=str, nargs='+', default=None,
                   metavar='KEY=VAL', help='Subset cells before integration')
    g.add_argument('--groupby', type=str, default=None,
                   help='obs column to split data into groups')

    g = parser.add_argument_group('Preprocessing')
    g.add_argument('--min_features', type=int, default=None,
                   help='Minimum genes per cell (default: 600 for RNA)')
    g.add_argument('--min_cells', type=int, default=3,
                   help='Minimum cells per gene (default: 3)')
    g.add_argument('--n_top_features', default=None,
                   help='Number of highly variable genes (default: all)')
    g.add_argument('--target_sum', type=int, default=None,
                   help='Normalization target sum (default: None)')
    g.add_argument('--min_cell_per_batch', type=int, default=200,
                   help='Minimum cells required per batch (default: 200)')
    g.add_argument('--processed', action='store_true', default=False,
                   help='Skip preprocessing (data already normalized)')
    g.add_argument('--fraction', type=float, default=None,
                   help='Downsample fraction per batch')
    g.add_argument('--n_obs', type=int, default=None,
                   help='Downsample to this many cells per batch')
    g.add_argument('--use_layer', type=str, default='X',
                   help='AnnData layer to use as input (default: X)')
    g.add_argument('--backed', action='store_true', default=False,
                   help='Load h5ad in backed mode (saves memory)')
    g.add_argument('--keep_mt', action='store_true',
                   help='Keep mitochondrial genes (default: remove)')

    g = parser.add_argument_group('Model')
    g.add_argument('--projection', '-p', default=None, metavar='OUTDIR',
                   help='Project onto an existing model (path to previous outdir)')
    g.add_argument('--impute', type=str, default=None,
                   help='Impute expression for this batch')
    g.add_argument('--lr', type=float, default=2e-4,
                   help='Learning rate (default: 2e-4)')
    g.add_argument('--batch_size', type=int, default=64,
                   help='Training batch size (default: 64)')
    g.add_argument('--max_iteration', type=int, default=30000,
                   help='Maximum training iterations (default: 30000)')
    g.add_argument('--seed', type=int, default=124,
                   help='Random seed (default: 124)')
    g.add_argument('--chunk_size', type=int, default=20000,
                   help='Chunk size for large datasets (default: 20000)')
    g.add_argument('-g', '--gpu', type=int, default=0,
                   help='GPU device index; -1 for CPU (default: 0)')
    g.add_argument('--num_workers', type=int, default=4,
                   help='DataLoader worker processes (default: 4)')

    g = parser.add_argument_group('Output')
    g.add_argument('--outdir', '-o', type=str, default='output/',
                   help='Output directory (default: output/)')
    g.add_argument('--name', type=str, default=None,
                   help='Run name prefix for saved files')
    g.add_argument('--ignore_umap', action='store_true',
                   help='Skip UMAP computation')
    g.add_argument('--repeat', action='store_true',
                   help='Concatenate projected data with reference')
    g.add_argument('--assess', action='store_true',
                   help='Compute batch mixing and silhouette scores')
    g.add_argument('--eval', action='store_true',
                   help='Evaluate on held-out data')


def _run_integrate(argv):
    parser = argparse.ArgumentParser(
        prog='scalex integrate',
        description='Integrate multiple single-cell datasets via SCALEX VAE',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_integrate_args(parser)
    args = parser.parse_args(argv)

    from scalex import SCALEX
    SCALEX(
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


def _run_cluster(argv):
    parser = argparse.ArgumentParser(
        prog='scalex cluster',
        description='Cluster cells in a SCALEX output h5ad using the latent embedding',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--adata', '-i', type=str, required=True,
                        help='Path to h5ad file (e.g. output/adata.h5ad)')
    parser.add_argument('--use_rep', type=str, default='latent',
                        help='obsm key for the embedding to cluster on')
    parser.add_argument('--resolution', '-r', type=float, nargs='+', default=[0.5],
                        metavar='R',
                        help='Leiden resolution(s). Pass multiple values to sweep '
                             '(e.g. --resolution 0.3 0.5 1.0)')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='k-NN graph neighbours')
    parser.add_argument('--n_pcs', type=int, default=None,
                        help='PCs to use when building the neighbour graph '
                             '(default: use full embedding)')
    parser.add_argument('--key_added', type=str, default=None,
                        help='obs column name for cluster labels. '
                             'Defaults to "leiden" (single resolution) or '
                             '"leiden_<R>" (multiple resolutions)')
    parser.add_argument('--outdir', '-o', type=str, default=None,
                        help='Directory to write updated h5ad and UMAP plots '
                             '(default: same directory as --adata)')
    parser.add_argument('--plot', action='store_true',
                        help='Save UMAP coloured by each resolution')
    parser.add_argument('--batch', type=str, default='batch',
                        help='obs column for batch labels used in UMAP plot')
    args = parser.parse_args(argv)

    import os
    import scanpy as sc
    import anndata as ad

    print(f'Loading {args.adata} …')
    adata = ad.read_h5ad(args.adata)

    if args.use_rep not in adata.obsm:
        raise ValueError(
            f"'{args.use_rep}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )

    print(f'Building k-NN graph (k={args.n_neighbors}, rep={args.use_rep}) …')
    sc.pp.neighbors(adata, use_rep=args.use_rep, n_neighbors=args.n_neighbors,
                    n_pcs=args.n_pcs)

    cluster_keys = []
    for res in args.resolution:
        key = args.key_added if (args.key_added and len(args.resolution) == 1) \
              else f'leiden_{res}'
        if len(args.resolution) == 1 and args.key_added is None:
            key = 'leiden'
        print(f'Leiden clustering at resolution {res} → obs["{key}"] …')
        sc.tl.leiden(adata, resolution=res, key_added=key)
        n_clusters = adata.obs[key].nunique()
        print(f'  {n_clusters} clusters found')
        cluster_keys.append(key)

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.adata))
    os.makedirs(outdir, exist_ok=True)

    out_path = os.path.join(outdir, os.path.basename(args.adata))
    print(f'Saving → {out_path}')
    adata.write_h5ad(out_path)

    if args.plot:
        if 'X_umap' not in adata.obsm:
            print('Computing UMAP …')
            sc.tl.umap(adata)
        color_cols = ([args.batch] if args.batch in adata.obs else []) + cluster_keys
        sc.settings.figdir = outdir
        sc.pl.umap(adata, color=color_cols, save='_cluster.png', show=False)
        print(f'UMAP saved → {outdir}/umap_cluster.png')

    print('Done.')


def _run_annotate(argv):
    parser = argparse.ArgumentParser(
        prog='scalex annotate',
        description='Annotate clusters with marker genes and GO enrichment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--adata', '-i', type=str, required=True,
                        help='Path to h5ad file')
    parser.add_argument('--groupby', '-g', type=str, default='leiden',
                        help='obs column holding cluster labels')
    parser.add_argument('--markers', type=str, default='human',
                        choices=['human', 'mouse', 'none'],
                        help='Built-in cell-type marker set to display')
    parser.add_argument('--method', type=str, default='wilcoxon',
                        choices=['wilcoxon', 't-test', 'logreg'],
                        help='Differential expression test method')
    parser.add_argument('--top_n', type=int, default=300,
                        help='Top marker genes per cluster')
    parser.add_argument('--pval_cutoff', type=float, default=0.01,
                        help='Adjusted p-value cutoff for markers')
    parser.add_argument('--logfc_cutoff', type=float, default=1.5,
                        help='Log fold-change cutoff for markers')
    parser.add_argument('--gene_sets', type=str,
                        default='GO_Biological_Process_2023',
                        help='gseapy gene set library for enrichment')
    parser.add_argument('--no_enrich', action='store_true',
                        help='Skip GO enrichment (faster)')
    parser.add_argument('--outdir', '-o', type=str, default=None,
                        help='Output directory for results and plots '
                             '(default: same directory as --adata)')
    parser.add_argument('--processed', action='store_true',
                        help='Skip normalization (data already log-normalized)')
    args = parser.parse_args(argv)

    import os
    import anndata as ad

    print(f'Loading {args.adata} …')
    adata = ad.read_h5ad(args.adata)

    if args.groupby not in adata.obs:
        raise ValueError(
            f"'{args.groupby}' not found in adata.obs. "
            f"Run 'scalex cluster' first, or choose from: "
            f"{[c for c in adata.obs.columns if 'leiden' in c or 'cluster' in c]}"
        )

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.adata))
    os.makedirs(outdir, exist_ok=True)

    from scalex.pp.markers import get_markers
    print(f'Computing marker genes (method={args.method}, top_n={args.top_n}) …')
    markers = get_markers(
        adata,
        groupby=args.groupby,
        method=args.method,
        top_n=args.top_n,
        pval_cutoff=args.pval_cutoff,
        logfc_cutoff=args.logfc_cutoff,
        processed=args.processed,
    )

    import pandas as pd
    marker_df = pd.DataFrame.from_dict(
        {k: pd.Series(v) for k, v in markers.items()}
    )
    marker_path = os.path.join(outdir, 'markers.csv')
    marker_df.to_csv(marker_path, index=False)
    print(f'Markers saved → {marker_path}')

    from scalex.pl.plot import plot_heatmap
    from scalex.pp.celltype import reorder_marker_dict_diagonal
    from scalex.data import aggregate_data
    print('Plotting diagonal heatmap …')
    avg = aggregate_data(adata, groupby=args.groupby, scale=True)
    ordered = reorder_marker_dict_diagonal(markers, avg, groupby=args.groupby)
    plot_heatmap(
        rna=avg,
        genes=ordered,
        groupby=args.groupby,
        labels=['gene'],
        show_gene_labels='bar',
        save=os.path.join(outdir, 'heatmap.pdf'),
    )
    print(f'Heatmap saved → {outdir}/heatmap.pdf')

    if not args.no_enrich:
        print('Running GO enrichment …')
        try:
            from scalex.pp.enrichment import enrich_and_plot
            enrich_and_plot(
                markers,
                gene_sets=args.gene_sets,
                save=os.path.join(outdir, 'go_enrichment.pdf'),
            )
            print(f'GO enrichment saved → {outdir}/go_enrichment.pdf')
        except Exception as e:
            print(f'  GO enrichment skipped: {e}')

    out_path = os.path.join(outdir, os.path.basename(args.adata))
    adata.write_h5ad(out_path)
    print(f'Updated adata saved → {out_path}')
    print('Done.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_SUBCOMMANDS = {
    'integrate': (_run_integrate, 'Integrate datasets (VAE batch correction)'),
    'cluster':   (_run_cluster,   'Cluster cells on the SCALEX latent embedding'),
    'annotate':  (_run_annotate,  'Annotate clusters with markers and GO enrichment'),
}


def main():
    # If the first argument is not a known subcommand (or there are none),
    # fall through to the original flat integrate behaviour so that
    # existing scripts and pipelines continue to work unchanged.
    if len(sys.argv) < 2 or sys.argv[1].startswith('-') or sys.argv[1] not in _SUBCOMMANDS:
        _run_integrate(sys.argv[1:])
        return

    subcommand = sys.argv[1]
    if subcommand in ('integrate', 'cluster', 'annotate'):
        _SUBCOMMANDS[subcommand][0](sys.argv[2:])
    else:
        # Should never reach here given the guard above, but show help if so
        _print_help()
        sys.exit(1)


def _print_help():
    print('usage: scalex [-h] <command> [options]\n')
    print('Online single-cell data integration through projecting heterogeneous')
    print('datasets into a common cell-embedding space\n')
    print('commands:')
    w = max(len(k) for k in _SUBCOMMANDS)
    for name, (_, desc) in _SUBCOMMANDS.items():
        print(f'  {name:<{w}}  {desc}')
    print('\nRun  scalex <command> --help  for per-command options.')
    print('Omitting <command> defaults to integrate (backward compatible).')


if __name__ == '__main__':
    main()

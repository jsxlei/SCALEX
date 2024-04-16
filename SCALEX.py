#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Wed 10 Jul 2019 08:42:21 PM CST

# File Name: SCALEX.py
# Description:

"""


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Online single-cell data integration through projecting heterogeneous datasets into a common cell-embedding space')
    
    parser.add_argument('--data_list', '-d', type=str, nargs='+', default=[])
    parser.add_argument('--batch_categories', '-b', type=str, nargs='+', default=None)
    parser.add_argument('--join', type=str, default='inner')
    parser.add_argument('--batch_key', type=str, default='batch')
    parser.add_argument('--batch_name', type=str, default='batch')
    parser.add_argument('--profile', type=str, default='RNA')
    parser.add_argument('--test_list', '-t', type=str, nargs='+', default=[])
    
    parser.add_argument('--min_features', type=int, default=None)
    parser.add_argument('--min_cells', type=int, default=3)
    parser.add_argument('--n_top_features', default=None)
    parser.add_argument('--target_sum', type=int, default=None)
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
        min_features=args.min_features, 
        min_cells=args.min_cells, 
        target_sum=args.target_sum,
        n_top_features=args.n_top_features, 
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
        verbose=True,
        assess=args.assess,
        eval=args.eval,
        num_workers=args.num_workers,
        show=False,
    )
        

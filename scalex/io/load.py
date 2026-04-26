"""File I/O: load, concatenate, and download single-cell datasets."""

import os
from glob import glob

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import issparse
import scanpy as sc
from anndata import AnnData, concat

from scalex.utils import DATA_PATH, GENOME_PATH, DEFAULT_CHUNK_SIZE

CHUNK_SIZE = DEFAULT_CHUNK_SIZE

__all__ = [
    'DATA_PATH', 'GENOME_PATH', 'CHUNK_SIZE',
    'read_mtx', 'load_file', 'load_files', 'concat_data', 'download_file',
]


def read_mtx(path: str) -> AnnData:
    """Read an MTX format data folder.

    Expects the folder to contain:
    - A matrix file (``count.mtx``, ``matrix.mtx``, or ``data.mtx``, optionally gzipped)
    - A barcode file (e.g. ``barcodes.txt``)
    - A feature/gene/peaks file (e.g. ``features.txt``)

    Parameters
    ----------
    path : str
        Directory path containing the MTX files.

    Returns
    -------
    AnnData
        Annotated data matrix with cells as rows and features as columns.
    """
    for filename in glob(path + '/*'):
        if ('count' in filename or 'matrix' in filename or 'data' in filename) and ('mtx' in filename):
            adata = sc.read_mtx(filename).T
    for filename in glob(path + '/*'):
        if 'barcode' in filename:
            barcode = pd.read_csv(filename, sep='\t', header=None).iloc[:, -1].values
            adata.obs = pd.DataFrame(index=barcode)
        if 'gene' in filename or 'peaks' in filename:
            gene = pd.read_csv(filename, sep='\t', header=None).iloc[:, -1].values
            adata.var = pd.DataFrame(index=gene)
        elif 'feature' in filename:
            gene = pd.read_csv(filename, sep='\t', header=None).iloc[:, 1].values
            adata.var = pd.DataFrame(index=gene)
    return adata


def load_file(path: str, backed: bool = False) -> AnnData:
    """Load a single-cell dataset from a file or directory.

    Supports H5AD, CSV, TSV, MTX directories, and MuData formats.

    Parameters
    ----------
    path : str
        Path to the file or directory, or a dataset name to look up in
        ``~/.scalex/``.
    backed : bool, default False
        Open the H5AD file in backed mode (memory-mapped).

    Returns
    -------
    AnnData
        Annotated data matrix.

    Raises
    ------
    ValueError
        If the file or directory does not exist.
    """
    print(path)
    if os.path.exists(DATA_PATH + path + '.h5ad'):
        adata = sc.read_h5ad(DATA_PATH + path + '.h5ad', backed=backed)
    elif path.endswith(tuple(['.h5mu/rna', '.h5mu/atac', 'h5mu/prot', 'h5mu/adt'])):
        import muon as mu
        adata = mu.read(path, backed=backed)
    elif os.path.isdir(path):
        adata = read_mtx(path)
    elif os.path.isfile(path):
        if path.endswith(('.csv', '.csv.gz')):
            adata = sc.read_csv(path).T
        elif path.endswith(('.txt', '.txt.gz', '.tsv', '.tsv.gz')):
            df = pd.read_csv(path, sep='\t', index_col=0).T
            adata = AnnData(df.values, dict(obs_names=df.index.values), dict(var_names=df.columns.values))
        elif path.endswith('.h5ad'):
            adata = sc.read_h5ad(path)
    else:
        raise ValueError(f"File {path!r} not found. Check the path and try again.")

    if not issparse(adata.X) and not backed:
        adata.X = scipy.sparse.csr_matrix(adata.X)
    adata.var_names_make_unique()
    return adata


def load_files(root: str) -> AnnData:
    """Load single-cell data from a path or a glob pattern.

    Parameters
    ----------
    root : str
        Path to a file/directory, or a glob pattern ending in ``*``.

    Returns
    -------
    AnnData
        Single dataset or concatenation of all matched datasets.
    """
    if root.split('/')[-1] == '*':
        adata_list = [load_file(p) for p in sorted(glob(root))]
        return AnnData.concatenate(*adata_list, batch_key='sub_batch', index_unique=None)
    return load_file(root)


def concat_data(
    data_list,
    batch_categories=None,
    join: str = 'inner',
    batch_key: str = 'batch',
    index_unique=None,
    save=None,
) -> AnnData:
    """Concatenate multiple datasets along the observations axis.

    Parameters
    ----------
    data_list : list[str | AnnData]
        Paths to AnnData files or AnnData objects to concatenate.
    batch_categories : list[str] | None
        Labels for each batch. Defaults to increasing integers.
    join : {'inner', 'outer'}, default 'inner'
        How to handle variable mismatches across batches.
    batch_key : str, default 'batch'
        Key in ``obs`` where the batch label is stored.
    index_unique : str | None
        Separator for making obs indices unique, or None to keep as-is.
    save : str | None
        If provided, write the merged AnnData to this path.

    Returns
    -------
    AnnData
        Merged annotated data matrix.
    """
    if len(data_list) == 1:
        return load_files(data_list[0])
    if isinstance(data_list, str):
        return load_files(data_list)
    if isinstance(data_list, AnnData):
        return data_list

    adata_list = []
    for root in data_list:
        if isinstance(root, AnnData):
            adata_list.append(root)
        else:
            adata_list.append(load_files(root))

    adata_concat = concat(adata_list, join=join, label=batch_key, keys=batch_categories, index_unique=index_unique)
    if batch_categories is None:
        adata_concat.obs['batch'] = 'batch'
    return adata_concat


def download_file(url: str, local_filename: str) -> None:
    """Download a file from a URL using wget.

    Parameters
    ----------
    url : str
        URL of the file to download.
    local_filename : str
        Local path where the file should be saved.
    """
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    try:
        import wget
        wget.download(url, local_filename)
        print(f"File downloaded successfully: {local_filename}")
    except Exception as e:
        print(f"An error occurred while downloading the file: {e}")

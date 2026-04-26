"""Preprocessing pipelines for RNA, ATAC, and ADT single-cell data."""

import os
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import issparse, csr_matrix
from anndata import AnnData
import scanpy as sc
from sklearn.preprocessing import MaxAbsScaler

from scalex.utils import (
    DEFAULT_TARGET_SUM, DEFAULT_N_TOP_FEATURES,
    DEFAULT_MIN_FEATURES_RNA, DEFAULT_MIN_FEATURES_ATAC,
    DEFAULT_MIN_CELLS, DEFAULT_MIN_CELL_PER_BATCH, DEFAULT_CHUNK_SIZE,
    PROFILE_RNA, PROFILE_ATAC, PROFILE_ADT,
)

CHUNK_SIZE = DEFAULT_CHUNK_SIZE


def aggregate_data(
    rna: AnnData,
    groupby: str = 'cell_type',
    processed: bool = False,
    target_sum: float = 1e4,
    scale: bool = False,
) -> AnnData:
    """Aggregate (pseudobulk) cells by group using mean or sum.

    Parameters
    ----------
    rna : AnnData
        Single-cell expression data.
    groupby : str, default 'cell_type'
        Column in ``obs`` to group by.
    processed : bool, default False
        If True, use ``adata.raw`` as-is and compute mean directly.
        If False, sum counts, normalize, and log-transform.
    target_sum : float, default 1e4
        Target library size for normalization (only when ``processed=False``).
    scale : bool or str, default False
        Apply scaling. ``True`` or ``'scale'`` → z-score scaling.
        ``'quantile'`` → quantile normalization.

    Returns
    -------
    AnnData
        Pseudobulk expression matrix with one row per group.
    """
    if scale is True:
        scale = 'scale'

    if processed:
        if rna.raw is not None:
            rna = rna.raw.to_adata()
        rna_agg = sc.get.aggregate(rna, by=groupby, func='mean')
        rna_agg.X = rna_agg.layers['mean']
    else:
        rna_agg = sc.get.aggregate(rna, by=groupby, func='sum')
        rna_agg.X = rna_agg.layers['sum']
        sc.pp.normalize_total(rna_agg, target_sum=target_sum)
        sc.pp.log1p(rna_agg)

    if scale == 'scale':
        sc.pp.scale(rna_agg, zero_center=True)
    elif scale == 'quantile':
        X = rna_agg.X.toarray() if issparse(rna_agg.X) else np.array(rna_agg.X)
        rank_idx = np.argsort(X, axis=1)
        mean_per_rank = np.sort(X, axis=1).mean(axis=0)
        X_norm = np.empty_like(X)
        for i in range(X.shape[0]):
            X_norm[i, rank_idx[i]] = mean_per_rank
        rna_agg.X = scipy.sparse.csr_matrix(X_norm) if issparse(rna_agg.X) else X_norm

    return rna_agg


def preprocessing_rna(
    adata: AnnData,
    min_features: int = DEFAULT_MIN_FEATURES_RNA,
    min_cells: int = DEFAULT_MIN_CELLS,
    target_sum: int = DEFAULT_TARGET_SUM,
    n_top_features=DEFAULT_N_TOP_FEATURES,
    chunk_size: int = CHUNK_SIZE,
    min_cell_per_batch: int = DEFAULT_MIN_CELL_PER_BATCH,
    keep_mt: bool = False,
    backed: bool = False,
    log=None,
) -> AnnData:
    """Preprocess single-cell RNA-seq data.

    Applies quality filtering, normalization, log transformation, highly
    variable gene selection, and MaxAbs scaling in batch-aware fashion.

    Parameters
    ----------
    adata : AnnData
        Raw count matrix (cells × genes).
    min_features : int, default 600
        Minimum number of detected genes per cell.
    min_cells : int, default 3
        Minimum number of cells expressing a gene.
    target_sum : int, default 10000
        Library size for total-count normalization.
    n_top_features : int or list[str], default 2000
        Number of highly variable genes to retain, or an explicit gene list.
    chunk_size : int, default 20000
        Internal chunk size for batch processing.
    min_cell_per_batch : int, default 10
        Minimum cells per batch for HVG selection.
    keep_mt : bool, default False
        Retain mitochondrial genes (MT-, mt-, ERCC).
    backed : bool, default False
        Whether ``adata`` is opened in backed (memory-mapped) mode.
    log : logging.Logger | None
        Optional logger for progress messages.

    Returns
    -------
    AnnData
        Preprocessed data with HVGs selected and scaled.
    """
    if min_features is None:
        min_features = DEFAULT_MIN_FEATURES_RNA
    if n_top_features is None:
        n_top_features = DEFAULT_N_TOP_FEATURES
    if target_sum is None:
        target_sum = DEFAULT_TARGET_SUM

    if log:
        log.info('Preprocessing')
    if not isinstance(adata.X, csr_matrix) and not backed and not adata.isbacked:
        adata.X = scipy.sparse.csr_matrix(adata.X)

    if not keep_mt:
        if log:
            log.info('Filtering out MT genes')
        adata = adata[:, [gene for gene in adata.var_names
                          if not str(gene).startswith(('ERCC', 'MT-', 'mt-'))]]

    if log:
        log.info('Filtering cells')
    sc.pp.filter_cells(adata, min_genes=min_features)

    if log:
        log.info('Filtering features')
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if log:
        log.info('Normalizing total per cell')
    sc.pp.normalize_total(adata, target_sum=target_sum)

    if log:
        log.info('Log1p transforming')
    sc.pp.log1p(adata)

    adata.raw = adata

    if log:
        log.info('Finding variable features')
    if isinstance(n_top_features, int) and n_top_features > 0:
        batch_counts = adata.obs['batch'].value_counts()
        valid_batches = batch_counts[batch_counts >= min_cell_per_batch].index
        if len(valid_batches) < 2:
            adata_ = adata
        elif len(valid_batches) < len(batch_counts):
            adata_ = adata[adata.obs['batch'].isin(valid_batches)].copy()
            print(batch_counts[batch_counts >= min_cell_per_batch])
        else:
            adata_ = adata
        if log:
            log.info(adata_.obs['batch'].value_counts())
        sc.pp.highly_variable_genes(adata_, n_top_genes=n_top_features, batch_key='batch')
        adata = adata[:, adata_.var.highly_variable].copy()
    elif not isinstance(n_top_features, int):
        raw = adata.raw.to_adata()
        adata = reindex(adata, n_top_features)
        adata.raw = raw

    if log:
        log.info('Batch specific maxabs scaling')
    adata.X = MaxAbsScaler().fit_transform(adata.X)
    if log:
        log.info('Processed dataset shape: {}'.format(adata.shape))
    return adata


def preprocessing_atac(
    adata: AnnData,
    min_features: int = DEFAULT_MIN_FEATURES_ATAC,
    min_cells: int = DEFAULT_MIN_CELLS,
    target_sum=None,
    n_top_features=100000,
    chunk_size: int = CHUNK_SIZE,
    min_cell_per_batch: int = DEFAULT_MIN_CELL_PER_BATCH,
    backed: bool = False,
    log=None,
) -> AnnData:
    """Preprocess single-cell ATAC-seq data.

    Parameters
    ----------
    adata : AnnData
        Raw accessibility count matrix (cells × peaks).
    min_features : int, default 100
        Minimum number of detected peaks per cell.
    min_cells : int, default 3
        Minimum number of cells with a peak open.
    target_sum : int | None, default None
        Library size for normalization. Defaults to 10,000 if None.
    n_top_features : int or list[str], default 100000
        Number of variable peaks to retain, or an explicit peak list.
    chunk_size : int, default 20000
        Internal chunk size for batch processing.
    min_cell_per_batch : int, default 10
        Minimum cells per batch for feature selection.
    backed : bool, default False
        Whether ``adata`` is opened in backed (memory-mapped) mode.
    log : logging.Logger | None
        Optional logger for progress messages.

    Returns
    -------
    AnnData
        Preprocessed data with selected features and MaxAbs scaled.
    """
    if min_features is None:
        min_features = DEFAULT_MIN_FEATURES_ATAC
    if n_top_features is None:
        n_top_features = 100000
    if target_sum is None:
        target_sum = DEFAULT_TARGET_SUM

    batch_counts = adata.obs['batch'].value_counts()
    print('min_cell_per_batch', min_cell_per_batch)
    valid_batches = batch_counts[batch_counts >= min_cell_per_batch].index
    if len(valid_batches) < len(batch_counts):
        adata = adata[adata.obs['batch'].isin(valid_batches)].copy()

    if log:
        log.info('Preprocessing')
    if not isinstance(adata.X, csr_matrix):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    if log:
        log.info('Filtering cells')
    sc.pp.filter_cells(adata, min_genes=min_features)

    if log:
        log.info('Filtering features')
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if log:
        log.info('Finding variable features')
    if isinstance(n_top_features, int) and n_top_features > 0 and n_top_features < adata.shape[1]:
        from scalex.atac.snapatac2._basic import select_features
        select_features(adata, n_top_features, inplace=True)
        adata = adata[:, adata.var['selected']].copy()
        print(adata.shape)
    elif not isinstance(n_top_features, int):
        adata = reindex(adata, n_top_features)

    if log:
        log.info('Batch specific maxabs scaling')
    adata.X = MaxAbsScaler().fit_transform(adata.X)
    if log:
        log.info('Processed dataset shape: {}'.format(adata.shape))
    return adata


def preprocessing_adt(
    adata: AnnData,
    min_features: int = 10,
    min_cells: int = DEFAULT_MIN_CELLS,
    target_sum: int = DEFAULT_TARGET_SUM,
    chunk_size: int = CHUNK_SIZE,
    backed: bool = False,
    log=None,
) -> AnnData:
    """Preprocess single-cell ADT (CITE-seq protein) data.

    Parameters
    ----------
    adata : AnnData
        Raw ADT count matrix (cells × proteins).
    min_features : int, default 10
        Minimum number of detected proteins per cell.
    min_cells : int, default 3
        Minimum number of cells with a protein detected.
    target_sum : int, default 10000
        Library size for normalization.
    chunk_size : int, default 20000
        Internal chunk size for batch processing.
    backed : bool, default False
        Whether ``adata`` is opened in backed (memory-mapped) mode.
    log : logging.Logger | None
        Optional logger for progress messages.

    Returns
    -------
    AnnData
        Preprocessed data with MaxAbs scaling applied.
    """
    if log:
        log.info(adata.obs['batch'].value_counts())
    if log:
        log.info('Preprocessing')
    if not isinstance(adata.X, csr_matrix) and not backed and not adata.isbacked:
        adata.X = scipy.sparse.csr_matrix(adata.X)

    if log:
        log.info('Filtering cells')
    sc.pp.filter_cells(adata, min_genes=min_features)

    if log:
        log.info('Filtering features')
    sc.pp.filter_genes(adata, min_cells=min_cells)

    adata.raw = adata

    if log:
        log.info('Batch specific maxabs scaling')
    adata.X = MaxAbsScaler().fit_transform(adata.X)
    if log:
        log.info('Processed dataset shape: {}'.format(adata.shape))
    return adata


def preprocessing(
    adata: AnnData,
    profile: str = PROFILE_RNA,
    min_features: int = DEFAULT_MIN_FEATURES_RNA,
    min_cells: int = DEFAULT_MIN_CELLS,
    target_sum: int = None,
    n_top_features=None,
    min_cell_per_batch: int = DEFAULT_MIN_CELL_PER_BATCH,
    keep_mt: bool = False,
    backed: bool = False,
    chunk_size: int = CHUNK_SIZE,
    log=None,
) -> AnnData:
    """Dispatch preprocessing to the appropriate profile handler.

    Parameters
    ----------
    adata : AnnData
        Raw count matrix.
    profile : {'RNA', 'ATAC', 'ADT'}, default 'RNA'
        Single-cell data type.
    min_features : int, default 600
        Minimum features per cell.
    min_cells : int, default 3
        Minimum cells per feature.
    target_sum : int | None
        Library size for normalization.
    n_top_features : int or list[str] | None
        Number or list of top features to retain.
    min_cell_per_batch : int, default 10
        Minimum cells per batch.
    keep_mt : bool, default False
        Keep mitochondrial genes (RNA only).
    backed : bool, default False
        Backed (memory-mapped) mode.
    chunk_size : int, default 20000
        Chunk size for batch-wise operations.
    log : logging.Logger | None
        Optional logger.

    Returns
    -------
    AnnData
        Preprocessed annotated data matrix.

    Raises
    ------
    ValueError
        If ``profile`` is not one of 'RNA', 'ATAC', 'ADT'.
    """
    if profile == PROFILE_RNA:
        return preprocessing_rna(
            adata,
            min_features=min_features,
            min_cells=min_cells,
            target_sum=target_sum,
            n_top_features=n_top_features,
            min_cell_per_batch=min_cell_per_batch,
            keep_mt=keep_mt,
            backed=backed,
            chunk_size=chunk_size,
            log=log,
        )
    elif profile == PROFILE_ATAC:
        return preprocessing_atac(
            adata,
            min_features=min_features,
            min_cells=min_cells,
            target_sum=target_sum,
            n_top_features=n_top_features,
            min_cell_per_batch=min_cell_per_batch,
            chunk_size=chunk_size,
            backed=backed,
            log=log,
        )
    elif profile == PROFILE_ADT:
        return preprocessing_adt(
            adata,
            min_features=min_features,
            min_cells=min_cells,
            backed=backed,
            log=log,
        )
    else:
        raise ValueError(
            f"Unknown profile {profile!r}. Expected one of: 'RNA', 'ATAC', 'ADT'."
        )


def clr_normalize(matrix) -> np.ndarray:
    """Centered log-ratio (CLR) normalization across rows (cells).

    Parameters
    ----------
    matrix : array-like
        Cells × features expression matrix.

    Returns
    -------
    np.ndarray
        CLR-normalized matrix.
    """
    matrix = matrix + 1
    geometric_mean = np.exp(np.mean(np.log(matrix), axis=1)).reshape(-1, 1)
    return np.log(matrix / geometric_mean)


def batch_scale(adata: AnnData, chunk_size: int = CHUNK_SIZE) -> AnnData:
    """Apply batch-specific MaxAbs scaling in place.

    Parameters
    ----------
    adata : AnnData
        Annotated data with ``adata.obs['batch']`` column.
    chunk_size : int, default 20000
        Unused; retained for API compatibility.

    Returns
    -------
    AnnData
        Same ``adata`` with ``X`` scaled per batch.
    """
    for b in adata.obs['batch'].unique():
        idx = np.where(adata.obs['batch'] == b)[0]
        scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])
        adata.X[idx] = scaler.transform(adata.X[idx])
    return adata


def reindex(adata: AnnData, genes, chunk_size: int = CHUNK_SIZE) -> AnnData:
    """Reindex an AnnData to a target gene list, zero-filling missing genes.

    Parameters
    ----------
    adata : AnnData
        Input data.
    genes : list[str] | np.ndarray
        Target gene list to index to.
    chunk_size : int, default 20000
        Unused; retained for API compatibility.

    Returns
    -------
    AnnData
        Reindexed data matrix with the same cell order.
    """
    idx = [i for i, g in enumerate(genes) if g in adata.var_names]
    print('There are {} gene in selected genes'.format(len(idx)))
    if len(idx) == len(genes):
        adata = adata[:, genes].copy()
    else:
        new_X = scipy.sparse.lil_matrix((adata.shape[0], len(genes)))
        new_X[:, idx] = adata[:, genes[idx]].copy().X
        adata = AnnData(new_X.tocsr(), obs=adata.obs, var=pd.DataFrame(index=genes))
    return adata


def convert_mouse_to_human(adata: AnnData, mapping_file: str = None, cache_file: str = None) -> AnnData:
    """Convert mouse gene symbols to human orthologs using BioMart.

    Parameters
    ----------
    adata : AnnData
        Single-cell data with mouse gene symbols in ``var_names``.
    mapping_file : str | None
        Path to a pre-built TSV mapping (columns: ``mouse``, ``human``).
    cache_file : str | None
        Path to cache the downloaded BioMart mapping for reuse.

    Returns
    -------
    AnnData
        Data subset to convertible genes with human gene names.
    """
    if mapping_file and os.path.exists(mapping_file):
        df = pd.read_csv(mapping_file, sep="\t")
    elif cache_file and os.path.exists(cache_file):
        df = pd.read_csv(cache_file, sep="\t")
    else:
        from gseapy import Biomart
        bm = Biomart()
        df = bm.query(
            dataset="mmusculus_gene_ensembl",
            attributes=["external_gene_name", "hsapiens_homolog_associated_gene_name"],
        ).rename(columns={
            "external_gene_name": "mouse",
            "hsapiens_homolog_associated_gene_name": "human",
        })
        df = df.dropna().drop_duplicates("mouse")
        if cache_file:
            df.to_csv(cache_file, sep="\t", index=False)

    map_dict = df.set_index("mouse")["human"].to_dict()  # noqa: E501
    keep = adata.var_names.to_series().map(map_dict).notna().values
    adata2 = adata[:, keep].copy()
    adata2.var["mouse"] = adata2.var_names
    adata2.var["human"] = adata2.var_names.to_series().map(map_dict).values
    adata2.var_names = adata2.var["human"].astype(str).values
    adata2.var_names_make_unique()
    return adata2

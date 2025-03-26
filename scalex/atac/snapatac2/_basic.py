import pybedtools
import pandas as pd
from typing import List
from anndata import AnnData
import numpy as np
from pathlib import Path
import scanpy as sc
import logging

from ._clustering import spectral
from ._knn import knn
from ._misc import aggregate_X


def _find_most_accessible_features(
    feature_count,
    filter_lower_quantile,
    filter_upper_quantile,
    total_features,
) -> np.ndarray:
    idx = np.argsort(feature_count)
    for i in range(idx.size):
        if feature_count[idx[i]] > 0:
            break
    idx = idx[i:]
    n = idx.size
    n_lower = int(filter_lower_quantile * n)
    n_upper = int(filter_upper_quantile * n)
    idx = idx[n_lower:n-n_upper]
    return idx[::-1][:total_features]





def intersect_bed(regions: List[str], bed_file: str) -> List[bool]:
    """
    Check if genomic regions intersect with a BED file.

    Parameters:
    - regions: List of genomic regions as strings (e.g., "chr1:1000-2000").
    - bed_file: Path to the BED file.

    Returns:
    - List of booleans indicating whether each region overlaps with the BED file.
    """
    # Load BED file as an interval tree using pybedtools
    bed_intervals = pybedtools.BedTool(bed_file)

    results = []
    for region in regions:
        # Convert "chr1:1000-2000" to BED format "chr1 1000 2000"
        chrom, coords = region.split(":")
        start, end = map(int, coords.split("-"))

        # Create a temporary interval
        query_interval = pybedtools.BedTool(f"{chrom}\t{start}\t{end}", from_string=True)

        # Check for intersection
        results.append(bool(query_interval.intersect(bed_intervals, u=True)))

    return results




def select_features(
    adata: AnnData | list[AnnData],
    n_features: int = 500000,
    filter_lower_quantile: float = 0.005,
    filter_upper_quantile: float = 0.005,
    whitelist: Path | None = None,
    blacklist: Path | None = None,
    max_iter: int = 1,
    inplace: bool = True,
    n_jobs: int = 8,
    verbose: bool = True,
) -> np.ndarray | list[np.ndarray] | None:
    """
    Perform feature selection by selecting the most accessibile features across
    all cells unless `max_iter` > 1.

    Note
    ----
    This function does not perform the actual subsetting. The feature mask is used by
    various functions to generate submatrices on the fly.
    Features that are zero in all cells will be always removed regardless of the
    filtering criteria.
    For more discussion about feature selection, see: https://github.com/kaizhang/SnapATAC2/discussions/116.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
        `adata` can also be a list of AnnData objects.
        In this case, the function will be applied to each AnnData object in parallel.
    n_features
        Number of features to keep. Note that the final number of features
        may be smaller than this number if there is not enough features that pass
        the filtering criteria.
    filter_lower_quantile
        Lower quantile of the feature count distribution to filter out.
        For example, 0.005 means the bottom 0.5% features with the lowest counts will be removed.
    filter_upper_quantile
        Upper quantile of the feature count distribution to filter out.
        For example, 0.005 means the top 0.5% features with the highest counts will be removed.
        Be aware that when the number of feature is very large, the default value of 0.005 may
        risk removing too many features.
    whitelist
        A user provided bed file containing genome-wide whitelist regions.
        None-zero features listed here will be kept regardless of the other
        filtering criteria.
        If a feature is present in both whitelist and blacklist, it will be kept.
    blacklist 
        A user provided bed file containing genome-wide blacklist regions.
        Features that are overlapped with these regions will be removed.
    max_iter
        If greater than 1, this function will perform iterative clustering and feature selection
        based on variable features found using previous clustering results.
        This is similar to the procedure implemented in ArchR, but we do not recommend it,
        see https://github.com/kaizhang/SnapATAC2/issues/111.
        Default value is 1, which means no iterative clustering is performed.
    inplace
        Perform computation inplace or return result.
    n_jobs
        Number of parallel jobs to use when `adata` is a list.
    verbose
        Whether to print progress messages.
    
    Returns
    -------
    np.ndarray | None:
        If `inplace = False`, return a boolean index mask that does filtering,
        where `True` means that the feature is kept, `False` means the feature is removed.
        Otherwise, store this index mask directly to `.var['selected']`.
    """

    count = np.zeros(adata.shape[1])
    for batch, _, _ in adata.chunked_X(2000):
        count += np.ravel(batch.sum(axis = 0))
    adata.var['count'] = count

    selected_features = _find_most_accessible_features(
        count, filter_lower_quantile, filter_upper_quantile, n_features)

    if blacklist is not None:
        blacklist = np.array(intersect_bed(adata.var_names, str(blacklist)))
        selected_features = selected_features[np.logical_not(blacklist[selected_features])]

    # Iteratively select features
    iter = 1
    while iter < max_iter:
        embedding = spectral(adata, features=selected_features, inplace=False)[1]
        clusters = sc.tl.leiden(knn(embedding, inplace=False))
        rpm = aggregate_X(adata, groupby=clusters).X
        var = np.var(np.log(rpm + 1), axis=0)
        selected_features = np.argsort(var)[::-1][:n_features]

        # Apply blacklist to the result
        if blacklist is not None:
            selected_features = selected_features[np.logical_not(blacklist[selected_features])]
        iter += 1

    result = np.zeros(adata.shape[1], dtype=bool)
    result[selected_features] = True

    # Finally, apply whitelist to the result
    if whitelist is not None:
        whitelist = np.array(intersect_bed(adata.var_names, str(whitelist)))
        whitelist &= count != 0
        result |= whitelist
    
    if verbose:
        logging.info(f"Selected {result.sum()} features.")

    if inplace:
        adata.var["selected"] = result
    else:
        return result
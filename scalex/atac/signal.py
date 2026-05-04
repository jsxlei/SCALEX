"""ATAC-seq signal extraction from bigwig files with read-depth normalization."""
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyBigWig
from tqdm import tqdm

from ..pl.trackplot import TrackSpec


class Atac:
    """Region-based ATAC signal extraction backed by per-cell-type bigwig files.

    Discovers cell types from a flat bigwig directory where files follow the
    ``{cell_type}{suffix}`` naming convention (e.g. ``Foamy_TypeI_unstranded.bw``).

    Normalization uses the total signal sum from each bigwig header
    (``sumData``), scaling to *target_sum* (default 1 million) before an
    optional log1p transform.

    Parameters
    ----------
    bw_dir : str
        Directory containing per-cell-type bigwig files.
    suffix : str
        Filename suffix after the cell-type name (default: ``"_unstranded.bw"``).

    Examples
    --------
    >>> atac = Atac(bw_dir="/path/to/bigwig")
    >>> matrix = atac.atac_matrix(intervals, normalize=True, log1p=True)
    """

    def __init__(self, bw_dir, suffix="_unstranded.bw"):
        self.bw_dir = bw_dir
        self.suffix = suffix
        self._bw_dict = {
            fname[: -len(suffix)]: os.path.join(bw_dir, fname)
            for fname in sorted(os.listdir(bw_dir))
            if fname.endswith(suffix)
        }
        self._total_cache = {}  # cell_type → sumData

    @property
    def cell_types(self):
        return sorted(self._bw_dict.keys())

    def _parse_interval(self, interval):
        m = re.match(r"(.+):(\d+)-(\d+)$", interval)
        if not m:
            raise ValueError(f"Invalid interval {interval!r}. Expected 'chr:start-end'.")
        return m.group(1), int(m.group(2)), int(m.group(3))

    def _get_total(self, cell_type):
        """Return sumData from bigwig header (cached)."""
        if cell_type not in self._total_cache:
            with pyBigWig.open(self._bw_dict[cell_type]) as bw:
                self._total_cache[cell_type] = float(bw.header()["sumData"])
        return self._total_cache[cell_type]

    def fetch(self, interval, cell_type, agg="sum"):
        """Return a scalar ATAC signal for a genomic interval.

        Parameters
        ----------
        interval : str
            Genomic region, e.g. ``"chr1:1000000-1100000"``.
        cell_type : str
            Must be in ``self.cell_types``.
        agg : {"sum", "mean"}
            How to aggregate the signal across the interval.

        Returns
        -------
        float  (raw, un-normalized signal)
        """
        chrom, start, end = self._parse_interval(interval)
        with pyBigWig.open(self._bw_dict[cell_type]) as bw:
            val = bw.stats(chrom, start, end, type=agg, exact=True)[0]
        return float(val) if val is not None else 0.0

    def _compute_ct_column(
        self, intervals_parsed, cell_type, agg, normalize, target_sum, log1p, progress=False, position=0
    ):
        """Compute signal for all intervals for one cell type (bigwig opened once)."""
        results = np.zeros(len(intervals_parsed))

        with pyBigWig.open(self._bw_dict[cell_type]) as bw:
            it = enumerate(intervals_parsed)
            if progress:
                it = enumerate(
                    tqdm(intervals_parsed, desc=cell_type, position=position, leave=True)
                )
            for i, (chrom, start, end) in it:
                val = bw.stats(chrom, start, end, type=agg, exact=True)[0]
                results[i] = float(val) if val is not None else 0.0

        if normalize:
            total = self._get_total(cell_type)
            results = results * (target_sum / max(total, 1.0))

        if log1p:
            results = np.log1p(results)

        return results

    def atac_matrix(
        self,
        intervals,
        cell_types=None,
        agg="sum",
        normalize=True,
        target_sum=1_000_000,
        log1p=True,
        n_jobs=1,
        progress=False,
    ):
        """Generate a (intervals × cell types) ATAC signal matrix.

        Parameters
        ----------
        intervals : list[str]
            Genomic intervals, e.g. ``["chr1:1000-2000", "chr2:3000-4000"]``.
        cell_types : list[str], optional
            Subset of ``self.cell_types``.  Defaults to all.
        agg : {"sum", "mean"}
            How to collapse signal across each interval.
        normalize : bool
            Scale raw signal by ``target_sum / sumData`` (read-depth normalization).
        target_sum : float
            Target read depth for normalization (default: 1 000 000 = RPM).
        log1p : bool
            Apply log1p after normalization.
        n_jobs : int
            Number of cell types to process in parallel (``-1`` uses all).
        progress : bool
            Show per-cell-type tqdm progress bars.

        Returns
        -------
        pd.DataFrame of shape ``(n_intervals, n_cell_types)``.
        """
        if cell_types is None:
            cell_types = self.cell_types

        # Pre-cache totals before any parallelism to avoid redundant I/O
        if normalize:
            for ct in cell_types:
                self._get_total(ct)

        intervals_parsed = [self._parse_interval(iv) for iv in intervals]

        data = {}
        if n_jobs == 1:
            for i, ct in enumerate(cell_types):
                data[ct] = self._compute_ct_column(
                    intervals_parsed, ct, agg, normalize, target_sum, log1p,
                    progress=progress, position=i,
                )
        else:
            n_workers = len(cell_types) if n_jobs == -1 else n_jobs
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = {
                    ex.submit(
                        self._compute_ct_column,
                        intervals_parsed, ct, agg, normalize, target_sum, log1p,
                        progress, i,
                    ): ct
                    for i, ct in enumerate(cell_types)
                }
                for fut in as_completed(futures):
                    data[futures[fut]] = fut.result()

        return pd.DataFrame(data, index=intervals)

    def to_trackspecs(self, interval, cell_types=None, **kwargs):
        """Return a list of :class:`TrackSpec` objects for :func:`compose_tracks`.

        Parameters
        ----------
        interval : str
            Genomic region, e.g. ``"chr3:69729463-69978332"``.
        cell_types : list[str], optional
            Subset of ``self.cell_types``.  Defaults to all.
        **kwargs
            Passed to :class:`TrackSpec` (e.g. ``height``, ``style``).
        """
        if cell_types is None:
            cell_types = self.cell_types
        return [
            TrackSpec("bigwig", data=self._bw_dict[ct], label=ct, **kwargs)
            for ct in cell_types
        ]


def quantile_normalize_atac(atac_df: pd.DataFrame) -> pd.DataFrame:
    """Quantile-normalize an ATAC matrix across cell types (columns).

    Maps each cell type's value distribution to the cross-cell-type mean
    quantile distribution.  Rows = intervals, columns = cell types.
    """
    mat = atac_df.values.astype(float)
    n_inst, n_ct = mat.shape
    if n_ct < 2 or n_inst < 2:
        return atac_df.copy()
    mat_T = mat.T
    rank_idx = np.argsort(np.argsort(mat_T, axis=1), axis=1)
    ref = np.mean(np.sort(mat_T, axis=1), axis=0)
    return pd.DataFrame(ref[rank_idx].T, index=atac_df.index, columns=atac_df.columns)

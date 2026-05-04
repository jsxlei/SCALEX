"""SHAP score extraction across cell types with per-cell-type checkpointing."""
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..pl.trackplot import TrackSpec

_EMPTY = np.array([], dtype=np.int64)


class Shap:
    """Region-based SHAP logo visualization backed by per-cell-type H5 files.

    Discovers cell types from a ``shap_dir`` and assembles a contiguous
    ``(region_len, 4)`` ACGT contribution-score array for any genomic interval.
    Peak coordinates are read from ``bed_filename`` (co-located with
    ``average.counts.bw`` in each cell-type subdirectory).

    Parameters
    ----------
    shap_dir : str, optional
        Directory with ``{cell_type}/`` subdirectories, each containing
        *h5_filename* and *bed_filename*.
    h5_dict : dict, optional
        Explicit ``{cell_type: h5_path}`` mapping (alternative to *shap_dir*).
    h5_filename : str
        H5 filename within each cell-type subdirectory
        (default: ``"average.counts.h5"``).
    h5_key : str
        Dataset path inside the H5 file for SHAP scores
        (default: ``"projected_shap/seq"``).
    bed_filename : str
        BED filename within each cell-type subdirectory used to look up peak
        coordinates (default: ``"fold_0.interpreted_regions.bed"``).

    Examples
    --------
    >>> shap = Shap(shap_dir="/path/to/shap")
    >>> tracks = shap.to_trackspecs("chr3:69729463-69978332",
    ...                             cell_types=["Foamy_TypeI", "Resident"])
    >>> fig = compose_tracks(tracks, region="chr3:69729463-69978332")
    """

    def __init__(
        self,
        shap_dir=None,
        h5_dict=None,
        h5_filename="average.counts.h5",
        h5_key="projected_shap/seq",
        bed_filename="fold_0.interpreted_regions.bed",
    ):
        if shap_dir is not None:
            cell_types = [
                ct for ct in sorted(os.listdir(shap_dir))
                if os.path.isdir(os.path.join(shap_dir, ct))
            ]
            self.h5_dict = {
                ct: os.path.join(shap_dir, ct, h5_filename)
                for ct in cell_types
                if os.path.exists(os.path.join(shap_dir, ct, h5_filename))
            }
            self.bed_dict = {
                ct: os.path.join(shap_dir, ct, bed_filename)
                for ct in self.h5_dict
                if os.path.exists(os.path.join(shap_dir, ct, bed_filename))
            }
        elif h5_dict is not None:
            self.h5_dict = {k: v for k, v in h5_dict.items() if os.path.exists(v)}
            self.bed_dict = {}
        else:
            raise ValueError("Provide either shap_dir or h5_dict.")

        self.h5_key = h5_key
        self._index_cache = {}  # cell_type → {chrom: (starts, ends, h5_idxs)}

    @property
    def cell_types(self):
        return sorted(self.h5_dict.keys())

    def _parse_interval(self, interval):
        m = re.match(r"(.+):(\d+)-(\d+)$", interval)
        if not m:
            raise ValueError(f"Invalid interval {interval!r}. Expected 'chr:start-end'.")
        return m.group(1), int(m.group(2)), int(m.group(3))

    def _build_index(self, cell_type):
        """Load coordinates from the BED file and build a per-chromosome sorted index."""
        if cell_type in self._index_cache:
            return self._index_cache[cell_type]

        bed_path = self.bed_dict.get(cell_type)
        if bed_path is None or not os.path.exists(bed_path):
            self._index_cache[cell_type] = {}
            return {}

        df = pd.read_csv(bed_path, sep="\t", header=None, usecols=[0, 1, 2],
                         names=["chrom", "start", "end"])
        chroms = df["chrom"].values
        starts = df["start"].values.astype(np.int64)
        ends = df["end"].values.astype(np.int64)

        index = {}
        for chrom in np.unique(chroms):
            mask = chroms == chrom
            h5_idxs = np.where(mask)[0]
            order = np.argsort(starts[mask])
            index[chrom] = (
                starts[mask][order],
                ends[mask][order],
                h5_idxs[order],
            )

        self._index_cache[cell_type] = index
        return index

    def _flanked_peaks(self, cell_type, chrom, region_start, region_end):
        """Return (pk_starts, pk_ends, h5_idxs) for peaks overlapping the interval."""
        index = self._build_index(cell_type)
        if chrom not in index:
            return _EMPTY, _EMPTY, _EMPTY
        starts, ends, h5_idxs = index[chrom]
        lo = np.searchsorted(starts, region_end, side="left")
        if lo == 0:
            return _EMPTY, _EMPTY, _EMPTY
        mask = ends[:lo] > region_start
        return starts[:lo][mask], ends[:lo][mask], h5_idxs[:lo][mask]

    def fetch(self, interval, cell_type):
        """Return a ``(region_len, 4)`` ACGT SHAP array for a genomic interval.

        Parameters
        ----------
        interval : str
            Genomic region, e.g. ``"chr1:1000000-1100000"``.
        cell_type : str
            Must be in ``self.cell_types``.

        Returns
        -------
        np.ndarray of shape ``(region_len, 4)`` with columns [A, C, G, T].
        """
        chrom, region_start, region_end = self._parse_interval(interval)
        region_len = region_end - region_start
        pk_starts, pk_ends, h5_idxs = self._flanked_peaks(cell_type, chrom, region_start, region_end)

        with h5py.File(self.h5_dict[cell_type], "r") as h5:
            shap_data = h5[self.h5_key]
            n_bases = shap_data.shape[1]
            seq_len = shap_data.shape[2]

            if h5_idxs.size == 0:
                return np.zeros((region_len, n_bases))

            assembled = []
            cursor = region_start

            for pk_start, pk_end, h5_idx in zip(pk_starts, pk_ends, h5_idxs):
                # BED gives the peak window; H5 stores seq_len bases centered on it
                seq_start = (int(pk_start) + int(pk_end)) // 2 - seq_len // 2

                if cursor < seq_start:
                    gap = min(seq_start, region_end) - cursor
                    assembled.append(np.zeros((gap, n_bases)))
                    cursor = seq_start

                src_start = max(0, cursor - seq_start)
                src_end = min(seq_len, region_end - seq_start)
                if src_end <= src_start:
                    continue

                assembled.append(shap_data[int(h5_idx)][:, src_start:src_end].T)
                cursor = seq_start + src_end

            if cursor < region_end:
                assembled.append(np.zeros((region_end - cursor, n_bases)))

        return np.concatenate(assembled, axis=0)

    def _compute_ct_column(self, intervals_parsed, cell_type, agg, progress=False, position=0):
        """Scalar SHAP aggregates for all intervals for one cell type."""
        results = np.zeros(len(intervals_parsed))

        with h5py.File(self.h5_dict[cell_type], "r") as h5:
            ds = h5[self.h5_key]
            n_bases = ds.shape[1]
            seq_len = ds.shape[2]

            it = enumerate(intervals_parsed)
            if progress:
                it = enumerate(tqdm(intervals_parsed, desc=cell_type, position=position, leave=True))

            for i, (chrom, region_start, region_end) in it:
                pk_starts, pk_ends, h5_idxs = self._flanked_peaks(
                    cell_type, chrom, region_start, region_end
                )
                if h5_idxs.size == 0:
                    continue

                total = 0.0
                for pk_start, pk_end, h5_idx in zip(pk_starts, pk_ends, h5_idxs):
                    seq_start = (int(pk_start) + int(pk_end)) // 2 - seq_len // 2
                    src_start = max(0, region_start - seq_start)
                    src_end = min(seq_len, region_end - seq_start)
                    if src_end <= src_start:
                        continue
                    total += float(np.sum(ds[int(h5_idx)][:, src_start:src_end]))

                if agg == "sum":
                    results[i] = total
                else:
                    region_len = region_end - region_start
                    results[i] = total / (region_len * n_bases) if region_len else 0.0

        return results

    def shap_matrix(self, intervals, cell_types=None, agg="sum", n_jobs=1, progress=False):
        """Generate a (intervals × cell types) SHAP score matrix.

        Parameters
        ----------
        intervals : list[str]
            Genomic intervals, e.g. ``["chr1:1000-2000", "chr2:3000-4000"]``.
        cell_types : list[str], optional
            Subset of ``self.cell_types``.  Defaults to all.
        agg : {"sum", "mean"}
            How to collapse the ``(region_len, 4)`` SHAP array to a scalar.
        n_jobs : int
            Number of cell types to process in parallel (``-1`` uses all).
        progress : bool
            Show a tqdm progress bar over cell types.

        Returns
        -------
        pd.DataFrame of shape ``(n_intervals, n_cell_types)``.
        """
        if cell_types is None:
            cell_types = self.cell_types

        intervals_parsed = [self._parse_interval(iv) for iv in intervals]

        for ct in cell_types:
            self._build_index(ct)

        data = {}
        if n_jobs == 1:
            for i, ct in enumerate(cell_types):
                data[ct] = self._compute_ct_column(intervals_parsed, ct, agg, progress=progress, position=i)
        else:
            n_workers = len(cell_types) if n_jobs == -1 else n_jobs
            data = {}
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = {
                    ex.submit(self._compute_ct_column, intervals_parsed, ct, agg, progress, i): ct
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
            TrackSpec("shap", data=self.fetch(interval, ct), label=ct, **kwargs)
            for ct in cell_types
        ]


def aggregate_shap_by_group(matrix, annotations, group_col, agg="mean"):
    """Aggregate :meth:`Shap.shap_matrix` rows by group labels from *annotations*.

    Parameters
    ----------
    matrix : pd.DataFrame
        Output of :meth:`Shap.shap_matrix` — shape ``(n_intervals, n_cell_types)``.
    annotations : pd.DataFrame
        Indexed by interval strings and containing at least *group_col*.
    group_col : str
        Column in *annotations* defining the groups.
    agg : {"mean", "sum", "median", "max"}

    Returns
    -------
    pd.DataFrame of shape ``(n_groups, n_cell_types)``.
    """
    groups = annotations[group_col].reindex(matrix.index).dropna()
    return matrix.loc[groups.index].groupby(groups).agg(agg)


def filter_variable_intervals(matrix, method="var", threshold=None, top_n=None):
    """Filter motif intervals whose SHAP scores vary across cell types.

    Parameters
    ----------
    matrix : pd.DataFrame
        Output of :meth:`Shap.shap_matrix` — shape ``(n_intervals, n_cell_types)``.
    method : {"var", "std", "range"}
    threshold : float, optional
        Keep intervals exceeding this dispersion value.
    top_n : int, optional
        Keep the top *n* intervals by dispersion.

    Returns
    -------
    pd.DataFrame filtered and sorted by dispersion (descending).
    """
    if (threshold is None) == (top_n is None):
        raise ValueError("Provide exactly one of 'threshold' or 'top_n'.")

    _dispatchers = {
        "var":   lambda df: df.var(axis=1),
        "std":   lambda df: df.std(axis=1),
        "range": lambda df: df.max(axis=1) - df.min(axis=1),
    }
    if method not in _dispatchers:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(_dispatchers)}.")

    scores = _dispatchers[method](matrix)
    sorted_idx = scores.sort_values(ascending=False).index

    if top_n is not None:
        kept = sorted_idx[:top_n]
    else:
        kept = sorted_idx[scores[sorted_idx] > threshold]

    return matrix.loc[kept]

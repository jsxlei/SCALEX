"""Motif: load merged hit file, intersect with peaks, filter by pattern."""
import re

import pandas as pd

from ...atac.bedtools import intersect_bed, df_to_bed


class Motif:
    """Motif instances loaded from a merged hit file.

    Typical workflow::

        m = Motif("results/4_linkage/hits/hits_merged.tsv")  # from file
        m = Motif(hits_df)                                    # from DataFrame

        # Run bedtools once to build the motif↔peak mapping
        m._intersect_peaks(peaks_df)

        # Then filter peaks by any motif column — no bedtools re-run
        pos_peaks = m.filter_peaks(col="annotation", pattern=r"USF")
        sp1_peaks = m.filter_peaks(col="pattern", values=["pos_pattern_6"])
        foamy_peaks = m.filter_peaks(col="name", pattern="^Foamy")
        multi_peaks = m.filter_peaks(col=["annotation", "selin_match", "hocohomo_match"], pattern="USF")
    """

    def __init__(self, hit_file: str | pd.DataFrame) -> None:
        if isinstance(hit_file, pd.DataFrame):
            self.motifs = hit_file.copy()
        else:
            self.motifs = pd.read_csv(hit_file, sep="\t")
        self.motifs["interval"] = df_to_bed(self.motifs)
        self._peak_edges: pd.DataFrame | None = None  # (motif_pos, peak_pos) pairs
        self._peaks: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # One-time intersection
    # ------------------------------------------------------------------

    def _intersect_peaks(self, peaks: pd.DataFrame) -> None:
        """Precompute motif↔peak overlaps. Call once before filter_peaks.

        Stores a (motif_pos, peak_pos) edge table so subsequent calls to
        :meth:`filter_peaks` only do an in-memory lookup.

        Parameters
        ----------
        peaks:
            Peak set. Accepts ``chr/start/end`` or ``Chromosome/Start/End``
            column naming.
        """
        self._peaks = peaks
        self._peaks['peak_name'] = df_to_bed(peaks)
        if len(self.motifs) == 0:
            self._peak_edges = pd.DataFrame(columns=["motif_pos", "peak_pos"])
            return

        edges = intersect_bed(self.motifs, peaks, out="edge_index")
        self._peak_edges = pd.DataFrame({
            "motif_pos": edges[0],
            "peak_pos": edges[1],
        })

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_motifs(
        self,
        col: str | list[str],
        values: list | None = None,
        pattern: str | list[str] | None = None,
    ) -> "Motif":
        """Return a new Motif with hits matching criteria on *col*.

        Parameters
        ----------
        col:
            Column(s) to filter on. Pass a list to match across multiple
            columns with OR logic, e.g.
            ``["annotation", "selin_match", "hocohomo_match"]``.
        values:
            Exact values to keep; takes precedence over *pattern*.
        pattern:
            Regex to match (case-insensitive). Pass a list of strings to
            match any of them (OR logic), e.g. ``["USF", "MITF", "TFEB"]``.
        """
        cols = [col] if isinstance(col, str) else col
        missing = [c for c in cols if c not in self.motifs.columns]
        if missing:
            raise ValueError(f"Column(s) {missing} not found. Available: {list(self.motifs.columns)}")

        if isinstance(pattern, list):
            pattern = "|".join(pattern)

        def _col_mask(c: str) -> "pd.Series":
            col_s = self.motifs[c].astype(str)
            if values is not None:
                return col_s.isin([str(v) for v in values])
            if pattern is not None:
                return col_s.str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
            raise ValueError("Provide either 'values' or 'pattern'.")

        masks = [_col_mask(c) for c in cols]
        mask = masks[0]
        for m in masks[1:]:
            mask = mask | m

        new = object.__new__(Motif)
        new.motifs = self.motifs[mask].copy()
        new._peaks = self._peaks
        if self._peak_edges is not None:
            kept = new.motifs.index
            new._peak_edges = self._peak_edges[self._peak_edges["motif_pos"].isin(kept)].copy()
        else:
            new._peak_edges = None
        return new

    def filter_motif_by_similarity(self, cutoff: float) -> "Motif":
        """Return a new Motif keeping only hits with ``hit_similarity > cutoff``.

        Parameters
        ----------
        cutoff:
            Minimum (exclusive) similarity threshold.
        """
        mask = self.motifs["hit_similarity"] > cutoff
        new = object.__new__(Motif)
        new.motifs = self.motifs[mask].copy()
        new._peaks = self._peaks
        if self._peak_edges is not None:
            kept = new.motifs.index
            new._peak_edges = self._peak_edges[self._peak_edges["motif_pos"].isin(kept)].copy()
        else:
            new._peak_edges = None
        return new

    def filter_peaks(
        self,
        col: str | list[str] | None = None,
        values: list | None = None,
        pattern: str | list[str] | None = None,
        merge_motifs: bool = True,
        motif_col: str = "motif_name",
    ) -> pd.DataFrame:
        """Return peaks overlapping hits that match the given criteria.

        Requires :meth:`_intersect_peaks` to have been called first.

        Parameters
        ----------
        col:
            Column(s) in ``motifs`` to filter on. If None, all hits are used.
        values:
            Passed to :meth:`filter_motifs`.
        pattern:
            Passed to :meth:`filter_motifs`.
        merge_motifs:
            If True, append a *motif_col* column to each peak row containing
            the sorted unique motif names that overlap it.
        motif_col:
            Column in ``motifs`` to aggregate when *merge_motifs* is True.

        Returns
        -------
        pd.DataFrame
            Subset of the peaks passed to :meth:`_intersect_peaks` that
            contain at least one matching motif hit. When *merge_motifs* is
            True, each row also has a *motif_col* list column.
        """
        if self._peak_edges is None:
            raise RuntimeError("Call _intersect_peaks(peaks) first.")

        if col is None:
            motif_sub = self
            matching = self._peak_edges
        else:
            motif_sub = self.filter_motifs(col=col, values=values, pattern=pattern)
            motif_positions = motif_sub.motifs.index
            matching = self._peak_edges[self._peak_edges["motif_pos"].isin(motif_positions)]

        if len(matching) == 0:
            result = self._peaks.iloc[:0].copy()
            if merge_motifs:
                result[motif_col] = pd.Series(dtype=object)
            return result

        if not merge_motifs:
            return self._peaks.iloc[matching["peak_pos"].unique()]

        agg = (
            pd.DataFrame({
                "peak_pos": matching["peak_pos"].values,
                motif_col: motif_sub.motifs.loc[matching["motif_pos"].values, motif_col].values,
            })
            .groupby("peak_pos")[motif_col]
            .apply(lambda x: ";".join(sorted(set(x.tolist()))))
            .reset_index()
        )
        result = self._peaks.iloc[agg["peak_pos"].values].reset_index(drop=True).copy()
        result[motif_col] = agg[motif_col].values
        return result

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.motifs)

    def __repr__(self) -> str:
        n = self.motifs["motif_name"].nunique() if "motif_name" in self.motifs.columns else "?"
        return f"Motif({len(self.motifs):,} hits, {n} unique motif_name values)"

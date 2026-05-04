"""Cell-type registry and scE2G prediction loading / intersection."""
import argparse
import logging
import os

import anndata
import numpy as np
import pandas as pd
import scipy.sparse
from pybedtools import BedTool
from scipy.stats import pearsonr
from scalex.data import aggregate_data


def sce2g_predictions_to_loop_df(
    file_path: str, 
    exclude_self_loop: bool = True,
    score_col: str = "E2G.Score.qnorm",
) -> pd.DataFrame:
    """Read scE2G predictions and convert to loop-like DataFrame.

    Output columns:
    - ``Chromosome``: copied from input ``chr``
    - ``Start``: midpoint of input ``start`` and ``end``
    - ``End``: input ``TargetGeneTSS``
    - ``score_col``: default ``E2G.Score.qnorm``

    If ``exclude_self_loop`` is True, rows with ``isSelfPromoter == True``
    are removed.
    """
    df = pd.read_csv(file_path, sep="\t", compression="gzip", low_memory=False)

    required = ["chr", "start", "end", "TargetGeneTSS", score_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {file_path}: {', '.join(missing)}"
        )

    out = df.copy()
    out["Chromosome"] = df["chr"].astype(str)
    out["Start"] = ((pd.to_numeric(df["start"], errors="coerce")
                     + pd.to_numeric(df["end"], errors="coerce")) // 2)
    out["End"] = pd.to_numeric(df["TargetGeneTSS"], errors="coerce")
    out[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    out = out.dropna(subset=["Start", "End", score_col])
    out["Start"] = out["Start"].astype(int)
    out["End"] = out["End"].astype(int)

    if exclude_self_loop:
        if "isSelfPromoter" not in df.columns:
            raise ValueError(
                "exclude_self_loop=True requires 'isSelfPromoter' column in input file."
            )
        prom = df.loc[out.index, "isSelfPromoter"]
        if pd.api.types.is_bool_dtype(prom):
            is_self_promoter = prom.fillna(False)
        else:
            is_self_promoter = (
                prom.astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "t", "yes", "y"})
            )
        out = out[~is_self_promoter.values]

    return out.reset_index(drop=True)


def load_scE2G_loops(scE2G_DIR):
    import glob

    pattern = os.path.join(scE2G_DIR, "*/multiome_powerlaw_v3", "scE2G_predictions_threshold*.tsv.gz")
    loops = {
        os.path.basename(os.path.dirname(os.path.dirname(p))): sce2g_predictions_to_loop_df(p)
        for p in sorted(glob.glob(pattern))
    }

    return loops


def merge_loops(
    loops: dict,
    score_col: str = "E2G.Score.qnorm",
) -> pd.DataFrame:
    """Merge per-cell-type loop DataFrames, grouping identical peak–TSS pairs.

    Parameters
    ----------
    loops:
        Dict mapping cell-type name to a loop DataFrame as returned by
        :func:`load_scE2G_loops`.
    score_col:
        Name of the score column present in each per-cell-type DataFrame.

    Returns
    -------
    DataFrame with columns ``Chromosome``, ``Start``, ``End``, ``CellType``
    (list of cell types sharing this peak–TSS pair), and ``score_col``
    (max score across cell types).
    """
    frames = []
    for cell_type, df in loops.items():
        frame = df.copy()
        frame["CellType"] = cell_type
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    key = ["Chromosome", "Start", "End"]
    other_cols = [c for c in combined.columns if c not in key + ["CellType", score_col]]
    agg = {c: (c, "first") for c in other_cols}
    agg["CellType"] = ("CellType", lambda x: ";".join(sorted(x.tolist())))
    agg[score_col] = (score_col, "max")

    return combined.groupby(key, sort=False).agg(**agg).reset_index()


def _parse_peaks(peaks) -> pd.DataFrame:
    """Coerce peaks to a DataFrame with Chromosome, Start, End columns.

    Accepts:
    - list/Series of ``chr:start-end`` strings
    - BED file path (tab-separated, no header)
    - DataFrame with ``Chromosome``/``Start``/``End`` or ``chr``/``start``/``end``
    """
    if isinstance(peaks, dict):
        peaks = list(dict.fromkeys(p for group in peaks.values() for p in group))

    if isinstance(peaks, (list, pd.Series, np.ndarray)):
        records = []
        for p in peaks:
            chrom, coords = str(p).split(":")
            start, end = coords.split("-")
            records.append((chrom.strip(), int(start.strip()), int(end.strip())))
        return pd.DataFrame(records, columns=["Chromosome", "Start", "End"])

    if isinstance(peaks, str):
        df = pd.read_csv(peaks, sep="\t", header=None, comment="#")
        df.columns = ["Chromosome", "Start", "End"] + list(df.columns[3:])
        return df

    df = peaks.copy()
    if "chr" in df.columns and "Chromosome" not in df.columns:
        df = df.rename(columns={"chr": "Chromosome", "start": "Start", "end": "End"})
    return df


def link_peaks_to_genes(
    peaks,
    merged_loops,
    gene_col: str = "TargetGene",
    score_col: str = "E2G.Score.qnorm",
    score_threshold: float | None = None,
    return_dict: bool = False,
) -> "pd.DataFrame | dict":
    """Find all scE2G-mediated peak→gene links by overlapping peaks with enhancers.

    Each (peak, gene) pair is a separate row in the returned DataFrame.
    This makes filtering by gene set or peak set straightforward.

    Parameters
    ----------
    peaks:
        list/Series/ndarray of ``chr:start-end`` strings, dict mapping group
        names to lists of peak strings, BED file path, or DataFrame with
        ``Chromosome``/``Start``/``End`` columns.
    merged_loops:
        scE2G merged loops — file path (.tsv/.tsv.gz) or DataFrame from
        :func:`merge_loops`.
    gene_col:
        Column in merged_loops with target gene names (default: ``TargetGene``).
    score_col:
        Score column (default: ``E2G.Score.qnorm``).
    score_threshold:
        If set, only links with ``score_col >= score_threshold`` are used.
    return_dict:
        If True, return ``{group: list[gene]}`` instead of a DataFrame.
        Requires ``peaks`` to be a dict.

    Returns
    -------
    DataFrame with columns: ``peak``, ``TargetGene``, ``distance``,
    ``score_col``, ``CellType`` (if present in loops).
    One row per unique (peak, gene) pair. Distance is the absolute distance
    from the peak center to the **enhancer center**. Peaks not overlapping any
    enhancer are excluded.
    If ``return_dict=True``, returns ``{group: sorted list[gene]}``.
    """
    group_map = None
    if isinstance(peaks, dict):
        if return_dict:
            group_map = {p: g for g, ps in peaks.items() for p in ps}
        peaks_list = list(dict.fromkeys(p for ps in peaks.values() for p in ps))
    elif isinstance(peaks, (list, pd.Series, np.ndarray)):
        peaks_list = list(peaks)
    else:
        peaks_list = None

    if return_dict and group_map is None:
        raise ValueError("return_dict=True requires peaks to be a dict")

    peaks_df = _parse_peaks(peaks)
    peaks_df = peaks_df.reset_index(drop=True)
    peaks_df["_peak_idx"] = peaks_df.index
    if peaks_list is not None:
        peaks_df["peak"] = peaks_list

    if isinstance(merged_loops, str):
        compression = "gzip" if merged_loops.endswith(".gz") else None
        loops_df = pd.read_csv(merged_loops, sep="\t", compression=compression)
    else:
        loops_df = merged_loops.copy()

    if score_threshold is not None:
        loops_df = loops_df[loops_df[score_col] >= score_threshold]

    if gene_col not in loops_df.columns:
        raise ValueError(f"Gene column '{gene_col}' not found in merged loops.")

    # Use original enhancer coordinates (lowercase) when available
    if "chr" in loops_df.columns:
        enh_cols = ["chr", "start", "end"]
    elif "start" in loops_df.columns and "end" in loops_df.columns:
        enh_cols = ["Chromosome", "start", "end"]
    else:
        enh_cols = ["Chromosome", "Start", "End"]
    loops_df = loops_df.reset_index(drop=True)
    loops_df["_loop_idx"] = loops_df.index

    enh_bed = BedTool.from_dataframe(
        loops_df[enh_cols + ["_loop_idx"]].rename(columns={"chr": "Chromosome", "start": "Start", "end": "End"})
    )
    peaks_bed = BedTool.from_dataframe(
        peaks_df[["Chromosome", "Start", "End", "_peak_idx"]]
    )

    hit = enh_bed.intersect(peaks_bed, wa=True, wb=True).to_dataframe(
        names=["e_chr", "e_start", "e_end", "_loop_idx",
               "p_chr", "p_start", "p_end", "_peak_idx"]
    )

    if hit.empty:
        cols = ["peak", gene_col, "distance", score_col]
        if "CellType" in loops_df.columns:
            cols.append("CellType")
        return pd.DataFrame(columns=cols)

    link_cols = ["_loop_idx", gene_col, score_col]
    if "CellType" in loops_df.columns:
        link_cols.append("CellType")
    hit = hit.merge(loops_df[link_cols], on="_loop_idx")

    hit["peak_mid"] = (hit["p_start"] + hit["p_end"]) // 2
    hit["enh_mid"]  = (hit["e_start"] + hit["e_end"]) // 2
    hit["distance"] = (hit["peak_mid"] - hit["enh_mid"]).abs()

    join_cols = ["_peak_idx", "peak"] if "peak" in peaks_df.columns else ["_peak_idx"]
    hit = hit.merge(peaks_df[join_cols], on="_peak_idx")

    keep_cols = ["peak", gene_col, "distance", score_col]
    if "CellType" in hit.columns:
        keep_cols.append("CellType")

    if not return_dict:
        return hit[keep_cols].drop_duplicates().reset_index(drop=True)

    df = hit[keep_cols].drop_duplicates()
    gene_dict = {}
    for group, ps in peaks.items():
        group_peaks = set(ps)
        genes = df[df["peak"].isin(group_peaks)][gene_col].dropna().unique().tolist()
        if genes:
            gene_dict[group] = sorted(genes)
    return gene_dict


def filter_linked_markers(
    peak_dict: dict,
    gene_dict: dict,
    merged_loops,
    gene_col: str = "TargetGene",
    score_col: str = "E2G.Score.qnorm",
    score_threshold: float | None = None,
) -> tuple:
    """Filter marker peaks and marker genes to only those mutually linked via scE2G.

    Uses scE2G as a bridge between independently computed ATAC peak programs
    and RNA DEG programs. Filtering is **group-aware**: for each group shared
    between ``peak_dict`` and ``gene_dict``, only peaks and genes that are
    mutually linked *within that same group* are retained.

    - A peak is kept if it overlaps an scE2G enhancer that targets a gene in
      the **same group** of ``gene_dict``.
    - A gene is kept if it is targeted by an scE2G enhancer that overlaps a
      peak in the **same group** of ``peak_dict``.

    Parameters
    ----------
    peak_dict : dict {group: list[str]}
        Marker peaks grouped by cluster (e.g. from ``find_peak_program``).
        Values are lists of ``chr:start-end`` peak name strings.
    gene_dict : dict {group: list[str]}
        Marker genes grouped by cell type (e.g. from ``find_gene_program``).
    merged_loops : str or pd.DataFrame
        scE2G merged loops — file path or DataFrame from :func:`merge_loops`.
    gene_col : str
        Column in merged_loops with target gene names.
    score_col : str
        Score column for optional thresholding.
    score_threshold : float or None
        If set, only links with score >= threshold are used.

    Returns
    -------
    filtered_peak_dict : dict {group: list[str]}
        Subset of ``peak_dict``: only peaks with an scE2G link to a marker gene
        in the same group.
    filtered_gene_dict : dict {group: list[str]}
        Subset of ``gene_dict``: only genes with an scE2G link from a marker
        peak in the same group.
    link_dict : dict {group: pd.DataFrame}
        Per-group (peak, gene) link tables from :func:`link_peaks_to_genes`.
        Contains only groups with at least one within-group link.
    """
    # Load and optionally threshold loops once for efficiency
    if isinstance(merged_loops, str):
        compression = "gzip" if merged_loops.endswith(".gz") else None
        loops_df = pd.read_csv(merged_loops, sep="\t", compression=compression)
    else:
        loops_df = merged_loops.copy()

    if score_threshold is not None:
        loops_df = loops_df[loops_df[score_col] >= score_threshold]

    # 1. Match keys: only groups present in both dicts
    groups = sorted(set(peak_dict) & set(gene_dict))

    # 2. Collect unique peaks across all matched groups, then find links in one pass
    all_peaks = list(dict.fromkeys(p for g in groups for p in peak_dict[g]))
    all_links = link_peaks_to_genes(
        peaks=all_peaks,
        merged_loops=loops_df,
        gene_col=gene_col,
        score_col=score_col,
        score_threshold=None,
    )

    # 3. For each group, keep only "matched" links: peak ∈ group AND gene ∈ group
    filtered_peak_dict, filtered_gene_dict, link_dict = {}, {}, {}

    for group in groups:
        group_peaks = set(peak_dict[group])
        group_genes = set(gene_dict[group])

        if not group_peaks or not group_genes:
            continue

        links = all_links[
            all_links["peak"].isin(group_peaks) & all_links[gene_col].isin(group_genes)
        ].reset_index(drop=True)

        if links.empty:
            continue

        linked_peaks = set(links["peak"])
        linked_genes = set(links[gene_col])

        filtered_peak_dict[group] = [p for p in peak_dict[group] if p in linked_peaks]
        filtered_gene_dict[group] = [g for g in gene_dict[group] if g in linked_genes]
        link_dict[group] = links

    return filtered_peak_dict, filtered_gene_dict, link_dict



def _pearson_pairs(pairs: pd.DataFrame, mat_x: pd.DataFrame, mat_y: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson r/pval for each (peak_id, TargetGene) pair.

    Parameters
    ----------
    pairs:
        DataFrame with columns ``peak_id``, ``TargetGene``, ``atac_var_name``
        (the ATAC var_name to look up in mat_x for each scE2G peak_id).
    mat_x:
        DataFrame (samples × atac_var_names) — accessibility profiles.
    mat_y:
        DataFrame (samples × gene_names) — expression profiles.

    Returns
    -------
    DataFrame with columns ``peak_id``, ``TargetGene``, ``pearson_r``, ``pearson_pval``.
    """
    results = []
    for _, row in pairs.iterrows():
        peak_id = row["peak_id"]
        gene = row["TargetGene"]
        atac_var = row["atac_var_name"]
        r, pval = np.nan, np.nan
        if (
            not pd.isna(atac_var)
            and gene in mat_y.columns
            and atac_var in mat_x.columns
        ):
            x = mat_x[atac_var].values.astype(float)
            y = mat_y[gene].values.astype(float)
            if np.std(x) > 0 and np.std(y) > 0:
                r, pval = pearsonr(x, y)
        results.append({"peak_id": peak_id, "TargetGene": gene,
                        "pearson_r": r, "pearson_pval": pval})
    return pd.DataFrame(results)


def add_pearson_to_links(
    links: pd.DataFrame,
    rna: anndata.AnnData,
    atac: anndata.AnnData,
    cell_type_col: str = "cell_type",
    mode: str = "pseudobulk",
    peak_col: str = "peak_id",
    gene_col: str = "TargetGene",
) -> pd.DataFrame:
    """Add pearson_r and pearson_pval columns to a scE2G links DataFrame.

    Parameters
    ----------
    links:
        DataFrame with ``peak_col`` (chr:start-end) and ``gene_col`` columns.
        If ``peak_col`` is absent it is built from ``chr``, ``start``, ``end``.
    rna / atac:
        AnnData objects. For pseudobulk mode both must have ``cell_type_col``
        in ``.obs``; for singlecell mode ``obs_names`` must be matching barcodes.
    mode:
        ``"pseudobulk"`` (default) or ``"singlecell"``.

    Returns
    -------
    ``links`` with two new columns merged on (``peak_col``, ``gene_col``):
    ``pearson_r`` and ``pearson_pval``. Pairs with no overlapping ATAC peak or
    absent gene receive NaN.
    """
    links = links.copy()

    # 1. Build peak_id if absent
    if peak_col not in links.columns:
        links[peak_col] = (
            links["chr"].astype(str) + ":"
            + links["start"].astype(str) + "-"
            + links["end"].astype(str)
        )

    # 2. BED intersect scE2G peaks → ATAC var_names
    unique_peaks = links[["chr", "start", "end", peak_col]].drop_duplicates(subset=[peak_col])

    atac_peaks_df = _parse_peaks(list(atac.var_names)).reset_index(drop=True)
    atac_peaks_df["atac_var_name"] = list(atac.var_names)
    atac_peaks_df["_atac_idx"] = atac_peaks_df.index

    sce2g_bed_df = pd.DataFrame({
        "Chromosome": unique_peaks["chr"].astype(str).values,
        "Start": pd.to_numeric(unique_peaks["start"], errors="coerce").astype(int).values,
        "End": pd.to_numeric(unique_peaks["end"], errors="coerce").astype(int).values,
        "_peak_id": unique_peaks[peak_col].values,
    })
    sce2g_bed = BedTool.from_dataframe(sce2g_bed_df)
    atac_bed = BedTool.from_dataframe(atac_peaks_df[["Chromosome", "Start", "End", "_atac_idx"]])

    hit = sce2g_bed.intersect(atac_bed, wa=True, wb=True).to_dataframe(
        names=["s_chr", "s_start", "s_end", "_peak_id",
               "a_chr", "a_start", "a_end", "_atac_idx"]
    )

    if hit.empty:
        links["pearson_r"] = np.nan
        links["pearson_pval"] = np.nan
        return links

    hit = hit.merge(atac_peaks_df[["_atac_idx", "atac_var_name"]], on="_atac_idx")
    peak_to_atac = hit.groupby("_peak_id")["atac_var_name"].first().to_dict()

    # 3. Subset ATAC to matched peaks (memory saving)
    matched_atac_vars = [v for v in set(peak_to_atac.values()) if v in atac.var_names]
    atac_sub = atac[:, matched_atac_vars]

    # 4. Build unique pairs with atac_var_name for lookup
    unique_pairs = (
        links[[peak_col, gene_col]]
        .drop_duplicates()
        .rename(columns={peak_col: "peak_id", gene_col: "TargetGene"})
        .copy()
    )
    unique_pairs["atac_var_name"] = unique_pairs["peak_id"].map(peak_to_atac)

    # 5. Build expression / accessibility matrices
    if mode == "pseudobulk":
        agg_rna = aggregate_data(rna, groupby=cell_type_col)
        agg_atac = aggregate_data(atac_sub, groupby=cell_type_col)
        shared = sorted(set(agg_rna.obs_names) & set(agg_atac.obs_names))
        if len(shared) < 2:
            raise ValueError(f"Need ≥2 shared cell types for correlation; got {shared}")
        rna_agg = agg_rna[shared]
        atac_agg = agg_atac[shared]
        X_rna = rna_agg.X.toarray() if scipy.sparse.issparse(rna_agg.X) else np.array(rna_agg.X)
        X_atac = atac_agg.X.toarray() if scipy.sparse.issparse(atac_agg.X) else np.array(atac_agg.X)
        mat_y = pd.DataFrame(X_rna, index=shared, columns=rna_agg.var_names)
        mat_x = pd.DataFrame(X_atac, index=shared, columns=atac_agg.var_names)
    else:  # singlecell
        shared = sorted(set(rna.obs_names) & set(atac_sub.obs_names))
        if len(shared) < 2:
            raise ValueError(f"Need ≥2 shared barcodes; found {len(shared)}")
        rna_sc = rna[shared]
        atac_sc = atac_sub[shared]
        X_rna = rna_sc.X.toarray() if scipy.sparse.issparse(rna_sc.X) else np.array(rna_sc.X)
        X_atac = atac_sc.X.toarray() if scipy.sparse.issparse(atac_sc.X) else np.array(atac_sc.X)
        mat_y = pd.DataFrame(X_rna, index=shared, columns=rna_sc.var_names)
        mat_x = pd.DataFrame(X_atac, index=shared, columns=atac_sc.var_names)

    # 6. Compute correlations and merge back
    corr_df = _pearson_pairs(unique_pairs, mat_x, mat_y)
    corr_df = corr_df.rename(columns={"peak_id": peak_col, "TargetGene": gene_col})
    return links.merge(
        corr_df[[peak_col, gene_col, "pearson_r", "pearson_pval"]],
        on=[peak_col, gene_col],
        how="left",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Merge scE2G prediction links across cell types."
    )
    parser.add_argument(
        "scE2G_dir",
        help="Directory containing per-cell-type scE2G prediction subdirectories.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file path (e.g. merged_loops.tsv or merged_loops.tsv.gz).",
    )
    parser.add_argument(
        "--score-col",
        default="E2G.Score.qnorm",
        help="Score column to use (default: E2G.Score.qnorm).",
    )
    parser.add_argument(
        "--keep-self-loops",
        action="store_true",
        help="Keep self-promoter loops (excluded by default).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logging.info("Loading scE2G loops from %s", args.scE2G_dir)
    loops = load_scE2G_loops(args.scE2G_dir)
    logging.info("Loaded %d cell types: %s", len(loops), ", ".join(sorted(loops)))

    merged = merge_loops(loops, score_col=args.score_col)
    logging.info("Merged into %d peak–TSS links", len(merged))

    compression = "gzip" if args.output.endswith(".gz") else None
    merged.to_csv(args.output, sep="\t", index=False, compression=compression)
    logging.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()


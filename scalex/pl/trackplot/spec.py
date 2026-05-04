"""Composable track plotting for multi-omics genomic visualization.

Basic usage::

    from scalex.pl.trackplot import TrackSpec, compose_tracks

    tracks = [
        TrackSpec("scalebar"),
        TrackSpec("coverage", data=frag_file, cell_type="Foamy_SPP1_high", label="SPP1+"),
        TrackSpec("coverage", data=frag_file, cell_type="Foamy_SPP1_low",  label="SPP1-"),
        TrackSpec("annotation", data=gwas, label="GWAS CAD",
                  params={"score_column": "-logPval", "threshold": 10}),
        TrackSpec("loop", data=loop_, label="scE2G",
                  params={"score_column": "ENCODE-rE2G.Score"}),
        TrackSpec("gene", data=transcripts, label="Genes"),
    ]

    fig = compose_tracks(
        tracks, region="chr3:69729463-69978332",
        cell_groups=obs["cell_type"], cell_counts=obs["nCount_ATAC"],
        save="figure.png",
    )

To register a new track type::

    from scalex.pl.trackplot import register_track, TrackSpec, TrackContext

    @register_track("my_track")
    def _build_my_track(spec: TrackSpec, ax, ctx: TrackContext) -> None:
        # render into ax using spec.data, spec.params, ctx.*
        ...

BigWig track usage::

    tracks = [
        TrackSpec("bigwig", data="h3k27ac.bw", label="H3K27ac"),
        TrackSpec("bigwig", data="atac.bw", label="ATAC (RPM)",
                  params={"normalize_to_1m": True, "total_reads": 45_000_000,
                          "color": "darkorange"}),
    ]
    fig = compose_tracks(tracks, region="chr3:69729463-69978332")

SHAP track usage::

    from scalex.pl.trackplot import flank_peaks, assemble_shap_for_region, TrackSpec, compose_tracks

    # Prepare data (no file I/O — pass pre-loaded arrays)
    peaks_flanked = flank_peaks(peaks_df, seq_len=shap_values.shape[2])
    shap_foamy  = assemble_shap_for_region(shap_foamy_arr,  peaks_flanked, "chr1", 1000000, 1100000)
    shap_resident = assemble_shap_for_region(shap_resident_arr, peaks_flanked, "chr1", 1000000, 1100000)

    tracks = [
        TrackSpec("shap", data=shap_foamy,    label="Foamy"),
        TrackSpec("shap", data=shap_resident, label="Resident"),
    ]
    fig = compose_tracks(tracks, region="chr1:1000000-1100000")
"""
## Adapt from BPCell
import dataclasses
import re
from typing import Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pysam
import logomaker
from matplotlib.ticker import FuncFormatter


def trackplot_calculate_segment_height(data):
    """Calculate y positions for trackplot segments to avoid overlap"""
    if len(data) == 0:
        return []

    data = data.copy()
    data['row_number'] = range(len(data))

    # Sort by start position for efficient overlap checking
    data = data.sort_values('Start').reset_index(drop=True)

    # Initialize y positions
    y_pos = [0] * len(data)
    used_positions = []  # Track which y positions are occupied

    for i, row in data.iterrows():
        current_start = row['Start']
        current_end = row['End']

        # Check if this gene overlaps with any existing genes at each y position
        best_y = 0
        for y in range(len(used_positions) + 1):
            overlap_found = False

            # Check overlap with genes at this y position
            for j, (y_pos_j, start_j, end_j) in enumerate(used_positions):
                if y == y_pos_j:
                    # Check if current gene overlaps with gene at this y position
                    if not (current_end <= start_j or current_start >= end_j):
                        overlap_found = True
                        break

            if not overlap_found:
                best_y = y
                break

        # Assign y position
        y_pos[i] = best_y

        # Add to used positions
        used_positions.append((best_y, current_start, current_end))

    return y_pos


# ============================================================
# Composable track API
# ============================================================

@dataclasses.dataclass
class TrackSpec:
    """Specification for a single track in a composite plot.

    Parameters
    ----------
    track_type : str
        One of "coverage", "gene", "annotation", "loop", "scalebar",
        or any type registered via ``@register_track``.
    data : Any, optional
        Fragment file path (for coverage) or a DataFrame.
    cell_type : str, optional
        Group name to display (coverage tracks only).
    label : str, optional
        Y-axis label override.
    height : float, optional
        Relative panel height; defaults to ``_DEFAULT_HEIGHTS[track_type]``.
    style : dict, optional
        Rendering overrides (e.g. ``{"font_pt": 9, "exon_size": 3}``).
    params : dict
        Track-type-specific keyword arguments passed to the builder.
    """
    track_type: str
    data: Any = None
    cell_type: str = None
    label: str = None
    height: float = None
    style: dict = None
    params: dict = dataclasses.field(default_factory=dict)
    cell_groups: pd.Series = None   # barcode → group mapping (overrides global)
    cell_counts: pd.Series = None   # barcode → read count (overrides global)
    colors: dict = None             # group → color (overrides global)
    cell_order: list = None         # ordered subset of cell types to display; None = all in .unique() order
    gene: str = None                # gene filter for sce2g tracks (TargetGene must match)
    frameon: bool = False           # show axis frame; False hides all spines
    expr: float = None              # gene expression value; dict {cell_type: value} for list expansion
    highlight_loop: Any = None      # DataFrame/PyRanges with Chromosome,Start,End — arcs to highlight


@dataclasses.dataclass
class TrackContext:
    """Shared context passed to every track builder.

    Parameters
    ----------
    chrom, start, end : str / int
        Parsed genomic region coordinates.
    cell_groups : pd.Series
        Barcode → group name mapping.
    cell_counts : pd.Series
        Barcode → read count mapping (for RPKM normalisation).
    colors : dict
        Group → matplotlib color mapping.
    bins : int
        Number of genomic bins for coverage histograms.
    clip_quantile : float
        Upper quantile for clipping coverage extremes.
    coverage_cache : dict
        Populated by ``_precompute_coverage``; maps frag_file path →
        DataFrame(pos, group, normalized_insertions).
    global_coverage_max : float
        Shared y-axis ceiling across all coverage tracks.
    bigwig_cache : dict
        Populated by ``_precompute_bigwig``; maps bw_file path →
        dict(positions, values, label).
    global_bigwig_max : float
        Shared y-axis ceiling across all bigwig tracks.
    """
    chrom: str
    start: int
    end: int
    cell_groups: pd.Series = None
    cell_counts: pd.Series = None
    colors: dict = None
    bins: int = 1000
    clip_quantile: float = 0.999
    coverage_cache: dict = dataclasses.field(default_factory=dict)
    global_coverage_max: float = None
    global_shap_ylim: tuple = None  # shared (ymin, ymax) across all shap tracks
    bigwig_cache: dict = dataclasses.field(default_factory=dict)
    global_bigwig_max: float = None
    sce2g_cache: dict = dataclasses.field(default_factory=dict)
    global_sce2g_score_range: tuple = None  # (min_score, max_score) across all cell types
    scglue_cache: dict = dataclasses.field(default_factory=dict)
    global_scglue_score_range: tuple = None  # (min_score, max_score) across all scglue links
    global_expr_max: float = None
    expr_axes_map: dict = dataclasses.field(default_factory=dict)  # id(spec) -> expr_ax


# Registry of builder functions: track_type → callable(spec, ax, ctx)
TRACK_BUILDERS: dict = {}


def register_track(name: str) -> Callable:
    """Decorator to register a track builder under *name*."""
    def decorator(fn: Callable) -> Callable:
        TRACK_BUILDERS[name] = fn
        return fn
    return decorator


_DEFAULT_HEIGHTS: dict = {
    "scalebar":   0.1,
    "coverage":   0.5,
    "gene":       1.0,
    "annotation": 1.5,
    "loop":       .8,
    "shap":       1.0,
    "bigwig":     .5,
    "sce2g":      .8,
    "scglue":     .8,
}


def _sort_tracks(tracks: list, sort_by: list) -> list:
    """Return *tracks* sorted by any combination of 'cell_type' / 'track_type'."""
    key_fns = {
        "cell_type":  lambda t: t.cell_type or "",
        "track_type": lambda t: t.track_type,
    }
    return sorted(tracks, key=lambda t: tuple(key_fns[k](t) for k in sort_by))


def _precompute_coverage(tracks: list, ctx: TrackContext) -> None:
    """Scan each unique fragment file once and populate ``ctx.coverage_cache``.

    Groups all coverage ``TrackSpec`` objects by ``(frag_file, id(cell_groups))``
    so the same file with different groupings is read independently. Sets
    ``ctx.global_coverage_max`` across all files.
    """
    coverage_specs = [t for t in tracks if t.track_type == "coverage"]
    if not coverage_specs:
        return

    # Collect which groups are needed per (frag_file, cell_groups identity)
    key_to_info: dict = {}
    for spec in coverage_specs:
        eff_cg = spec.cell_groups if spec.cell_groups is not None else ctx.cell_groups
        eff_cc = spec.cell_counts if spec.cell_counts is not None else ctx.cell_counts
        cache_key = (spec.data, id(eff_cg))
        if cache_key not in key_to_info:
            key_to_info[cache_key] = {
                "frag_file": spec.data,
                "cg": eff_cg,
                "cc": eff_cc,
                "requested": set(),
            }
        if spec.cell_type is not None:
            key_to_info[cache_key]["requested"].add(spec.cell_type)

    global_max = 0.0
    for cache_key, info in key_to_info.items():
        if cache_key in ctx.coverage_cache:
            local_max = ctx.coverage_cache[cache_key]["normalized_insertions"].max()
            global_max = max(global_max, local_max)
            continue

        eff_cg = info["cg"]
        eff_cc = info["cc"]
        frag_file = info["frag_file"]
        requested_groups = info["requested"]

        if not requested_groups:
            if eff_cg is not None:
                requested_groups = set(
                    eff_cg.cat.categories if pd.api.types.is_categorical_dtype(eff_cg)
                    else eff_cg.unique()
                )
        if not requested_groups or eff_cg is None:
            continue

        bin_size = max((ctx.end - ctx.start) // ctx.bins, 1)
        bin_edges = np.arange(ctx.start, ctx.end, bin_size)
        bin_centers = bin_edges[:-1] + (bin_size // 2)

        coverage = {g: np.zeros(len(bin_centers), dtype=int) for g in requested_groups}

        with pysam.TabixFile(frag_file) as tbx:
            for entry in tbx.fetch(ctx.chrom, ctx.start, ctx.end):
                parts = entry.split("\t")
                s, e, bc = int(parts[1]), int(parts[2]), parts[3]
                if bc not in eff_cg.index:
                    continue
                group = eff_cg.loc[bc]
                if group not in coverage:
                    continue
                idx_start = max(0, (s - ctx.start) // bin_size)
                idx_end = min(len(bin_centers) - 1, (e - ctx.start) // bin_size)
                coverage[group][idx_start:idx_end + 1] += 1

        group_read_counts = eff_cc.groupby(eff_cg).sum()
        norm_factors = 1e9 / (group_read_counts * bin_size)

        records = []
        for g, cov in coverage.items():
            norm_cov = cov * norm_factors[g]
            for pos, val in zip(bin_centers, norm_cov):
                records.append((pos, g, val))

        data = pd.DataFrame(records, columns=["pos", "group", "normalized_insertions"])
        ymax = data["normalized_insertions"].quantile(ctx.clip_quantile)
        data["normalized_insertions"] = np.minimum(data["normalized_insertions"], ymax)

        ctx.coverage_cache[cache_key] = data
        local_max = data["normalized_insertions"].max()
        global_max = max(global_max, local_max)

    if ctx.global_coverage_max is None:
        ctx.global_coverage_max = global_max


# ------------------------------------------------------------------
# Builder functions
# ------------------------------------------------------------------

@register_track("scalebar")
def _build_scalebar(spec: TrackSpec, ax: plt.Axes, ctx: TrackContext) -> None:
    """Render region label and scale bar into *ax*."""
    chrom, start, end = ctx.chrom, ctx.start, ctx.end
    width = end - start
    bar_width = _nice_scale(width * 0.1)
    font_pt = (spec.style or {}).get("font_pt", 11)

    if bar_width >= 1_000_000:
        bar_label = f"{bar_width / 1_000_000:.0f}M"
    elif bar_width >= 1_000:
        bar_label = f"{bar_width / 1_000:.0f}k"
    else:
        bar_label = f"{bar_width:.0f}"

    ax.text(start, 0, f"{chrom}: {start:,} - {end:,}", fontsize=font_pt, ha="left", va="center")

    bar_start = end - bar_width - width * 0.05
    bar_end = end - width * 0.05

    ax.plot([bar_start, bar_end], [0, 0], "k-", linewidth=1)
    ax.plot([bar_start, bar_start], [-0.05, 0.05], "k-", linewidth=1)
    ax.plot([bar_end, bar_end], [-0.05, 0.05], "k-", linewidth=1)

    ax.text(bar_start - width * 0.01, 0, bar_label,
            fontsize=font_pt, ha="right", va="center")

    ax.set_ylim(-0.2, 0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)


@register_track("coverage")
def _build_coverage(spec: TrackSpec, ax: plt.Axes, ctx: TrackContext) -> None:
    """Render a single-group coverage track into *ax* using the precomputed cache."""
    eff_cg = spec.cell_groups if spec.cell_groups is not None else ctx.cell_groups
    cache_key = (spec.data, id(eff_cg))
    data = ctx.coverage_cache.get(cache_key)
    if data is None:
        ax.set_yticks([])
        return

    cell_type = spec.cell_type
    group_data = data[data["group"] == cell_type]

    eff_colors = spec.colors if spec.colors is not None else ctx.colors
    color = eff_colors.get(cell_type, "steelblue") if eff_colors else "steelblue"

    if not group_data.empty:
        ax.fill_between(group_data["pos"], group_data["normalized_insertions"],
                        color=color, linewidth=0)

    ymax = ctx.global_coverage_max or data["normalized_insertions"].max()
    ax.set_ylim(0, ymax * 1.1)

    label = spec.label or cell_type or ""
    ax.text(-0.005, 0.0, label, transform=ax.transAxes, va="bottom", ha="right", fontsize=9)

    if spec.expr is not None and ctx.global_expr_max:
        expr_ax = getattr(spec, '_expr_ax', None) or ctx.expr_axes_map.get(id(spec))
        if expr_ax is not None:
            expr_ax.bar([0], [spec.expr], color=color, width=0.4)
            expr_ax.set_xlim(-0.5, 0.5)
            expr_ax.set_ylim(0, ctx.global_expr_max * 1.1)
            expr_ax.set_xticks([])
            expr_ax.set_yticks([])
            for spine in expr_ax.spines.values():
                spine.set_visible(False)

    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


@register_track("gene")
def _build_gene(spec: TrackSpec, ax: plt.Axes, ctx: TrackContext) -> None:
    """Render consolidated gene/transcript models into *ax*."""
    transcripts_data = spec.data
    chrom, start, end = ctx.chrom, ctx.start, ctx.end
    exon_size  = (spec.style or {}).get("exon_size", 7.0)
    gene_size  = (spec.style or {}).get("gene_size", 0.3)
    label_size = (spec.style or {}).get("label_size", 11 * 0.8)

    if hasattr(transcripts_data, "df"):
        df = transcripts_data.df
    else:
        df = transcripts_data

    df = df[
        (df["Chromosome"] == chrom) &
        (df["End"] > start) &
        (df["Start"] < end) &
        (df["Feature"].isin(["transcript", "exon"]))
    ].copy()

    ax.text(-0.005, 0.0, spec.label or "Genes", transform=ax.transAxes, va="bottom", ha="right", fontsize=9)

    if df.empty:
        ax.set_yticks([])
        return

    consolidated_data = []
    for gene_name in df["gene_name"].unique():
        gene_data = df[df["gene_name"] == gene_name]
        exons = gene_data[gene_data["Feature"] == "exon"].copy()
        gene_transcripts = gene_data[gene_data["Feature"] == "transcript"].copy()

        if len(exons) == 0:
            continue

        if not gene_transcripts.empty:
            gene_start = gene_transcripts["Start"].min()
            gene_end = gene_transcripts["End"].max()
        else:
            gene_start = exons["Start"].min()
            gene_end = exons["End"].max()
        strand_counts = gene_transcripts["Strand"].value_counts()
        dominant_strand = strand_counts.index[0] if len(strand_counts) > 0 else "+"

        consolidated_data.append({
            "gene_name": gene_name,
            "Start": gene_start,
            "End": gene_end,
            "Strand": dominant_strand,
            "exons": exons,
        })

    if not consolidated_data:
        ax.set_yticks([])
        return

    consolidated_df = pd.DataFrame(consolidated_data)
    consolidated_df["y"] = trackplot_calculate_segment_height(consolidated_df)
    consolidated_df["Start"] = consolidated_df["Start"].clip(start, end)
    consolidated_df["End"] = consolidated_df["End"].clip(start, end)

    for _, row in consolidated_df.iterrows():
        color = "black" if row["Strand"] in ["+", True, 1] else "darkgrey"
        ax.hlines(y=row["y"], xmin=row["Start"], xmax=row["End"],
                  linewidth=gene_size, color=color)
        for _, exon in row["exons"].iterrows():
            ax.hlines(y=row["y"], xmin=exon["Start"], xmax=exon["End"],
                      linewidth=exon_size, color=color)
        label_x = (row["Start"] + row["End"]) / 2
        ax.text(label_x, row["y"] + 0.1, row["gene_name"],
                ha="center", va="bottom", fontsize=label_size, color="black")

    max_y = consolidated_df["y"].max()
    ax.set_ylim(-0.2, max_y + 0.5)
    ax.set_yticks([])
    ax.grid(False)


@register_track("annotation")
def _build_annotation(spec: TrackSpec, ax: plt.Axes, ctx: TrackContext) -> None:
    """Render genomic annotation lines into *ax*, with optional score-based y positions."""
    loci = spec.data
    chrom, start, end = ctx.chrom, ctx.start, ctx.end
    params = spec.params or {}
    score_column   = params.get("score_column")
    threshold      = params.get("threshold")
    color_by       = params.get("color_by")
    annotation_size = (spec.style or {}).get("annotation_size", 2.5)

    if hasattr(loci, "df"):
        df = loci.df
    else:
        df = loci

    df = df[
        (df["Chromosome"] == chrom) &
        (df["End"] > start) &
        (df["Start"] < end)
    ].copy()

    ax.set_ylabel(spec.label or "Annotation")

    if df.empty:
        ax.set_yticks([])
        return

    if score_column and score_column in df.columns:
        min_score = df[score_column].min()
        max_score = df[score_column].max()
        df["y"] = (
            (df[score_column] - min_score) / (max_score - min_score)
            if max_score > min_score else 0.5
        )
    else:
        df["y"] = 0.5

    df["Start"] = df["Start"].clip(start, end)
    df["End"] = df["End"].clip(start, end)

    for _, row in df.iterrows():
        color = "blue"
        if color_by and color_by in df.columns:
            if df[color_by].dtype in ["int64", "float64"]:
                color = plt.cm.viridis(
                    (row[color_by] - df[color_by].min()) /
                    (df[color_by].max() - df[color_by].min())
                )
            else:
                color = plt.cm.Set3(hash(row[color_by]) % 12)
        ax.hlines(y=row["y"], xmin=row["Start"], xmax=row["End"],
                  linewidth=annotation_size, color=color)

    if threshold is not None and score_column and score_column in df.columns:
        min_score = df[score_column].min()
        max_score = df[score_column].max()
        if max_score > min_score:
            threshold_norm = (threshold - min_score) / (max_score - min_score)
            if 0 <= threshold_norm <= 1:
                ax.axhline(y=threshold_norm, color="red", linestyle="--",
                           linewidth=2, alpha=0.8)

    if score_column and score_column in df.columns:
        ax.set_ylim(-0.05, 1.05)
        min_score = df[score_column].min()
        max_score = df[score_column].max()
        ax2 = ax.twinx()
        ax2.set_ylim(-0.05, 1.05)
        tick_positions = [0, 0.25, 0.5, 0.75, 1.0]
        tick_labels = [
            f"{min_score + p * (max_score - min_score):.1f}" for p in tick_positions
        ]
        ax2.set_yticks(tick_positions)
        ax2.set_yticklabels(tick_labels)
        ax2.set_ylabel(f"{score_column} Score", rotation=270, labelpad=15)
        ax2.tick_params(axis="y", labelsize=9)
        ax.set_yticks([])
    else:
        ax.set_yticks([])


@register_track("loop")
def _build_loop(spec: TrackSpec, ax: plt.Axes, ctx: TrackContext) -> None:
    """Render sin-arc loop curves into *ax*, scaled by an optional score column."""
    loops = spec.data
    chrom, start, end = ctx.chrom, ctx.start, ctx.end
    params = spec.params or {}
    score_column = params.get("score_column")
    curvature    = (spec.style or {}).get("curvature", 0.75)

    if hasattr(loops, "df"):
        df = loops.df
    else:
        df = loops

    loop_left  = df[["Start", "End"]].min(axis=1)
    loop_right = df[["Start", "End"]].max(axis=1)
    df = df[
        (df["Chromosome"] == chrom) &
        (loop_right > start) &
        (loop_left  < end)
    ].copy()

    ax.text(-0.005, 0.0, spec.label or "Loops", transform=ax.transAxes, va="bottom", ha="right", fontsize=9)

    if df.empty:
        ax.set_ylim(0, curvature)
        ax.set_yticks([])
        return

    max_score = min_score = None
    score_range = 1
    if score_column and score_column in df.columns:
        max_score = df[score_column].max()
        min_score = df[score_column].min()
        score_range = max_score - min_score if max_score > min_score else 1

    for _, row in df.iterrows():
        x_left = min(row["Start"], row["End"])
        x_right = max(row["Start"], row["End"])
        if x_left == x_right:
            continue
        x = np.linspace(x_left, x_right, 100)
        arc_height = curvature * 0.9
        y = arc_height * np.sin(np.pi * (x - x_left) / (x_right - x_left))
        color = row.get("color", "blue") if hasattr(row, "get") else "blue"
        if score_column and score_column in df.columns and score_range:
            normalized_score = (row[score_column] - min_score) / score_range  # 0–1
            lw = 0.5 + normalized_score * 1.5  # 0.5–2.0
        else:
            lw = 0.5
        ax.plot(x, y, color=color, linewidth=lw)

    ax.set_ylim(0, curvature)
    ax.set_yticks([])


# ============================================================
# SHAP track utilities and builder
# ============================================================

def flank_peaks(
    peaks_df: pd.DataFrame,
    seq_len: int,
    chrom_col=0,
    start_col=1,
    summit_col=9,
) -> pd.DataFrame:
    """Expand raw narrowPeak rows to fixed-width windows centred on the summit.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Raw BED/narrowPeak table (integer-indexed columns by default).
    seq_len : int
        Total window length (= SHAP sequence length).  Must be even.
    chrom_col, start_col, summit_col : int or str
        Column labels for chromosome, peak start, and summit offset.

    Returns
    -------
    pd.DataFrame with extra columns ``chrom``, ``start``, ``end`` and the
    original index preserved (so rows align 1-to-1 with the SHAP value array).
    """
    flank = seq_len // 2
    df = peaks_df.copy()
    summit = df[start_col] + df[summit_col]
    df["chrom"] = df[chrom_col]
    df["start"] = summit - flank
    df["end"]   = summit + flank
    df.sort_values(by=["chrom", "start", "end"], inplace=True)
    return df


def assemble_shap_for_region(
    shap_values: np.ndarray,
    peaks_flanked: pd.DataFrame,
    chrom: str,
    region_start: int,
    region_end: int,
) -> np.ndarray:
    """Assemble a contiguous (region_len, 4) SHAP array for a genomic region.

    Gaps between peaks are zero-padded; partial overlaps at region edges are
    clipped to the requested interval.

    Parameters
    ----------
    shap_values : np.ndarray
        Shape ``(n_peaks, 4, seq_len)`` — pre-loaded, one entry per peak row.
        Indices must align with the index of *peaks_flanked*.
    peaks_flanked : pd.DataFrame
        Output of :func:`flank_peaks`.  Must have columns ``chrom``,
        ``start``, ``end``.
    chrom : str
        Target chromosome (e.g. ``"chr3"``).
    region_start, region_end : int
        Half-open genomic coordinates of the requested window.

    Returns
    -------
    np.ndarray of shape ``(region_end - region_start, 4)``
    """
    region_len = region_end - region_start
    n_bases = 4
    seq_len = shap_values.shape[2]

    region_peaks = peaks_flanked[
        (peaks_flanked["chrom"] == chrom) &
        (peaks_flanked["start"] < region_end) &
        (peaks_flanked["end"] > region_start)
    ].copy()
    region_peaks.sort_values("start", inplace=True)

    if region_peaks.empty:
        return np.zeros((region_len, n_bases))

    assembled = []
    cursor = region_start  # genomic position of next unwritten base

    for idx, row in region_peaks.iterrows():
        pk_start, pk_end = int(row["start"]), int(row["end"])
        assert (pk_end - pk_start) == seq_len, (
            f"Peak width {pk_end - pk_start} != seq_len {seq_len}"
        )

        # zero-pad gap before this peak
        if cursor < pk_start:
            gap = min(pk_start, region_end) - cursor
            assembled.append(np.zeros((gap, n_bases)))
            cursor = pk_start

        # clip peak to [region_start, region_end)
        src_start = max(0, cursor - pk_start)
        src_end   = min(seq_len, region_end - pk_start)
        if src_end <= src_start:
            continue

        # shap_values[idx] has shape (4, seq_len) → slice and transpose to (bp, 4)
        assembled.append(shap_values[idx][:, src_start:src_end].T)
        cursor = pk_start + src_end

    # zero-pad tail if needed
    if cursor < region_end:
        assembled.append(np.zeros((region_end - cursor, n_bases)))

    return np.concatenate(assembled, axis=0)


def _precompute_shap(tracks: list, ctx: TrackContext) -> None:
    """Compute a shared y-axis range across all shap tracks and store in ctx."""
    shap_specs = [t for t in tracks if t.track_type == "shap"]
    if not shap_specs:
        return

    global_min, global_max = np.inf, -np.inf
    for spec in shap_specs:
        arr = spec.data
        if arr is None or len(arr) == 0:
            continue
        # logomaker stacks positive values up, negative values down
        pos_sum = np.sum(np.where(arr > 0, arr, 0), axis=1)
        neg_sum = np.sum(np.where(arr < 0, arr, 0), axis=1)
        global_min = min(global_min, neg_sum.min())
        global_max = max(global_max, pos_sum.max())

    if global_min == np.inf:
        global_min, global_max = -0.025, 0.1

    ctx.global_shap_ylim = (global_min, global_max)


@register_track("shap")
def _build_shap(spec: TrackSpec, ax: plt.Axes, ctx: TrackContext) -> None:
    """Render a ``(region_len, 4)`` ACGT SHAP array as a DNA logo track.

    All shap tracks in a :func:`compose_tracks` call share the same y-axis
    scale so groups can be visually compared.
    """
    arr = spec.data
    if arr is None or len(arr) == 0:
        ax.set_yticks([])
        return

    df = pd.DataFrame(arr, columns=["A", "C", "G", "T"],
                      index=np.arange(ctx.start, ctx.start + len(arr)))
    logomaker.Logo(df, ax=ax)

    ylim = ctx.global_shap_ylim or (arr.min(), arr.max())
    ypad = (ylim[1] - ylim[0]) * 0.1
    ax.set_ylim(ylim[0] - ypad, ylim[1] + ypad)

    label = spec.label or "SHAP"
    ax.text(-0.005, 0.0, label, transform=ax.transAxes, va="bottom", ha="right", fontsize=9)
    ax.text(0.015, 0.97, f"[{ylim[0]:.3f} \u2013 {ylim[1]:.3f}]",
            transform=ax.transAxes, va="top", ha="left", fontsize=8)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


# ============================================================
# BigWig track utilities and builder
# ============================================================

def _bigwig_total_reads(bw) -> float:
    """Sum all values genome-wide in a pyBigWig handle (RPM denominator)."""
    total = 0.0
    for chrom, size in bw.chroms().items():
        s = bw.stats(chrom, 0, size, type="sum", nBins=1)
        if s and s[0] is not None:
            total += s[0]
    return total


def _precompute_bigwig(tracks: list, ctx: TrackContext) -> None:
    """Fetch signal from each unique BigWig file and populate ``ctx.bigwig_cache``.

    Each entry in the cache maps the file path to a dict with keys
    ``positions`` (array) and ``values`` (array, optionally RPM-normalised).

    RPM normalisation is applied when a ``TrackSpec`` sets
    ``params["normalize_to_1m"] = True``.  The total read count is taken from
    ``params["total_reads"]`` when supplied; otherwise it is estimated by
    summing all values in the BigWig file.
    """
    try:
        import pyBigWig
    except ImportError as exc:
        raise ImportError("pyBigWig is required for bigwig tracks: pip install pyBigWig") from exc

    bw_specs = [t for t in tracks if t.track_type == "bigwig"]
    if not bw_specs:
        return

    global_max = 0.0
    for spec in bw_specs:
        bw_file = spec.data
        params = spec.params or {}
        normalize = params.get("normalize_to_1m", False)

        # Use a cache key that includes normalization intent so that the same
        # file can appear both normalised and raw in the same figure.
        cache_key = (bw_file, normalize)
        if cache_key in ctx.bigwig_cache:
            local_max = ctx.bigwig_cache[cache_key]["values"].max()
            global_max = max(global_max, local_max)
            continue

        with pyBigWig.open(bw_file) as bw:
            chrom_sizes = bw.chroms()
            if ctx.chrom not in chrom_sizes:
                ctx.bigwig_cache[cache_key] = {
                    "positions": np.array([]),
                    "values": np.array([]),
                }
                continue

            region_len = ctx.end - ctx.start
            n_bins = min(ctx.bins, region_len)
            raw = bw.stats(ctx.chrom, ctx.start, ctx.end, type="mean", nBins=n_bins)

        values = np.array([v if v is not None else 0.0 for v in raw], dtype=float)

        if normalize:
            total_reads = params.get("total_reads")
            if total_reads is None:
                with pyBigWig.open(bw_file) as bw:
                    total_reads = _bigwig_total_reads(bw)
            scale = 1e6 / total_reads if total_reads else 1.0
            values = values * scale

        bin_size = region_len / n_bins
        positions = ctx.start + (np.arange(n_bins) + 0.5) * bin_size

        ctx.bigwig_cache[cache_key] = {"positions": positions, "values": values}
        local_max = values.max() if len(values) else 0.0
        global_max = max(global_max, local_max)

    if ctx.global_bigwig_max is None and global_max > 0:
        ctx.global_bigwig_max = global_max


@register_track("bigwig")
def _build_bigwig(spec: TrackSpec, ax: plt.Axes, ctx: TrackContext) -> None:
    """Render a BigWig signal track as a filled area plot.

    Parameters accepted via ``TrackSpec.params``
    -------------------------------------------
    normalize_to_1m : bool, default False
        Divide signal by ``total_reads / 1e6`` (RPM).
    total_reads : int or float, optional
        Total mapped reads for RPM normalisation.  When omitted the value is
        estimated by summing all BigWig entries genome-wide.
    color : str, optional
        Fill colour; defaults to ``"steelblue"``.
    color_map : dict, optional
        Mapping of label → colour.  When provided, the track's label is looked
        up here first; falls back to ``color`` if not found.
    share_ymax : bool, default True
        Use the shared y-axis ceiling across all bigwig tracks in the figure.
    """
    params = spec.params or {}
    normalize = params.get("normalize_to_1m", False)
    share_ymax = params.get("share_ymax", True)
    color_map = spec.colors or params.get("color_map")
    color = (color_map.get(spec.label) if color_map and spec.label in color_map
             else params.get("color", "steelblue"))

    cache_key = (spec.data, normalize)
    cached = ctx.bigwig_cache.get(cache_key)
    if cached is None or len(cached["values"]) == 0:
        ax.set_yticks([])
        return

    positions = cached["positions"]
    values = cached["values"]

    ax.fill_between(positions, values, color=color, linewidth=0)

    if share_ymax and ctx.global_bigwig_max:
        ymax = ctx.global_bigwig_max
    else:
        ymax = values.max() if values.max() > 0 else 1.0
    ax.set_ylim(0, ymax * 1.1)

    label = spec.label or ""
    ax.text(-0.005, 0.0, label, transform=ax.transAxes, va="bottom", ha="right", fontsize=9)

    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


# ============================================================
# scE2G track utilities and builder
# ============================================================

def _discover_sce2g_cell_types(folder: str) -> list:
    """Return sorted list of cell type names found as subdirectories of *folder*."""
    import os
    return sorted(
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d))
        and os.path.isdir(os.path.join(folder, d, "multiome_powerlaw_v3"))
    )


def _nice_scale(target: float) -> float:
    """Round *target* to the nearest human-readable genomic scale (1k, 2k, 5k, 10k…)."""
    import math
    mag = 10 ** math.floor(math.log10(target))
    candidates = [mag, 2 * mag, 5 * mag, 10 * mag]
    return min(candidates, key=lambda x: abs(x - target))


def _get_gene_tss(gene_data, gene_name: str):
    """Return the TSS position for a single gene from a transcript DataFrame."""
    df = gene_data.df if hasattr(gene_data, "df") else gene_data
    txs = df[(df["Feature"] == "transcript") & (df["gene_name"] == gene_name)]
    if txs.empty:
        return None
    strand = txs["Strand"].mode().iloc[0]
    return int(txs["Start"].min()) if strand == "+" else int(txs["End"].max())


def _precompute_sce2g(tracks: list, ctx: TrackContext) -> None:
    """Load scE2G prediction files and populate ``ctx.sce2g_cache``.

    Cache key is ``(folder, cell_type)``.  Sets ``ctx.global_sce2g_score_range``
    to ``(min, max)`` of ``E2G.Score.qnorm`` across all loaded DataFrames.
    """
    import glob as _glob

    sce2g_specs = [t for t in tracks if t.track_type == "sce2g"]
    if not sce2g_specs:
        return

    # Auto-detect gene track data as fallback TSS source
    gene_track = next((t for t in tracks if t.track_type == "gene"), None)
    default_gene_data = gene_track.data if gene_track else None

    # Collect only the genes explicitly requested via gene= option
    target_genes = {spec.gene for spec in sce2g_specs if spec.gene}

    # For specs without gene=, add only genes visible in the current window
    if any(spec.gene is None for spec in sce2g_specs) and default_gene_data is not None:
        gdf = default_gene_data.df if hasattr(default_gene_data, "df") else default_gene_data
        window_genes = gdf[
            (gdf["Feature"] == "transcript") &
            (gdf["Chromosome"] == ctx.chrom) &
            (gdf["Start"] <= ctx.end) &
            (gdf["End"] >= ctx.start)
        ]["gene_name"].unique()
        target_genes.update(window_genes)

    # Compute TSS only for those genes (fast: one boolean filter per gene)
    tss_map = {}
    if target_genes and default_gene_data is not None:
        for gene in target_genes:
            tss = _get_gene_tss(default_gene_data, gene)
            if tss is not None:
                tss_map[gene] = tss

    global_min, global_max = np.inf, -np.inf

    for spec in sce2g_specs:
        folder = spec.data
        cell_type = spec.cell_type
        if folder is None or cell_type is None:
            continue

        cache_key = (folder, cell_type)
        if cache_key in ctx.sce2g_cache:
            df = ctx.sce2g_cache[cache_key]
        else:
            params = spec.params or {}
            model_dir = params.get("model_dir", "multiome_powerlaw_v3")
            file_glob = params.get("file_glob", "scE2G_predictions_threshold*.tsv.gz")
            pattern = f"{folder}/{cell_type}/{model_dir}/{file_glob}"
            matches = _glob.glob(pattern)
            if not matches:
                ctx.sce2g_cache[cache_key] = pd.DataFrame()
                continue
            df = pd.read_csv(matches[0], sep="\t")
            score_col = params.get("score_column", "E2G.Score.qnorm")
            threshold = params.get("threshold", 0.0)

            df["Start"] = (df["start"] + df["end"]) / 2
            df["End"] = df["TargetGeneTSS"]
            df["Chromosome"] = df["chr"]
            df["score"] = df[score_col]

            if tss_map:
                df["End"] = df["TargetGene"].map(tss_map).fillna(df["End"])

            if threshold > 0:
                df = df[df["score"] > threshold]

            df = df[
                (df["Chromosome"] == ctx.chrom) &
                (
                    ((df["Start"] >= ctx.start) & (df["Start"] <= ctx.end)) |
                    ((df["End"] >= ctx.start) & (df["End"] <= ctx.end))
                )
            ].copy()
            ctx.sce2g_cache[cache_key] = df

        if not df.empty and "score" in df.columns:
            global_min = min(global_min, df["score"].min())
            global_max = max(global_max, df["score"].max())

    if global_min != np.inf:
        ctx.global_sce2g_score_range = (global_min, global_max)


@register_track("sce2g")
def _build_sce2g(spec: TrackSpec, ax: plt.Axes, ctx: TrackContext) -> None:
    """Render scE2G enhancer-gene arcs into *ax*.

    Each arc connects the enhancer midpoint to the gene TSS, with line width
    scaled by ``E2G.Score.qnorm`` relative to the global score range.
    """
    cell_type = spec.cell_type
    cache_key = (spec.data, cell_type)
    df = ctx.sce2g_cache.get(cache_key, pd.DataFrame())

    curvature = (spec.style or {}).get("curvature", 0.75)
    color = (spec.style or {}).get("color", "steelblue")
    label = spec.label or cell_type or "scE2G"

    if spec.gene:
        df = df[df["TargetGene"] == spec.gene]

    if df.empty:
        ax.set_ylim(0, curvature)
        ax.set_yticks([])
        ax.text(-0.005, 0.0, label, transform=ax.transAxes, va="bottom", ha="right", fontsize=9)
        return

    score_range = ctx.global_sce2g_score_range
    if score_range is not None:
        s_min, s_max = score_range
        score_span = s_max - s_min if s_max > s_min else 1.0
    else:
        s_min, score_span = 0.0, 1.0

    for _, row in df.iterrows():
        x_left  = min(row["Start"], row["End"])
        x_right = max(row["Start"], row["End"])
        if x_left == x_right:
            continue
        x = np.linspace(x_left, x_right, 100)
        arc_height = curvature * 0.9
        y = arc_height * np.sin(np.pi * (x - x_left) / (x_right - x_left))

        score = row["score"]
        normalized = (score - s_min) / score_span
        lw = 0.3 + normalized * 1.7  # 0.3–2.0
        ax.plot(x, y, color=color, linewidth=lw)

    if spec.highlight_loop is not None:
        hdf = spec.highlight_loop.df if hasattr(spec.highlight_loop, "df") else spec.highlight_loop
        hdf = hdf[
            (hdf["Chromosome"] == ctx.chrom) &
            (hdf[["Start", "End"]].max(axis=1) > ctx.start) &
            (hdf[["Start", "End"]].min(axis=1) < ctx.end)
        ]
        h_color = (spec.style or {}).get("highlight_color", "pink")
        h_alpha = (spec.style or {}).get("highlight_alpha", 0.7)
        for _, row in hdf.iterrows():
            x_left  = min(row["Start"], row["End"])
            x_right = max(row["Start"], row["End"])
            if x_left == x_right:
                continue
            x = np.linspace(x_left, x_right, 100)
            y = curvature * 0.9 * np.sin(np.pi * (x - x_left) / (x_right - x_left))
            ax.plot(x, y, color=h_color, linewidth=3.0, alpha=h_alpha)

    ax.set_ylim(0, curvature)
    ax.set_yticks([])
    ax.text(-0.005, 0.0, label, transform=ax.transAxes, va="bottom", ha="right", fontsize=9)


# ============================================================
# scGLUE peak-to-gene link track
# ============================================================

def _parse_peak_midpoint(coord):
    """Parse 'chrN:start-end' into (chrom, midpoint_int), or None on failure."""
    try:
        chrom, rest = str(coord).split(":", 1)
        a, b = rest.split("-")
        return chrom, (int(a) + int(b)) // 2
    except Exception:
        return None, None


def _precompute_scglue(tracks: list, ctx: TrackContext) -> None:
    """Load scGLUE peak2gene TSV and populate ``ctx.scglue_cache``.

    Cache key is ``spec.data`` (file path).  The file has swapped column
    names: ``peak`` actually holds gene names and ``gene`` holds peak
    coordinates (``chrN:start-end``).  TSS inference mirrors ``_precompute_sce2g``.
    """
    scglue_specs = [t for t in tracks if t.track_type == "scglue"]
    if not scglue_specs:
        return

    gene_track = next((t for t in tracks if t.track_type == "gene"), None)
    default_gene_data = gene_track.data if gene_track else None

    # Collect target genes (explicit spec.gene + window-visible transcripts)
    target_genes = {spec.gene for spec in scglue_specs if spec.gene}
    if any(spec.gene is None for spec in scglue_specs) and default_gene_data is not None:
        gdf = default_gene_data.df if hasattr(default_gene_data, "df") else default_gene_data
        window_genes = gdf[
            (gdf["Feature"] == "transcript") &
            (gdf["Chromosome"] == ctx.chrom) &
            (gdf["Start"] <= ctx.end) &
            (gdf["End"] >= ctx.start)
        ]["gene_name"].unique()
        target_genes.update(window_genes)

    # Build TSS map for only those genes
    tss_map = {}
    if default_gene_data is None:
        import warnings
        warnings.warn(
            "scGLUE: no gene track found — cannot infer TSS positions. "
            "Add a TrackSpec('gene', data=gtf) to your track list."
        )
    elif target_genes:
        for gene in target_genes:
            tss = _get_gene_tss(default_gene_data, gene)
            if tss is not None:
                tss_map[gene] = tss
        missing = target_genes - tss_map.keys()
        if missing:
            import warnings
            warnings.warn(f"scGLUE: TSS not found for genes: {missing}")


    global_min, global_max = np.inf, -np.inf

    for spec in scglue_specs:
        cache_key = spec.data
        if cache_key in ctx.scglue_cache:
            df = ctx.scglue_cache[cache_key]
        else:
            try:
                params = spec.params or {}
                score_col = params.get("score_column", "score")
                threshold = params.get("threshold", 0.0)

                # utf-8-sig strips BOM if present
                raw = pd.read_csv(spec.data, sep="\t", encoding="utf-8-sig")
                raw.columns = raw.columns.str.strip()
                # Columns are swapped: 'peak' = gene name, 'gene' = peak coords
                raw = raw.rename(columns={"peak": "gene_name", "gene": "peak_coord"})

                # Infer Start from peak locus midpoint
                parsed = raw["peak_coord"].map(_parse_peak_midpoint)
                raw["Chromosome"] = parsed.map(lambda x: x[0])
                raw["Start"] = parsed.map(lambda x: x[1])
                raw = raw.dropna(subset=["Chromosome", "Start"]).copy()
                raw["Start"] = raw["Start"].astype(int)

                # Infer End from gene TSS (same as scE2G)
                raw["End"] = raw["gene_name"].map(tss_map)

                raw["score"] = raw[score_col]
                if threshold > 0:
                    raw = raw[raw["score"] > threshold]

                raw = raw.dropna(subset=["End"]).copy()
                raw["End"] = raw["End"].astype(int)

                df = raw[
                    (raw["Chromosome"] == ctx.chrom) &
                    (
                        ((raw["Start"] >= ctx.start) & (raw["Start"] <= ctx.end)) |
                        ((raw["End"] >= ctx.start) & (raw["End"] <= ctx.end))
                    )
                ].copy()
            except Exception as e:
                import warnings
                warnings.warn(f"scGLUE: failed to load {spec.data!r}: {e}")
                df = pd.DataFrame(columns=["gene_name", "peak_coord", "Chromosome", "Start", "End", "score"])

            ctx.scglue_cache[cache_key] = df

        if not df.empty and "score" in df.columns:
            global_min = min(global_min, df["score"].min())
            global_max = max(global_max, df["score"].max())

    if global_min != np.inf:
        ctx.global_scglue_score_range = (global_min, global_max)


@register_track("scglue")
def _build_scglue(spec: TrackSpec, ax: plt.Axes, ctx: TrackContext) -> None:
    """Render scGLUE peak-to-gene arcs into *ax*.

    Each arc connects the peak midpoint to the gene TSS, with line width
    scaled by ``score`` relative to the global score range.
    """
    df = ctx.scglue_cache.get(spec.data, pd.DataFrame())

    curvature = (spec.style or {}).get("curvature", 0.75)
    color = (spec.style or {}).get("color", "steelblue")
    label = spec.label or "scGLUE"

    if spec.gene and "gene_name" in df.columns:
        df = df[df["gene_name"] == spec.gene]

    ax.text(-0.005, 0.0, label, transform=ax.transAxes, va="bottom", ha="right", fontsize=9)

    if df.empty:
        ax.set_ylim(0, curvature)
        ax.set_yticks([])
        return

    score_range = ctx.global_scglue_score_range
    if score_range is not None:
        s_min, s_max = score_range
        score_span = s_max - s_min if s_max > s_min else 1.0
    else:
        s_min, score_span = 0.0, 1.0

    for _, row in df.iterrows():
        x_left  = min(row["Start"], row["End"])
        x_right = max(row["Start"], row["End"])
        if x_left == x_right:
            continue
        x = np.linspace(x_left, x_right, 100)
        arc_height = curvature * 0.9
        y = arc_height * np.sin(np.pi * (x - x_left) / (x_right - x_left))
        score = row["score"]
        normalized = (score - s_min) / score_span
        lw = 0.3 + normalized * 1.7  # 0.3–2.0
        ax.plot(x, y, color=color, linewidth=lw)

    ax.set_ylim(0, curvature)
    ax.set_yticks([])


# ------------------------------------------------------------------
# compose_tracks — main public API
# ------------------------------------------------------------------

def compose_tracks(
    tracks: list,
    region: str,
    cell_groups: pd.Series = None,
    cell_counts: pd.Series = None,
    colors: dict = None,
    sort_by: list = None,
    bins: int = 1000,
    clip_quantile: float = 0.999,
    hspace: float = 0,
    save: str = None,
    dpi: int = 300,
    highlight_regions: list = None,
    highlight_color: str = "yellow",
    highlight_alpha: float = 0.3,
    figwidth: float = 12,
    expr_col_width: float = 0.06,
) -> plt.Figure:
    """Compose a list of :class:`TrackSpec` objects into one multi-panel figure.

    Parameters
    ----------
    tracks : list[TrackSpec]
        Ordered track specifications.
    region : str
        Genomic region, e.g. ``"chr3:69729463-69978332"``.
    cell_groups : pd.Series, optional
        Barcode → group name mapping (required for coverage tracks).
    cell_counts : pd.Series, optional
        Barcode → read count mapping (for RPKM normalisation).
    colors : dict, optional
        Group → colour mapping; defaults to a ``tab10`` palette.
    sort_by : list[str], optional
        Re-order tracks before rendering.  Supported keys:
        ``"cell_type"``, ``"track_type"``.
    bins : int
        Genomic bins used for coverage histograms.
    clip_quantile : float
        Upper quantile for clipping coverage extremes.
    hspace : float
        Vertical gap between track panels (matplotlib *hspace*).
    save : str, optional
        File path to save the figure.
    dpi : int
        Resolution for the saved figure.
    highlight_regions : list[str], optional
        Genomic regions to shade, e.g. ``["chr3:69800000-69850000"]``.
        Each entry uses the same ``"chrN:start-end"`` format as *region*.
    highlight_color : str
        Fill colour for all highlight zones (default ``"yellow"``).
    highlight_alpha : float
        Opacity of the highlight zones (default ``0.3``).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If a ``TrackSpec`` references an unregistered ``track_type``.
    """
    chrom, start, end = re.split(r"[:-]", region)
    start, end = int(start), int(end)

    if sort_by:
        tracks = _sort_tracks(tracks, sort_by)

    # Generate per-spec default colors for specs with their own cell_groups
    for spec in tracks:
        if spec.track_type == "coverage" and spec.cell_groups is not None and spec.colors is None:
            unique = list(spec.cell_groups.unique())
            palette = sns.color_palette("tab10", len(unique))
            spec.colors = dict(zip(unique, palette))

    # Expand coverage / sce2g specs with no cell_type into one panel per cell type
    expanded = []
    for spec in tracks:
        if spec.track_type == "coverage" and isinstance(spec.cell_type, list):
            for i, ct in enumerate(spec.cell_type):
                if isinstance(spec.expr, dict):
                    expr_val = spec.expr.get(ct)
                elif hasattr(spec.expr, '__len__'):
                    expr_val = spec.expr[i]
                else:
                    expr_val = spec.expr  # scalar or None
                expanded.append(dataclasses.replace(spec, cell_type=ct, label=spec.label or ct, expr=expr_val))
        elif spec.track_type == "coverage" and spec.cell_type is None:
            eff_cg = spec.cell_groups if spec.cell_groups is not None else cell_groups
            if eff_cg is not None:
                order = spec.cell_order if spec.cell_order is not None else list(eff_cg.unique())
                for ct in order:
                    expanded.append(dataclasses.replace(spec, cell_type=ct))
            else:
                expanded.append(spec)
        elif spec.track_type == "sce2g":
            ct = spec.cell_type
            if ct is None:
                cell_types = spec.cell_order or _discover_sce2g_cell_types(spec.data)
            elif isinstance(ct, list):
                cell_types = ct
            else:
                cell_types = None  # single string — no expansion needed
            if cell_types is not None:
                for c in cell_types:
                    expanded.append(dataclasses.replace(spec, cell_type=c, label=spec.label or c))
                continue
            expanded.append(spec)
        else:
            expanded.append(spec)
    tracks = expanded

    expr_values = [s.expr for s in tracks if s.track_type == "coverage" and s.expr is not None]
    global_expr_max = max(expr_values) if expr_values else None

    ctx = TrackContext(
        chrom=chrom, start=start, end=end,
        cell_groups=cell_groups, cell_counts=cell_counts,
        colors=colors, bins=bins, clip_quantile=clip_quantile,
    )

    # Default colour palette for global cell_groups
    if ctx.colors is None and ctx.cell_groups is not None:
        unique_groups = ctx.cell_groups.unique()
        palette = sns.color_palette("tab10", len(unique_groups))
        ctx.colors = dict(zip(unique_groups, palette))

    ctx.global_expr_max = global_expr_max

    _precompute_coverage(tracks, ctx)
    _precompute_shap(tracks, ctx)
    _precompute_bigwig(tracks, ctx)
    _precompute_sce2g(tracks, ctx)
    _precompute_scglue(tracks, ctx)

    heights = [t.height or _DEFAULT_HEIGHTS.get(t.track_type, 1.0) for t in tracks]
    n_tracks = len(tracks)

    from matplotlib.gridspec import GridSpec

    has_expr = any(s.track_type == "coverage" and s.expr is not None for s in tracks)

    fig = plt.figure(figsize=(figwidth, sum(heights)))

    if has_expr:
        gs = GridSpec(
            n_tracks, 2, figure=fig,
            height_ratios=heights,
            width_ratios=[1, expr_col_width],
            hspace=hspace, wspace=0.005,
        )
    else:
        gs = GridSpec(
            n_tracks, 1, figure=fig,
            height_ratios=heights,
            hspace=hspace,
        )

    axes = []
    first_ax = None
    shared_expr_ax = None
    for i, spec in enumerate(tracks):
        ax = fig.add_subplot(gs[i, 0], sharex=first_ax)
        if first_ax is None:
            first_ax = ax
        axes.append(ax)

        if has_expr:
            if spec.track_type == "coverage" and spec.expr is not None:
                expr_ax = fig.add_subplot(gs[i, 1], sharey=shared_expr_ax)
                if shared_expr_ax is None:
                    shared_expr_ax = expr_ax
                ctx.expr_axes_map[id(spec)] = expr_ax
                spec._expr_ax = expr_ax  # direct reference to avoid id() lookup issues
            else:
                empty = fig.add_subplot(gs[i, 1])
                empty.set_visible(False)
                spec._expr_ax = None
        else:
            spec._expr_ax = None

    for spec, ax in zip(tracks, axes):
        if spec.track_type not in TRACK_BUILDERS:
            raise ValueError(
                f"Unknown track type: '{spec.track_type}'. "
                f"Available types: {sorted(TRACK_BUILDERS.keys())}"
            )
        TRACK_BUILDERS[spec.track_type](spec, ax, ctx)
        ax.set_xlim(start - 0.5, end - 0.5)
        if not spec.frameon:
            for spine in ax.spines.values():
                spine.set_visible(False)

    if has_expr and shared_expr_ax is not None and ctx.global_expr_max:
        max_val = ctx.global_expr_max
        for expr_ax in ctx.expr_axes_map.values():
            expr_ax.yaxis.tick_right()
            expr_ax.set_yticks([0, max_val])
            expr_ax.set_yticklabels(['', ''], fontsize=6)
            expr_ax.tick_params(axis='y', length=2, pad=0)
            expr_ax.spines['right'].set_visible(True)
            expr_ax.spines['right'].set_linewidth(0.5)
            expr_ax.spines['right'].set_bounds(0, max_val)
        shared_expr_ax.set_title("RNA-seq", fontsize=7, pad=2)
        shared_expr_ax.set_yticklabels([f"{0}", f"{max_val:.2g}"], fontsize=6)
        shared_expr_ax.tick_params(axis='y', length=2, pad=0)

    if highlight_regions:
        for zone in highlight_regions:
            _, zstart, zend = re.split(r"[:-]", zone)
            zstart, zend = int(zstart), int(zend)
            for i, ax in enumerate(axes):
                if tracks[i].track_type == "coverage":
                    ax.axvspan(zstart, zend, color=highlight_color, alpha=highlight_alpha, zorder=0)

    # x-axis formatting on the bottom panel only
    formatter = FuncFormatter(lambda x, _: f"{int(x):,}")
    axes[-1].xaxis.set_major_formatter(formatter)
    axes[-1].set_xlabel("")


    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.12, right=0.88)

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")

    return fig




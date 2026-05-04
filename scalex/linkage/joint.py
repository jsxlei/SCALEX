from scipy.cluster.hierarchy import linkage, dendrogram as _scipy_dendrogram
from scipy.spatial.distance import pdist

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ── Module 1: Filtering ───────────────────────────────────────────────────────

def filter_by_nonzero(primary_matrix, hits_merged, secondary_matrix=None):
    """
    Filter rows where ALL values in primary_matrix are zero.
    secondary_matrix (if provided) is reindexed to match and filtered with the same mask.
    """
    nonzero_mask = ~(primary_matrix == 0).all(axis=1)
    primary_f = primary_matrix[nonzero_mask].copy()
    secondary_f = (
        secondary_matrix.reindex(primary_matrix.index)[nonzero_mask].copy()
        if secondary_matrix is not None else None
    )
    pattern_labels = hits_merged.iloc[nonzero_mask.to_numpy()]['pattern'].values
    print(f"Kept {nonzero_mask.sum():,} / {len(primary_matrix):,} non-zero rows")
    return primary_f, secondary_f, pattern_labels


# ── Module 2: Per-pattern averaging ──────────────────────────────────────────

def avg_by_pattern(matrix, pattern_labels, patterns_to_show=None):
    """Attach pattern labels, groupby mean, optionally filter to specific patterns."""
    df = matrix.copy()
    df['pattern'] = pattern_labels
    avg = df.groupby('pattern').mean()
    if patterns_to_show is not None:
        avg = avg.loc[avg.index.isin(patterns_to_show)]
    return avg


# ── Module 3: Cell-type group parsing ────────────────────────────────────────

def parse_cell_type_groups(cell_type_annotation, columns):
    """
    Parse annotation into {group_name: [cell_type_cols]}, preserving row order.
    Accepts a dict {cell_type: group} or a DataFrame [cell_type, group].
    """
    if isinstance(cell_type_annotation, pd.DataFrame):
        ct_col, grp_col = cell_type_annotation.columns[:2]
        mapping = dict(zip(cell_type_annotation[ct_col], cell_type_annotation[grp_col]))
    else:
        mapping = dict(cell_type_annotation)

    cols_set = set(columns)
    missing  = set(mapping.keys()) - cols_set
    if missing:
        print(f"  Warning: cell types not found in matrix and skipped: {sorted(missing)}")

    groups = {}
    for cell_type, group in mapping.items():
        if cell_type in cols_set:
            groups.setdefault(group, []).append(cell_type)
    return groups


# ── Module 4: Clustermap cleaning ────────────────────────────────────────────

def clean_for_clustermap(df, z_score=False):
    """
    Drop rows with NaN/inf; for z-score mode also drop constant rows (std=0).
    Does NOT apply z-score — that is done manually where needed.
    """
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if z_score:
        df = df[df.std(axis=1) > 0]
    return df


# ── Module 5: Side-by-side heatmap plotting ───────────────────────────────────

def plot_heatmaps_side_by_side(shap_df=None, atac_df=None, group_name='', col_order=None):
    """
    Plot SHAP and/or ATAC heatmaps. When both are provided they appear side by side.
    When only one is provided a single-panel figure is shown.
    Rows are clustered from whichever matrix is primary (SHAP preferred).
    Produces raw and z-score variants.

    Returns
    -------
    captured : dict {title: (row_order, col_order)} for downstream boxplot alignment.
    """
    captured = {}
    primary_df  = shap_df if shap_df is not None else atac_df
    both        = shap_df is not None and atac_df is not None

    def _row_zscore(df):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

    for title, do_zscore in [
        ('raw',                 False),
        ('z-score per pattern', True),
    ]:
        primary_c = clean_for_clustermap(primary_df, z_score=do_zscore)
        atac_c    = clean_for_clustermap(atac_df, z_score=do_zscore) if both else None

        if col_order is not None:
            primary_c = primary_c[[c for c in col_order if c in primary_c.columns]]
            if atac_c is not None:
                atac_c = atac_c[[c for c in col_order if c in atac_c.columns]]

        # Cluster rows from primary matrix
        Z         = linkage(pdist(primary_c.fillna(0).values), method='average')
        row_order = primary_c.index[_scipy_dendrogram(Z, no_plot=True)['leaves']].tolist()
        primary_c = primary_c.reindex(row_order)
        if atac_c is not None:
            atac_c = atac_c.reindex([r for r in row_order if r in atac_c.index])
        captured[title] = (row_order, primary_c.columns.tolist())

        if do_zscore:
            primary_c = _row_zscore(primary_c)
            if atac_c is not None:
                atac_c = _row_zscore(atac_c)
                valid  = primary_c.dropna(how='any').index.intersection(
                         atac_c.dropna(how='any').index)
                primary_c = primary_c.loc[valid]
                atac_c    = atac_c.loc[valid]
            else:
                primary_c = primary_c.dropna(how='any')

        # Resolve which cleaned df is shap_c for labelling
        shap_c = primary_c if shap_df is not None else None

        n_rows = len(primary_c)
        row_h  = max(6, n_rows * 0.35 + 1)

        if both:
            n_s   = len(primary_c.columns)
            n_a   = len(atac_c.columns)
            col_w = (n_s + n_a) * 0.7 + 4
            fig, (ax_left, ax_right) = plt.subplots(
                1, 2,
                figsize=(col_w, row_h),
                gridspec_kw={'width_ratios': [n_s, n_a], 'wspace': 0.4},
            )
            sns.heatmap(primary_c, cmap='RdBu_r', center=0,
                        xticklabels=True, yticklabels=True, ax=ax_left)
            ax_left.set_title('SHAP')
            ax_left.tick_params(axis='x', rotation=45, labelsize=8)

            sns.heatmap(atac_c, cmap='RdBu_r', center=0,
                        xticklabels=True, yticklabels=False, ax=ax_right)
            ax_right.set_title('ATAC')
            ax_right.tick_params(axis='x', rotation=45, labelsize=8)
        else:
            panel_label = 'SHAP' if shap_df is not None else 'ATAC'
            n_cols = len(primary_c.columns)
            col_w  = n_cols * 0.7 + 4
            fig, ax = plt.subplots(figsize=(col_w, row_h))
            sns.heatmap(primary_c, cmap='RdBu_r', center=0,
                        xticklabels=True, yticklabels=True, ax=ax)
            ax.set_title(panel_label)
            ax.tick_params(axis='x', rotation=45, labelsize=8)

        fig.suptitle(f'{group_name} — {title}', y=1.01)
        plt.tight_layout()
        plt.show()

    return captured


# ── Module 6: Boxplot plotting ────────────────────────────────────────────────

def plot_boxplots(raw_df, group_name, data_label, patterns_sorted, col_order=None, ncols=4):
    """
    Grid of per-pattern boxplots. col_order fixes x-axis cell-type sequence.
    patterns_sorted controls subplot order (pass SHAP's row order for alignment).
    """
    value_cols = col_order if col_order is not None else [c for c in raw_df.columns if c != 'pattern']
    value_cols = [c for c in value_cols if c in raw_df.columns]

    n     = len(patterns_sorted)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, pattern in enumerate(patterns_sorted):
        group = raw_df[raw_df['pattern'] == pattern][value_cols]
        sns.boxplot(data=group, ax=axes[i], showfliers=False)
        axes[i].axhline(0, color='gray', linestyle='--', linewidth=0.8)
        axes[i].set_title(pattern, fontsize=8)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right', fontsize=7)
        axes[i].tick_params(axis='y', labelsize=7)
        axes[i].set_xlabel(f'n={len(group):,}', fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'{data_label} distribution by pattern | {group_name}', y=1.01)
    plt.tight_layout()
    plt.show()


# ── Main entry point ──────────────────────────────────────────────────────────

def analyze_motif(
    hits_merged,
    cell_type_annotation,
    shap_matrix=None,
    atac_matrix=None,
    patterns_to_show=None,
    plot_heatmap=True,
    plot_boxplot=True,
    ncols_boxplot=4,
):
    """
    Motif analysis grouped by pattern. SHAP and ATAC matrices are independently optional.

    Parameters
    ----------
    hits_merged          : DataFrame with 'pattern' column, same row order as the matrices
    cell_type_annotation : dict {cell_type: group} or DataFrame [cell_type, group].
                           Row order defines the fixed column order in all plots.
    shap_matrix          : DataFrame (intervals × cell_types), optional
    atac_matrix          : DataFrame (intervals × cell_types), optional
    patterns_to_show     : list of pattern names to include, or None for all
    plot_heatmap         : whether to produce heatmaps
    plot_boxplot         : whether to produce boxplot distribution plots
    ncols_boxplot        : columns in the boxplot subplot grid
    """
    if shap_matrix is None and atac_matrix is None:
        raise ValueError("At least one of shap_matrix or atac_matrix must be provided.")

    # Step 1 — filter non-zero rows; SHAP is primary when both are available
    primary   = shap_matrix if shap_matrix is not None else atac_matrix
    secondary = atac_matrix if shap_matrix is not None else None

    primary_f, secondary_f, pattern_labels = filter_by_nonzero(primary, hits_merged, secondary)

    shap_f = primary_f   if shap_matrix is not None else None
    atac_f = secondary_f if shap_matrix is not None else primary_f

    # Step 2 — per-pattern averages (for heatmaps)
    shap_avg = avg_by_pattern(shap_f, pattern_labels, patterns_to_show) if shap_f is not None else None
    atac_avg = avg_by_pattern(atac_f, pattern_labels, patterns_to_show) if atac_f is not None else None
    primary_avg = shap_avg if shap_avg is not None else atac_avg

    # Step 3 — attach pattern labels for boxplots, apply pattern filter
    def _make_raw(df):
        raw = df.copy()
        raw['pattern'] = pattern_labels
        if patterns_to_show is not None:
            raw = raw[raw['pattern'].isin(patterns_to_show)]
        return raw

    shap_raw = _make_raw(shap_f) if shap_f is not None else None
    atac_raw = _make_raw(atac_f) if atac_f is not None else None

    # Step 4 — parse annotation → {group: [cols]}
    if cell_type_annotation is None:
        all_cols = [c for c in primary_avg.columns if c != 'pattern']
        groups = {'all': all_cols}
    else:
        groups = parse_cell_type_groups(cell_type_annotation, primary_avg.columns)
    for group_name, cols in groups.items():
        print(f"{group_name}: {cols}")
    print()

    # Step 5 — per group: heatmap then per-source boxplots
    for group_name, cols in groups.items():
        if plot_heatmap:
            captured = plot_heatmaps_side_by_side(
                shap_df=shap_avg[cols] if shap_avg is not None else None,
                atac_df=atac_avg[cols] if atac_avg is not None else None,
                group_name=group_name,
                col_order=cols,
            )
            pattern_seq = captured.get('raw', (sorted(primary_avg.index.tolist()), None))[0]
        else:
            pattern_seq = sorted(primary_avg.index.tolist())

        if plot_boxplot:
            sources = []
            if shap_raw is not None: sources.append(('SHAP', shap_raw))
            if atac_raw is not None: sources.append(('ATAC', atac_raw))
            for data_label, raw_df in sources:
                plot_boxplots(raw_df[cols + ['pattern']], group_name, data_label,
                              pattern_seq, col_order=cols, ncols=ncols_boxplot)

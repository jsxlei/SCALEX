import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import hypergeom

from scalex.pl._utils import _sort_key


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 0.0


def overlap_coefficient(a, b):
    a, b = set(a), set(b)
    m = min(len(a), len(b))
    return len(a & b) / m if m else 0.0


def enrichment(A, B, N):
    """Hypergeometric over-representation of the overlap of two gene sets.

    A, B should already be intersected with the universe of size N. Returns
    ``(log2_fold_enrichment, pvalue)``. log2FE = log2(observed / expected) with
    expected = |A|*|B|/N; pvalue = P(X >= k) = hypergeom.sf(k-1, N, |A|, |B|).
    Undefined cases (empty set, no overlap) return ``(np.nan, 1.0)`` so they
    render as neutral rather than as extreme depletion.
    """
    A, B = set(A), set(B)
    a, b = len(A), len(B)
    k = len(A & B)
    if a == 0 or b == 0 or N == 0:
        return np.nan, 1.0
    expected = a * b / N
    if k == 0 or expected == 0:
        return np.nan, 1.0
    log2fe = float(np.log2(k / expected))
    pval = float(hypergeom.sf(k - 1, N, a, b))
    return log2fe, pval


def _bh_fdr(pvals):
    """Benjamini-Hochberg FDR (numpy-2 safe; no np.asfarray)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0, 1)
    return out


def plot_jaccard_heatmap(dict_a, dict_b, vmax=None, figsize_scale=0.4,
                         name_a=None, name_b=None, save_path=None,
                         order_a=None, order_b=None,
                         metric='jaccard', background=None, vmin=None,
                         fdr_threshold=0.05):
    """Overlap heatmap between the per-cell-type gene sets of dict_a and dict_b.

    metric:
        'jaccard'    -> |A∩B|/|A∪B|                       (default; viridis)
        'overlap'    -> |A∩B|/min(|A|,|B|)                (viridis)
        'enrichment' -> log2 fold-enrichment from a hypergeometric test, which
                        conditions on both set sizes and the gene universe so the
                        grid is comparable across cell types of different sizes.
                        Colored on a diverging map centered at 0 (= chance);
                        cells with BH-FDR < fdr_threshold are marked with '*'.
    background (enrichment only):
        a set/list of gene names defining the universe (preferred — each marker
        set is intersected with it and sizes taken relative to it), or an int N.
        If None, falls back to the union of all genes in dict_a/dict_b (markers
        only) and warns, since that inflates fold-enrichment.
    """
    rows = [c for c in order_a if c in dict_a] if order_a else sorted(dict_a, key=_sort_key)
    cols = [c for c in order_b if c in dict_b] if order_b else sorted(dict_b, key=_sort_key)

    M = pd.DataFrame(index=rows, columns=cols, dtype=float)
    FDR = None

    if metric == 'enrichment':
        # Resolve the gene universe.
        if background is None:
            import warnings
            U = set().union(*[set(v) for v in dict_a.values()],
                            *[set(v) for v in dict_b.values()])
            N = len(U)
            warnings.warn("plot_jaccard_heatmap(metric='enrichment') called without "
                          "`background`; using the marker-gene union as the universe, "
                          "which inflates fold-enrichment. Pass the shared measured-gene "
                          "set as `background` for correct values.")
        elif isinstance(background, (int, np.integer)):
            U, N = None, int(background)
        else:
            U = set(background)
            N = len(U)

        P = pd.DataFrame(index=rows, columns=cols, dtype=float)
        for ca in rows:
            A = set(dict_a[ca]) & U if U is not None else set(dict_a[ca])
            for cb in cols:
                B = set(dict_b[cb]) & U if U is not None else set(dict_b[cb])
                fe, p = enrichment(A, B, N)
                M.loc[ca, cb] = fe
                P.loc[ca, cb] = p
        # BH-FDR across the whole grid (valid tests only); invalid cells -> 1.
        fe_flat = M.values.astype(float).ravel()
        p_flat = P.values.astype(float).ravel()
        valid = np.isfinite(fe_flat)
        fdr_flat = np.ones_like(p_flat)
        if valid.any():
            fdr_flat[valid] = _bh_fdr(p_flat[valid])
        FDR = pd.DataFrame(fdr_flat.reshape(M.shape), index=rows, columns=cols)
    else:
        _fn = overlap_coefficient if metric == 'overlap' else jaccard
        for ca in rows:
            for cb in cols:
                M.loc[ca, cb] = _fn(dict_a[ca], dict_b[cb])

    # Colour scale / colormap per metric.
    if metric == 'enrichment':
        cmap = 'RdBu_r'
        finite = M.values[np.isfinite(M.values)]
        if vmax is None:
            absmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
            vmax = absmax if absmax > 0 else 1.0
        if vmin is None:
            vmin = -vmax
        cbar_label = 'log2 fold-enrichment'
        annot_color = 'black'
    else:
        cmap = 'viridis'
        if vmin is None:
            vmin = 0
        if vmax is None:
            data_max = float(np.nanmax(M.values)) if M.size else 0.0
            vmax = data_max if data_max > 0 else 1.0
        cbar_label = 'Overlap coefficient' if metric == 'overlap' else 'Jaccard index'
        annot_color = 'white'

    fig, ax = plt.subplots(figsize=(figsize_scale * len(cols) + 2,
                                    figsize_scale * len(rows) + 2))
    sns.heatmap(M, cmap=cmap, vmin=vmin, vmax=vmax, square=True, ax=ax,
                cbar=False, linewidths=0, linecolor='none')
    ax.grid(False)

    # Annotation: diagonal overlap counts; for enrichment also mark FDR-significant cells.
    if metric == 'enrichment':
        for i, ca in enumerate(rows):
            for j, cb in enumerate(cols):
                sig = bool(FDR.loc[ca, cb] < fdr_threshold)
                if ca == cb:
                    A = set(dict_a[ca]) & U if U is not None else set(dict_a[ca])
                    B = set(dict_b[cb]) & U if U is not None else set(dict_b[cb])
                    txt = f"{len(A & B)}{'*' if sig else ''}"
                    ax.text(j + 0.5, i + 0.5, txt, ha='center', va='center',
                            fontsize=8, color=annot_color)
                elif sig:
                    ax.text(j + 0.5, i + 0.5, '*', ha='center', va='center',
                            fontsize=10, color=annot_color)
    else:
        for c in [c for c in rows if c in set(cols)]:
            i = rows.index(c)
            j = cols.index(c)
            n_overlap = len(set(dict_a[c]) & set(dict_b[c]))
            ax.text(j + 0.5, i + 0.5, str(n_overlap),
                    ha='center', va='center', fontsize=9, color=annot_color)

    # Tick labels with gene counts (original marker-set sizes)
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_xticklabels(
        [f'{c}\n(n={len(dict_b[c])})' for c in cols],
        rotation=90, fontsize=10
    )
    ax.set_yticklabels(
        [f'{c}\n(n={len(dict_a[c])})' for c in rows],
        rotation=0, fontsize=10
    )

    # Colorbar at bottom-right corner
    cax = inset_axes(ax, width='4%', height='30%', loc='lower right',
                     bbox_to_anchor=(0.08, 0.02, 1, 1),
                     bbox_transform=ax.transAxes, borderpad=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.grid(False)
    if metric == 'enrichment':
        ax.text(1.02, 0.45, f'* FDR < {fdr_threshold}', transform=ax.transAxes,
                fontsize=8, va='center')

    ax.set_xlabel(name_b if name_b else 'Dict B cell types')
    ax.set_ylabel(name_a if name_a else 'Dict A cell types')
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')

    plt.show()

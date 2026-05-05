"""Embedding scatter plots for SCALEX latent spaces and UMAP visualizations."""

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage as hclust, leaves_list
from scipy.spatial.distance import pdist


def _cluster_order(X: np.ndarray) -> np.ndarray:
    """Hierarchically cluster rows of X and return leaf order."""
    Z = hclust(pdist(X, metric='correlation'), method='average')
    return leaves_list(Z)


def _diagonal_order(obs_sub, X_clust: np.ndarray, cat_order, color: str) -> np.ndarray:
    """Sort cells by category order, clustering within each category."""
    groups = []
    for ct in cat_order:
        local = np.where(obs_sub[color].values == ct)[0]
        if len(local) == 0:
            continue
        if len(local) > 2:
            local = local[_cluster_order(X_clust[local])]
        groups.append(local)
    return np.concatenate(groups) if groups else np.arange(len(obs_sub))


def _subsample(indices, obs_sub, color: str, max_cells_per_type: int, rng) -> np.ndarray:
    """Subsample so each cell-type contributes at most max_cells_per_type rows."""
    if not max_cells_per_type:
        return indices
    keep = []
    for ct in obs_sub[color].unique():
        ct_idx = indices[obs_sub[color].values == ct]
        if len(ct_idx) > max_cells_per_type:
            ct_idx = rng.choice(ct_idx, max_cells_per_type, replace=False)
        keep.append(ct_idx)
    return np.sort(np.concatenate(keep))


def plot_embedding(
    adata,
    color: str = 'cell_type',
    color_map=None,
    groupby: str = 'batch',
    groups=None,
    cell_order=None,
    cond2=None,
    v2=None,
    save=None,
    legend_loc: str = 'right margin',
    legend_fontsize=None,
    legend_fontweight=None,
    sep: str = '_',
    basis: str = 'X_umap',
    size: int = 20,
    wspace: float = 0.5,
    n_cols: int = 4,
    show: bool = True,
    **kwargs,
) -> None:
    """Plot per-batch embeddings with other cells as a grey background.

    For each group in ``groupby``, cells from that group are coloured by
    ``color``; all other cells are shown in light grey as context.

    Parameters
    ----------
    adata : AnnData
        Annotated data with a UMAP (or other) embedding in ``obsm``.
    color : str, default 'cell_type'
        Column in ``obs`` used for colouring cells.
    color_map : dict | None
        Explicit ``{category: colour}`` mapping. If None, uses scanpy
        defaults or ``adata.uns['{color}_colors']``.
    groupby : str, default 'batch'
        Column in ``obs`` whose unique values define the subplots.
    groups : list | None
        Subset of ``groupby`` categories to plot. Defaults to all.
    cell_order : list | None
        Explicit ordering of ``color`` categories for the legend and
        colour assignment. Defaults to the categorical order in ``obs``.
    cond2 : str | None
        Secondary condition column for cross-filtering.
    v2 : str | None
        Value of ``cond2`` to keep.
    save : str | None
        File path to save the figure.
    legend_loc : str, default 'right margin'
        Legend placement passed to each subplot. Set to None to hide.
    legend_fontsize : int | None
        Legend font size (default 8).
    legend_fontweight : str | None
        Legend font weight (default 'normal').
    sep : str, default '_'
        Separator used in subplot titles when ``cond2`` is set.
    basis : str, default 'X_umap'
        Key in ``obsm`` for the embedding coordinates.
    size : int, default 20
        Scatter point size.
    wspace : float, default 0.5
        Horizontal spacing between subplots.
    n_cols : int, default 4
        Maximum columns in the subplot grid.
    show : bool, default True
        Call ``plt.show()`` after drawing.
    **kwargs
        Additional arguments forwarded to ``sc.pl.embedding``.
    """
    if groups is None:
        _groups = adata.obs[groupby].astype('category').cat.categories
    else:
        _groups = groups

    if cell_order is not None:
        orig_order = list(cell_order)
    else:
        orig_order = adata.obs[color].astype('category').cat.categories.tolist()

    if color_map is not None:
        _color_map = dict(color_map)
    elif f'{color}_colors' in adata.uns:
        cats = adata.obs[color].astype('category').cat.categories
        _color_map = dict(zip(cats, adata.uns[f'{color}_colors']))
    else:
        default_pal = sc.pl.palettes.default_102
        _color_map = {c: default_pal[i % len(default_pal)] for i, c in enumerate(orig_order)}
    _color_map[''] = 'lightgray'

    n_plots = len(_groups)
    n_rows = (n_plots + n_cols - 1) // n_cols
    figsize = 4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * figsize + figsize * wspace * (n_cols - 1), n_rows * figsize),
    )

    for j, ax in enumerate(axes.flatten()):
        if j < n_plots:
            b = _groups[j]
            adata.obs['tmp'] = adata.obs[color].astype(str)
            adata.obs.loc[adata.obs[groupby] != b, 'tmp'] = ''
            if cond2 is not None:
                adata.obs.loc[adata.obs[cond2] != v2, 'tmp'] = ''
                subset = adata[(adata.obs[groupby] == b) & (adata.obs[cond2] == v2)]
                present = set(subset.obs[color].dropna().astype(str).unique())
                color_groups = [c for c in orig_order if c in present]
                size = max(size, 12000 / len(subset))
            else:
                subset = adata[adata.obs[groupby] == b]
                present = set(subset.obs[color].dropna().astype(str).unique())
                color_groups = [c for c in orig_order if c in present]
                size = max(size, 12000 / len(subset))

            adata.obs['tmp'] = adata.obs['tmp'].astype('category')
            tmp_cats = set(adata.obs['tmp'].cat.categories)
            ordered = [''] + [c for c in orig_order if c in tmp_cats]
            adata.obs['tmp'] = adata.obs['tmp'].cat.reorder_categories(ordered)
            palette = [_color_map.get(c, 'lightgray') for c in adata.obs['tmp'].cat.categories]

            title = b if cond2 is None else v2 + sep + b
            ax = sc.pl.embedding(
                adata, color='tmp', basis=basis, groups=color_groups,
                ax=ax, title=title, palette=palette, size=size,
                legend_loc='none', wspace=wspace, show=False, **kwargs,
            )

            if legend_loc not in (None, 'none', False):
                from matplotlib.lines import Line2D
                _fs = legend_fontsize or 8
                handles = [
                    Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=_color_map[c], label=c, markersize=_fs)
                    for c in color_groups
                ]
                ax.legend(
                    handles=handles,
                    loc='center left',
                    bbox_to_anchor=(1.05, 0.5),
                    frameon=False,
                    prop={'size': _fs, 'weight': legend_fontweight or 'normal'},
                )

            del adata.obs['tmp']
            del adata.uns['tmp_colors']
        else:
            fig.delaxes(ax)

    plt.subplots_adjust(wspace=wspace)

    if save:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()


# Backward-compatible alias for the original name.
embedding = plot_embedding


def plot_subplots(n_panels: int, ncols: int = 3):
    """Create a subplot grid and return figure and axes.

    Parameters
    ----------
    n_panels : int
        Total number of panels to display.
    ncols : int, default 3
        Number of columns in the grid.

    Returns
    -------
    tuple[Figure, list[Axes]]
        Figure and list of active axes (empty panels removed).
    """
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).reshape(-1)
    for ax in axes[n_panels:]:
        ax.remove()
    return fig, list(axes[:n_panels])

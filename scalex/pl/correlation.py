"""Correlation-based and evaluation plots: heatmaps, confusion matrices, radar charts."""

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.cluster.hierarchy import linkage as hclust, leaves_list
from scipy.spatial.distance import pdist
from scipy.optimize import linear_sum_assignment
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score, f1_score

from scalex.pl.embedding import _cluster_order, _diagonal_order, _subsample


def _get_representation(adata, use_rep: str) -> np.ndarray:
    """Return a dense 2-D numpy array from obsm, layers, or X."""
    if use_rep in adata.obsm:
        return np.array(adata.obsm[use_rep])
    X = adata.layers[use_rep] if use_rep in adata.layers else adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    return np.array(X)


def _rgba(series, palette: dict) -> np.ndarray:
    return np.array([to_rgba(palette.get(str(v), '#888888')) for v in series])


def _pearson_cross(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson cross-correlation matrix between rows of A and rows of B."""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_n @ B_n.T


def _add_left_strip(fig, pos, rgba, label, x_offset, bar_w, fontsize):
    ax_s = fig.add_axes([pos.x0 - x_offset, pos.y0, bar_w, pos.height])
    ax_s.imshow(rgba[:, np.newaxis, :], aspect='auto', interpolation='none', origin='upper')
    ax_s.set_xticks([])
    ax_s.set_yticks([])
    ax_s.set_title(label, fontsize=fontsize, pad=4)


def _add_bottom_strip(fig, pos, rgba, label, bar_w, gap, fontsize):
    ax_s = fig.add_axes([pos.x0, pos.y0 - bar_w - gap, pos.width, bar_w])
    ax_s.imshow(rgba[np.newaxis, :, :], aspect='auto', interpolation='none', origin='upper')
    ax_s.set_xticks([])
    ax_s.set_yticks([])
    ax_s.set_xlabel(label, fontsize=fontsize, labelpad=4)


def _render_heatmap(fig, ax, corr_ord, row_obs, col_obs,
                    color, batch, color_map, batch_palette,
                    cmap, vmin, vmax, fontsize,
                    title, row_label, col_label, compare, cat_order=None):
    """Render heatmap, color strips, and legend onto fig/ax."""
    sns.heatmap(corr_ord, ax=ax, cmap=cmap, center=0, vmin=vmin, vmax=vmax,
                xticklabels=False, yticklabels=False,
                square=(not compare),
                cbar_kws={'shrink': 0.25, 'pad': 0.02,
                          'anchor': (1.0, 0.0), 'label': 'Pearson r'})
    ax.set_title(title, fontsize=fontsize + 2)
    if row_label:
        ax.set_ylabel(row_label, fontsize=fontsize)
    if col_label:
        ax.set_xlabel(col_label, fontsize=fontsize)

    pos = ax.get_position()
    bar_w, gap = 0.015, 0.004

    if not compare and batch is not None:
        _add_left_strip(fig, pos, _rgba(row_obs[batch], batch_palette),
                        batch, bar_w + gap, bar_w, fontsize)
        color_x_offset = bar_w * 2 + gap * 2
    else:
        color_x_offset = bar_w + gap
    strip_label = color if not compare else f'{color} ({row_label})'
    _add_left_strip(fig, pos, _rgba(row_obs[color], color_map),
                    strip_label, color_x_offset, bar_w, fontsize)

    if compare:
        _add_bottom_strip(fig, pos, _rgba(col_obs[color], color_map),
                          f'{color} ({col_label})', bar_w, gap, fontsize)

    if not compare:
        ct_handles = [Patch(color=c, label=k) for k, c in color_map.items()]
        bat_handles = [Patch(color=c, label=k) for k, c in batch_palette.items()]
        legend_handles = ct_handles
        if bat_handles:
            legend_handles += [Patch(visible=False, label='')] + bat_handles
    else:
        cats_row = set(row_obs[color].values)
        cats_col = set(col_obs[color].values)
        b1_h = [Patch(visible=False, label=row_label)] + [
            Patch(color=color_map.get(k, '#888888'), label=k)
            for k in cat_order if k in cats_row]
        b2_h = [Patch(visible=False, label=col_label)] + [
            Patch(color=color_map.get(k, '#888888'), label=k)
            for k in cat_order if k in cats_col]
        legend_handles = b1_h + [Patch(visible=False, label='')] + b2_h
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1),
              loc='upper left', frameon=False, fontsize=fontsize)


def plot_meta(
    adata,
    use_rep: str = 'latent',
    color: str = 'celltype',
    batch: str = 'batch',
    colors=None,
    cmap: str = 'Blues',
    vmax: float = 1,
    vmin: float = 0,
    mask: bool = True,
    annot: bool = False,
    save=None,
    fontsize: int = 8,
) -> None:
    """Plot Pearson correlation heatmap of cell-type pseudobulk representations.

    One pseudobulk vector per (batch, cell-type) combination is computed;
    the resulting matrix is clustered and displayed as a heatmap.

    Parameters
    ----------
    adata : AnnData
        Integrated data with ``obsm[use_rep]``.
    use_rep : str, default 'latent'
        Embedding key in ``obsm`` or ``layers`` to use.
    color : str, default 'celltype'
        Column in ``obs`` with cell-type labels.
    batch : str, default 'batch'
        Column in ``obs`` denoting batch membership.
    colors : list | None
        Colour list for batches. Defaults to ``tab10`` palette.
    cmap : str, default 'Blues'
        Colormap for the heatmap.
    vmax, vmin : float, default 1 and 0
        Colour scale limits.
    mask : bool, default True
        Mask the upper triangle of the correlation matrix.
    annot : bool, default False
        Annotate cells with numerical values.
    save : str | None
        File path to save the figure.
    fontsize : int, default 8
        Tick label font size.
    """
    meta, name, color_list = [], [], []
    adata.obs[color] = adata.obs[color].astype('category')
    batches = np.unique(adata.obs[batch])
    if colors is None:
        colors = sns.color_palette("tab10", len(batches))
    for i, b in enumerate(batches):
        for cat in adata.obs[color].cat.categories:
            index = np.where((adata.obs[color] == cat) & (adata.obs[batch] == b))[0]
            if len(index) > 0:
                if use_rep and use_rep in adata.obsm:
                    meta.append(adata.obsm[use_rep][index].mean(0))
                elif use_rep and use_rep in adata.layers:
                    meta.append(adata.layers[use_rep][index].mean(0))
                else:
                    meta.append(adata.X[index].mean(0))
                name.append(cat)
                color_list.append(colors[i])

    meta = np.stack(meta)
    plt.figure(figsize=(10, 10))
    corr = np.corrcoef(meta)
    if mask:
        mask_arr = np.zeros_like(corr)
        mask_arr[np.triu_indices_from(mask_arr, k=1)] = True
    else:
        mask_arr = False
    grid = sns.heatmap(corr, mask=mask_arr, xticklabels=name, yticklabels=name,
                       annot=annot, cmap=cmap, square=True, cbar=True, vmin=vmin, vmax=vmax)
    [tick.set_color(c) for tick, c in zip(grid.get_xticklabels(), color_list)]
    [tick.set_color(c) for tick, c in zip(grid.get_yticklabels(), color_list)]
    plt.xticks(rotation=45, horizontalalignment='right', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    if save:
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()


def plot_meta2(
    adata,
    use_rep: str = 'latent',
    color: str = 'celltype',
    batch: str = 'batch',
    color_map=None,
    figsize=(10, 10),
    cmap: str = 'Blues',
    batches=None,
    annot: bool = False,
    save=None,
    cbar: bool = True,
    keep: bool = False,
    fontsize: int = 16,
    vmin: float = 0,
    vmax: float = 1,
) -> None:
    """Plot cross-batch Pearson correlation between two batches.

    Parameters
    ----------
    adata : AnnData
        Integrated data with embedding in ``obsm[use_rep]``.
    use_rep : str, default 'latent'
        Embedding key.
    color : str, default 'celltype'
        Cell-type annotation column.
    batch : str, default 'batch'
        Batch column. Exactly two batches are expected.
    color_map : dict | None
        Category-to-colour mapping for tick label colouring.
    figsize : tuple, default (10, 10)
        Figure size.
    cmap : str, default 'Blues'
        Colormap.
    batches : list | None
        Explicit pair of batch names to compare.
    annot : bool, default False
        Annotate heatmap cells.
    save : str | None
        Path to save the figure.
    cbar : bool, default True
        Show colour bar.
    keep : bool, default False
        Expand the matrix to include all categories (filling zeros for
        absent category pairs).
    fontsize : int, default 16
        Label font size.
    vmin, vmax : float, default 0 and 1
        Colour scale limits.
    """
    mpl.rcParams['axes.grid'] = False

    meta, name = [], []
    adata.obs[color] = adata.obs[color].astype('category')
    if batches is None:
        batches = np.unique(adata.obs[batch])

    for i, b in enumerate(batches):
        for cat in adata.obs[color].cat.categories:
            index = np.where((adata.obs[color] == cat) & (adata.obs[batch] == b))[0]
            if len(index) > 0:
                if use_rep and use_rep in adata.obsm:
                    meta.append(adata.obsm[use_rep][index].mean(0))
                elif use_rep and use_rep in adata.layers:
                    meta.append(adata.layers[use_rep][index].mean(0))
                else:
                    meta.append(adata.X[index].mean(0))
                name.append(cat)

    meta = np.stack(meta)
    plt.figure(figsize=figsize)
    corr = np.corrcoef(meta)

    xticklabels = adata[adata.obs[batch] == batches[0]].obs[color].cat.categories
    yticklabels = adata[adata.obs[batch] == batches[1]].obs[color].cat.categories
    corr = corr[len(xticklabels):, :len(xticklabels)]
    if keep:
        categories = adata.obs[color].cat.categories
        corr_ = np.zeros((len(categories), len(categories)))
        x_ind = [i for i, k in enumerate(categories) if k in xticklabels]
        y_ind = [i for i, k in enumerate(categories) if k in yticklabels]
        corr_[np.ix_(y_ind, x_ind)] = corr
        corr = corr_
        xticklabels, yticklabels = categories, categories
    grid = sns.heatmap(corr, xticklabels=xticklabels, yticklabels=yticklabels,
                       annot=annot, cmap=cmap, square=True, cbar=cbar, vmin=vmin, vmax=vmax)

    if color_map is not None:
        [tick.set_color(color_map[tick.get_text()]) for tick in grid.get_xticklabels()]
        [tick.set_color(color_map[tick.get_text()]) for tick in grid.get_yticklabels()]
    plt.xticks(rotation=45, horizontalalignment='right', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel(batches[0], fontsize=fontsize)
    plt.ylabel(batches[1], fontsize=fontsize)

    if save:
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()


def plot_expr(adata, gene: str, groupby: str = 'batch', category=None, n_cols: int = 5, size: int = 40, **kwargs) -> None:
    """Plot gene expression per batch on a UMAP.

    Parameters
    ----------
    adata : AnnData
        Single-cell data with ``obsm['X_umap']``.
    gene : str
        Gene name to visualise.
    groupby : str, default 'batch'
        Batch column in ``obs``.
    category : str | None
        Value in ``obs['category']`` to filter cells.
    n_cols : int, default 5
        Columns in the subplot grid.
    size : int, default 40
        Dot size.
    **kwargs
        Forwarded to ``sc.pl.umap``.
    """
    vmax = adata[:, gene].X.max() if adata.raw is None else adata.raw[:, gene].X.max()
    batches = np.unique(adata.obs.loc[adata.obs['category'] == category, groupby])
    n_plots = len(batches)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    for i, ax in enumerate(axes.flatten()):
        if i < n_plots:
            mask = adata.obs[groupby] == batches[i]
            s = max(size, 6000 / len(adata[mask]))
            sc.pl.umap(adata, color=None, size=s, show=False, alpha=0.2, ax=ax)
            sc.pl.umap(adata[mask], color=gene, size=s, vmax=vmax, ax=ax,
                       show=False, title=batches[i], **kwargs)
        else:
            fig.delaxes(ax)
    plt.show()


def reassign_cluster_with_ref(Y_pred, Y):
    """Reassign cluster labels to match a reference using the Hungarian algorithm.

    Parameters
    ----------
    Y_pred : np.ndarray
        Predicted cluster labels (integer array).
    Y : np.ndarray
        Reference labels (integer array).

    Returns
    -------
    y_pred : np.ndarray
        Remapped predicted labels aligned to ``Y``.
    ind : tuple
        Assignment indices from ``linear_sum_assignment``.
    """
    def _reassign(y_pred, index):
        y_ = np.zeros_like(y_pred)
        for i, j in index:
            y_[np.where(y_pred == i)] = j
        return y_

    from scipy.optimize import linear_sum_assignment as linear_assignment
    if Y_pred.size != Y.size:
        raise ValueError(
            f"Y_pred and Y must have the same number of elements, "
            f"got {Y_pred.size} and {Y.size}."
        )
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return _reassign(Y_pred, ind), ind


def plot_confusion(y, y_pred, save=None, cmap: str = 'Blues'):
    """Plot a normalised confusion matrix and return clustering metrics.

    Parameters
    ----------
    y : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    save : str | None
        Path to save the figure.
    cmap : str, default 'Blues'
        Colormap for the heatmap.

    Returns
    -------
    tuple[float, float, float]
        F1 score (micro), NMI score, ARI score.
    """
    y_class = np.unique(y)
    pred_class = np.unique(y_pred)
    cm = confusion_matrix(y, y_pred)
    f1 = f1_score(y, y_pred, average='micro')
    nmi = normalized_mutual_info_score(y, y_pred)
    ari = adjusted_rand_score(y, y_pred)
    cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]

    plt.figure(figsize=(14, 14))
    sns.heatmap(cm, xticklabels=y_class, yticklabels=pred_class,
                cmap=cmap, square=True, cbar=False, vmin=0, vmax=1)
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.yticks(fontsize=14, rotation=0)
    plt.ylabel('Leiden cluster', fontsize=18)

    if save:
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()
    return f1, nmi, ari


def plot_corr_clustermap(
    adata,
    use_rep: str = 'latent',
    color: str = 'celltype',
    batch=None,
    color_map=None,
    figsize=(12, 12),
    cmap: str = 'RdBu_r',
    batches=None,
    save=None,
    fontsize: int = 10,
    vmin: float = -1,
    vmax: float = 1,
    compare: bool = False,
    cat_order=None,
    cluster_by_corr: bool = False,
    max_cells_per_type: int = 500,
    seed: int = 0,
    transpose: bool = False,
):
    """Plot a cell-cell Pearson correlation heatmap with hierarchical clustering.

    Parameters
    ----------
    adata : AnnData
        Integrated data.
    use_rep : str, default 'latent'
        Embedding key in ``obsm`` or ``layers``.
    color : str, default 'celltype'
        Cell-type annotation column.
    batch : str | None
        Batch column for left-strip annotation.
    color_map : dict | None
        Category-to-colour mapping.
    figsize : tuple, default (12, 12)
        Figure size.
    cmap : str, default 'RdBu_r'
        Colormap.
    batches : list | None
        Pair of batch names for cross-batch comparison (``compare=True``).
    save : str | None
        Path to save the figure.
    fontsize : int, default 10
        Label font size.
    vmin, vmax : float, default -1 and 1
        Colour scale limits.
    compare : bool, default False
        If True, compute a cross-batch N1×N2 heatmap instead of N×N.
    cat_order : list | None
        Explicit cell-type ordering for diagonal alignment.
    cluster_by_corr : bool, default False
        Cluster axes by correlation similarity (hierarchical).
    max_cells_per_type : int, default 500
        Maximum cells per cell type (random subsampling).
    seed : int, default 0
        Random seed for subsampling.
    transpose : bool, default False
        Transpose the cross-batch matrix (swap rows and columns).

    Returns
    -------
    matplotlib.figure.Figure
    """
    mpl.rcParams['axes.grid'] = False

    if batch is not None and batches is None:
        batches = list(adata.obs[batch].astype('category').cat.categories)

    X_all = _get_representation(adata, use_rep)

    if color_map is None:
        cats = adata.obs[color].astype('category').cat.categories
        color_map = {k: mpl.colors.rgb2hex(c)
                     for k, c in zip(cats, sns.color_palette('tab10', len(cats)))}

    batch_palette = (
        dict(zip(batches, ['#4878CF', '#D65F5F', '#5cb85c', '#f0ad4e'][:len(batches)]))
        if batch is not None else {}
    )

    rng = np.random.default_rng(seed)

    if not compare:
        all_idx = _subsample(np.arange(len(adata)), adata.obs, color, max_cells_per_type, rng)
        X_sub = X_all[all_idx]
        obs_sub = adata.obs.iloc[all_idx]
        cat_order_all = adata.obs[color].astype('category').cat.categories
        corr = np.corrcoef(X_sub)
        order = (_cluster_order(corr) if cluster_by_corr
                 else _diagonal_order(obs_sub, X_sub, cat_order_all, color))
        corr_ord = corr[np.ix_(order, order)]
        row_obs = obs_sub.iloc[order]

        fig, ax = plt.subplots(figsize=figsize)
        _render_heatmap(fig, ax, corr_ord, row_obs=row_obs, col_obs=row_obs,
                        color=color, batch=batch, color_map=color_map,
                        batch_palette=batch_palette, cmap=cmap, vmin=vmin, vmax=vmax,
                        fontsize=fontsize,
                        title='Cell–cell Correlation (Hierarchically Clustered)',
                        row_label=None, col_label=None, compare=False)

    else:
        b1, b2 = batches[0], batches[1]
        idx1 = np.where(adata.obs[batch] == b1)[0]
        idx2 = np.where(adata.obs[batch] == b2)[0]
        X1, obs1 = X_all[idx1], adata.obs.iloc[idx1]
        X2, obs2 = X_all[idx2], adata.obs.iloc[idx2]

        sub1 = _subsample(np.arange(len(obs1)), obs1, color, max_cells_per_type, rng)
        sub2 = _subsample(np.arange(len(obs2)), obs2, color, max_cells_per_type, rng)
        X1, obs1 = X1[sub1], obs1.iloc[sub1]
        X2, obs2 = X2[sub2], obs2.iloc[sub2]

        if cat_order is None:
            cat_order = adata.obs[color].astype('category').cat.categories

        corr_full = _pearson_cross(X1, X2)

        if cluster_by_corr:
            order1 = _cluster_order(corr_full)
            order2 = _cluster_order(corr_full.T)
            obs1_tmp = obs1.iloc[order1]
            obs2_tmp = obs2.iloc[order2]
            corr_tmp = corr_full[np.ix_(order1, order2)]

            types1 = [ct for ct in cat_order if ct in set(obs1_tmp[color].values)]
            types2 = [ct for ct in cat_order if ct in set(obs2_tmp[color].values)]
            C = np.array([
                [corr_tmp[np.ix_(
                    np.where(obs1_tmp[color].values == t1)[0],
                    np.where(obs2_tmp[color].values == t2)[0],
                )].mean()
                for t2 in types2]
                for t1 in types1
            ])
            _, col_perm = linear_sum_assignment(-C)
            unmatched = [j for j in range(len(types2)) if j not in set(col_perm)]
            new_col_order = np.concatenate([
                np.where(obs2_tmp[color].values == types2[j])[0]
                for j in list(col_perm) + unmatched
            ])
            order2 = order2[new_col_order]
        else:
            order1 = _diagonal_order(obs1, X1, cat_order, color)
            order2 = _diagonal_order(obs2, X2, cat_order, color)

        corr_ord = corr_full[np.ix_(order1, order2)]
        obs1_ord, obs2_ord = obs1.iloc[order1], obs2.iloc[order2]

        row_b, col_b = (b2, b1) if transpose else (b1, b2)
        row_obs, col_obs = (obs2_ord, obs1_ord) if transpose else (obs1_ord, obs2_ord)
        heatmap_data = corr_ord.T if transpose else corr_ord

        fig, ax = plt.subplots(figsize=figsize)
        _render_heatmap(fig, ax, heatmap_data, row_obs=row_obs, col_obs=col_obs,
                        color=color, batch=batch, color_map=color_map,
                        batch_palette=batch_palette, cmap=cmap, vmin=vmin, vmax=vmax,
                        fontsize=fontsize,
                        title=f'Cell Correlation: {row_b} (rows) vs {col_b} (cols)',
                        row_label=f'{row_b} cells', col_label=f'{col_b} cells',
                        compare=True, cat_order=cat_order)

    if save:
        fig.savefig(save, bbox_inches='tight')
    plt.show()
    return fig


def plot_radar(df, save=None, vmax: float = 1) -> None:
    """Plot a radar (spider) chart from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Rows are samples/cell-types; columns are axes of the radar chart.
    save : str | None
        Path to save the figure.
    vmax : float, default 1
        Values are clipped to [-vmax, vmax] before plotting.
    """
    df = df.clip(lower=-vmax, upper=vmax).copy()
    categories = df.columns.tolist()
    labels = df.index.tolist()
    data = df.values.tolist()
    N = len(categories)
    M = len(labels)
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ][:M]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for values, label, color in zip(data, labels, colors):
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_rlabel_position(30)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], color="grey", size=8)
    plt.ylim(0, 1)
    plt.legend(loc="right", bbox_to_anchor=(1.4, 0.5))
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()

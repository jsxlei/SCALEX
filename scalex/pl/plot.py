#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Thu 16 Jul 2020 07:24:49 PM CST

# File Name: plot.py
# Description:

"""
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

            
            
def embedding(
        adata,
        color='cell_type',
        color_map=None,
        groupby='batch',
        groups=None,
        cell_order=None,
        cond2=None,
        v2=None,
        save=None,
        legend_loc='right margin',
        legend_fontsize=None,
        legend_fontweight=None,
        sep='_',
        basis='X_umap',
        size=20,
        wspace=0.5,
        n_cols=4,
        show=True,
        **kwargs
    ):
    """
    plot separated embeddings with others as background

    Parameters
    ----------
    adata
        AnnData
    color
        meta information to be shown
    color_map
        specific color map
    groupby
        condition which is based-on to separate
    groups
        specific groups to be shown
    cell_order
        explicit ordering for color category assignment
    cond2
        another targeted condition
    v2
        another targeted values of another condition
    basis
        embeddings used to visualize, default is X_umap for UMAP
    size
        dot size on the embedding
    """
    if groups is None:
        _groups = adata.obs[groupby].astype('category').cat.categories
    else:
        _groups = groups

    if cell_order is not None:
        orig_order = list(cell_order)
    else:
        orig_order = adata.obs[color].astype('category').cat.categories.tolist()

    # Build a globally-consistent color map so colors are stable across subplots
    if color_map is not None:
        _color_map = dict(color_map)
    elif f'{color}_colors' in adata.uns:
        cats = adata.obs[color].astype('category').cat.categories
        _color_map = dict(zip(cats, adata.uns[f'{color}_colors']))
    else:
        default_pal = sc.pl.palettes.default_102
        _color_map = {c: default_pal[i % len(default_pal)] for i, c in enumerate(orig_order)}
    _color_map[''] = 'lightgray'

    # Create subplots
    n_plots = len(_groups)
    n_rows = (n_plots + n_cols - 1) // n_cols
    figsize = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize + figsize * wspace * (n_cols - 1), n_rows * figsize))

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

            ax = sc.pl.embedding(adata, color='tmp', basis=basis, groups=color_groups, ax=ax, title=title,
                    palette=palette, size=size, legend_loc='none', wspace=wspace, show=False, **kwargs)

            # Manually add legend in cell_order order (bypasses scanpy's internal ordering)
            if legend_loc not in (None, 'none', False):
                from matplotlib.lines import Line2D
                _fs = legend_fontsize or 8
                handles = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=_color_map[c],
                           label=c, markersize=_fs)
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


def plot_expr(adata, gene, groupby='batch', category=None, n_cols=5, size=40, **kwargs):
    vmax = adata[:, gene].X.max() if adata.raw is None else adata.raw[:, gene].X.max()
    batches = np.unique(adata.obs.loc[adata.obs['category'] == category, groupby])
    n_plots = len(batches)  
    n_rows = (n_plots + n_cols - 1) // n_cols  # Calculate number of rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    for i, ax in enumerate(axes.flatten()):
        if i < n_plots:
            current_batch_mask = adata.obs[groupby] == batches[i] #if category is None else (adata.obs[groupby] == batches[i]) & (adata.obs['category'] == category)
            size = max(size, 6000/len(adata[current_batch_mask]))
    
            # Plot all cells in gray as the background
            sc.pl.umap(adata, color=None, size=size, show=False, alpha=0.2, ax=ax)
        
            # Plot only the current batch with gene expression
            sc.pl.umap(adata[current_batch_mask], color=gene, size=size, vmax=vmax, ax=ax, show=False, title=batches[i],  **kwargs)
        else:
            fig.delaxes(ax)
    # Show the final overlayed plot
    # plt.suptitle(gene)
    plt.show()

def plot_meta(
        adata, 
        use_rep='latent', 
        color='celltype', 
        batch='batch', 
        colors=None, 
        cmap='Blues', 
        vmax=1, 
        vmin=0, 
        mask=True,
        annot=False, 
        save=None, 
        fontsize=8
    ):
    """
    Plot meta correlations among batches
    
    Parameters
    ----------
    adata
        AnnData
    use_rep
        the cell representations or embeddings used to calculate the correlations, default is `latent` generated by `SCALEX`
    batch
        the meta information based-on, default is batch
    colors
        colors for each batch
    cmap
        color map for information to be shown
    vmax
        max value
    vmin
        min value
    mask
        value to be masked
    annot
        show specific values
    save
        save the figure
    fontsize
        font size
    """
    meta = []
    name = []
    color_list = []

    adata.obs[color] = adata.obs[color].astype('category')
    batches = np.unique(adata.obs[batch])
    if colors is None:
        colors = sns.color_palette("tab10", len(np.unique(adata.obs[batch])))
    for i,b in enumerate(batches):
        for cat in adata.obs[color].cat.categories:
            index = np.where((adata.obs[color]==cat) & (adata.obs[batch]==b))[0]
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
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask, k=1)] = True
    grid = sns.heatmap(corr, mask=mask, xticklabels=name, yticklabels=name, annot=annot, # name -> []
                cmap=cmap, square=True, cbar=True, vmin=vmin, vmax=vmax)
    [ tick.set_color(c) for tick,c in zip(grid.get_xticklabels(),color_list) ]
    [ tick.set_color(c) for tick,c in zip(grid.get_yticklabels(),color_list) ]
    plt.xticks(rotation=45, horizontalalignment='right', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    if save:
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()
        

def plot_meta2(
        adata, 
        use_rep='latent', 
        color='celltype', 
        batch='batch', 
        color_map=None, 
        figsize=(10, 10), 
        cmap='Blues',
        batches=None, 
        annot=False, 
        save=None, 
        cbar=True, 
        keep=False, 
        fontsize=16, 
        vmin=0, 
        vmax=1
    ):
    """
    Plot meta correlations between two batches
    
    Parameters
    ----------
    adata
        AnnData
    use_rep
        the cell representations or embeddings used to calculate the correlations, default is `latent` generated by `SCALEX`
    batch
        the meta information based-on, default is batch
    colors
        colors for each batch
    cmap
        color map for information to be shown
    vmax
        max value
    vmin
        min value
    mask
        value to be masked
    annot
        show specific values
    save
        save the figure
    fontsize
        font size
    """
    import matplotlib as mpl
    mpl.rcParams['axes.grid'] = False
    # mpl.rcParams.update(mpl.rcParamsDefault)

    meta = []
    name = []

    adata.obs[color] = adata.obs[color].astype('category')
    if batches is None:
        batches = np.unique(adata.obs[batch]);#print(batches)

    for i,b in enumerate(batches):
        for cat in adata.obs[color].cat.categories:
            index = np.where((adata.obs[color]==cat) & (adata.obs[batch]==b))[0]
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
    
    xticklabels = adata[adata.obs[batch]==batches[0]].obs[color].cat.categories
    yticklabels = adata[adata.obs[batch]==batches[1]].obs[color].cat.categories
#     print(len(xticklabels), len(yticklabels))
    corr = corr[len(xticklabels):, :len(xticklabels)] #;print(corr.shape)
    if keep:
        categories = adata.obs[color].cat.categories
        corr_ = np.zeros((len(categories), len(categories)))
        x_ind = [i for i,k in enumerate(categories) if k in xticklabels]
        y_ind = [i for i,k in enumerate(categories) if k in yticklabels]
        corr_[np.ix_(y_ind, x_ind)] = corr
        corr = corr_
        xticklabels, yticklabels = categories, categories
#         xticklabels, yticklabels = [], []
    grid = sns.heatmap(corr, xticklabels=xticklabels, yticklabels=yticklabels, annot=annot,
                cmap=cmap, square=True, cbar=cbar, vmin=vmin, vmax=vmax)

    if color_map is not None:
        [ tick.set_color(color_map[tick.get_text()]) for tick in grid.get_xticklabels() ]
        [ tick.set_color(color_map[tick.get_text()]) for tick in grid.get_yticklabels() ]
    plt.xticks(rotation=45, horizontalalignment='right', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel(batches[0], fontsize=fontsize)
    plt.ylabel(batches[1], fontsize=fontsize)
    
    if save:
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()
        
        
import seaborn as sns
from scipy.cluster.hierarchy import linkage as hclust, leaves_list
from scipy.spatial.distance import pdist
from scipy.optimize import linear_sum_assignment
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import matplotlib as mpl


def _get_representation(adata, use_rep):
    """Return a dense 2-D numpy array from obsm, layers, or X."""
    if use_rep in adata.obsm:
        return np.array(adata.obsm[use_rep])
    X = adata.layers[use_rep] if use_rep in adata.layers else adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    return np.array(X)


def _subsample(indices, obs_sub, color, max_cells_per_type, rng):
    """Subsample so each cell-type has at most max_cells_per_type rows."""
    if not max_cells_per_type:
        return indices
    keep = []
    for ct in obs_sub[color].unique():
        ct_idx = indices[obs_sub[color].values == ct]
        if len(ct_idx) > max_cells_per_type:
            ct_idx = rng.choice(ct_idx, max_cells_per_type, replace=False)
        keep.append(ct_idx)
    return np.sort(np.concatenate(keep))


def _cluster_order(X):
    Z = hclust(pdist(X, metric='correlation'), method='average')
    return leaves_list(Z)


def _diagonal_order(obs_sub, X_clust, cat_order, color):
    """Sort by category order; cluster within each type."""
    groups = []
    for ct in cat_order:
        local = np.where(obs_sub[color].values == ct)[0]
        if len(local) == 0:
            continue
        if len(local) > 2:
            local = local[_cluster_order(X_clust[local])]
        groups.append(local)
    return np.concatenate(groups) if groups else np.arange(len(obs_sub))


def _rgba(series, palette):
    return np.array([to_rgba(palette.get(str(v), '#888888')) for v in series])


def _pearson_cross(A, B):
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

    # ax.get_position() is valid immediately after plt.subplots — no canvas.draw() needed
    pos = ax.get_position()
    bar_w, gap = 0.015, 0.004

    # left strip
    if not compare and batch is not None:
        _add_left_strip(fig, pos, _rgba(row_obs[batch], batch_palette),
                        batch, bar_w + gap, bar_w, fontsize)
        color_x_offset = bar_w * 2 + gap * 2
    else:
        color_x_offset = bar_w + gap
    strip_label = color if not compare else f'{color} ({row_label})'
    _add_left_strip(fig, pos, _rgba(row_obs[color], color_map),
                    strip_label, color_x_offset, bar_w, fontsize)

    # bottom strip (compare mode only)
    if compare:
        _add_bottom_strip(fig, pos, _rgba(col_obs[color], color_map),
                          f'{color} ({col_label})', bar_w, gap, fontsize)

    # legend
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


def plot_corr_clustermap(
    adata,
    use_rep='latent',
    color='celltype',
    batch=None,
    color_map=None,
    figsize=(12, 12),
    cmap='RdBu_r',
    batches=None,
    save=None,
    fontsize=10,
    vmin=-1,
    vmax=1,
    compare=False,
    cat_order=None,
    cluster_by_corr=False,
    max_cells_per_type=500,
    seed=0,
    transpose=False,
):
    """
    Plot individual cell Pearson correlation heatmap with hierarchical clustering.

    compare=False : single N×N heatmap, all cells clustered together.
    compare=True  : cross-batch N1×N2 heatmap; rows = batches[0], cols = batches[1].
                    Both axes sorted by shared cell-type order + within-type clustering
                    → same-type blocks on the diagonal.
    cat_order     : optional list of cell-type names specifying the diagonal sort order.
                    Inferred from categories if None.
    cluster_by_corr: cluster axes by similarity (hierarchical); in compare mode, optimally
                    aligns col-type blocks to row-type blocks using the Hungarian algorithm.
    max_cells_per_type: max cells per cell-type (default 500); random subsampling applied
                    before computing correlations.
    seed          : random seed for subsampling reproducibility (default 0).
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

    # ── single heatmap ───────────────────────────────────────────────────────────
    if not compare:
        all_idx = _subsample(np.arange(len(adata)), adata.obs, color, max_cells_per_type, rng)
        X_sub   = X_all[all_idx]
        obs_sub = adata.obs.iloc[all_idx]

        cat_order_all = adata.obs[color].astype('category').cat.categories
        corr  = np.corrcoef(X_sub)
        order = (_cluster_order(corr) if cluster_by_corr
                 else _diagonal_order(obs_sub, X_sub, cat_order_all, color))
        corr_ord = corr[np.ix_(order, order)]
        row_obs  = obs_sub.iloc[order]

        fig, ax = plt.subplots(figsize=figsize)
        _render_heatmap(fig, ax, corr_ord, row_obs=row_obs, col_obs=row_obs,
                        color=color, batch=batch, color_map=color_map,
                        batch_palette=batch_palette, cmap=cmap, vmin=vmin, vmax=vmax,
                        fontsize=fontsize,
                        title='Cell–cell Correlation (Hierarchically Clustered)',
                        row_label=None, col_label=None, compare=False)

    # ── cross-batch comparison ───────────────────────────────────────────────────
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

        row_b, col_b     = (b2, b1) if transpose else (b1, b2)
        row_obs, col_obs = (obs2_ord, obs1_ord) if transpose else (obs1_ord, obs2_ord)
        heatmap_data     = corr_ord.T if transpose else corr_ord

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
        
from sklearn.metrics import confusion_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score

def reassign_cluster_with_ref(Y_pred, Y):
    """
    Reassign cluster to reference labels
    
    Parameters
    ----------
    Y_pred: predict y classes
    Y: true y classes
    
    Returns
    -------
    f1_score: clustering f1 score
    y_pred: reassignment index predict y classes
    indices: classes assignment
    """
    def reassign_cluster(y_pred, index):
        y_ = np.zeros_like(y_pred)
        for i, j in index:
            y_[np.where(y_pred==i)] = j
        return y_
#     from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
#     print(Y_pred.size, Y.size)
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)

    return reassign_cluster(Y_pred, ind), ind


def plot_confusion(y, y_pred, save=None, cmap='Blues'):
    """
    Plot confusion matrix
    
    Parameters
    ----------
    y
        ground truth labels
    y_pred 
        predicted labels
    save
        save the figure
    cmap
        color map
        
    Return
    ------
    F1 score
    NMI score
    ARI score
    """
    
    y_class, pred_class_ = np.unique(y), np.unique(y_pred)

    cm = confusion_matrix(y, y_pred)
    f1 = f1_score(y, y_pred, average='micro')
    nmi = normalized_mutual_info_score(y, y_pred)
    ari = adjusted_rand_score(y, y_pred)
    
    cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]

    plt.figure(figsize=(14, 14))
    sns.heatmap(cm, xticklabels=y_class, yticklabels=pred_class,
                    cmap=cmap, square=True, cbar=False, vmin=0, vmax=1)

    plt.xticks(rotation=45, horizontalalignment='right') #, fontsize=14)
    plt.yticks(fontsize=14, rotation=0)
    plt.ylabel('Leiden cluster', fontsize=18)
    
    if save:
        plt.save(save, bbox_inches='tight')
    else:
        plt.show()
    
    return f1, nmi, ari


import matplotlib.pyplot as plt
import numpy as np

def plot_subplots(n_panels, ncols=3):
    """
    Plot subplots dynamically based on the number of panels and columns.

    Parameters:
        n_panels (int): Total number of panels to display.
        ncols (int): Number of columns for the layout.
    """
    # Calculate the number of rows needed
    nrows = (n_panels + ncols - 1) // ncols  # Ceiling division

    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))

    # Flatten the axes array for easy indexing
    axes = np.array(axes).reshape(-1)

    # Plot data in each panel
    for i in range(n_panels):
        ax = axes[i]
        ax.plot(np.random.rand(10), label=f'Panel {i+1}')
        ax.set_title(f'Panel {i+1}')
        ax.legend()

    # Hide empty panels
    for ax in axes[n_panels:]:
        ax.remove()

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Example usage
# plot_subplots(n_panels=10, ncols=3)


import os
from configparser import ConfigParser
genome_dir = os.path.join(os.path.expanduser("~"), '.cache', 'genome')
hg38_dir = os.path.join(genome_dir, 'hg38')
GTF_FILE =  os.path.join(hg38_dir, 'gencode.v41.annotation.gtf.gz')
TSS_FILE = os.path.join(hg38_dir, 'tss.tsv')

import pandas as pd
import pyranges as pr

def strand_specific_start_site(df):
    df = df.copy()
    if set(df["Strand"]) != set(["+", "-"]):
        raise ValueError("Not all features are strand specific!")

    pos_strand = df.query("Strand == '+'").index
    neg_strand = df.query("Strand == '-'").index
    df.loc[pos_strand, "End"] = df.loc[pos_strand, "Start"] + 1
    df.loc[neg_strand, "Start"] = df.loc[neg_strand, "End"] - 1
    return df


def parse_tss(tss_file=TSS_FILE, gtf_file=GTF_FILE, drop_duplicates: str='gene_name') -> pr.PyRanges:
    """
    The transcription start sites (TSS) for the genome.

    Returns
    -------
    DataFrame
    """
    if tss_file is None or not os.path.exists(tss_file):
        gtf = pr.read_gtf(gtf_file).df.drop_duplicates(subset=[drop_duplicates], keep='first')
        tss = strand_specific_start_site(gtf)[['Chromosome', 'Start', 'End', 'Strand', drop_duplicates]]
        if tss_file is not None:
            os.makedirs(os.path.dirname(tss_file), exist_ok=True)
            tss.to_csv(tss_file, sep='\t', index=False)
    else:
        tss = pd.read_csv(tss_file, sep='\t')
    return tss


class Track:
    def __init__(self, genome='hg38', fig_dir='./'):
        self.config = ConfigParser()
        self.spacer_kws = {"file_type": "spacer", "height": 0.5, "title": ""}
        self.config.add_section("spacer")

        self.link_kws = {
            "links_type": "arcs", "line_width": 0.5, "line_style": "solid",
            "compact_arcs_level": 2, "use_middle": False, "file_type": "links"
        }

        self.bigwig_kws = {
            "file_type": "bigwig",
            "min_value": 0,
            "max_value": 10, #'auto',
            "height": 1
        }

        self.tss = parse_tss()

        # import matplotlib.pyplot as plt
        self.cell_type_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
                            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',]

        self.fig_dir = fig_dir

    def empty_config(self):
        self.config = ConfigParser()
        self.config.add_section("spacer")

    def add_link(self, links, name, color='darkgreen', height=1.2, **kwargs):
        self.link_kws.update(kwargs)

        if not os.path.isfile(links):
            link_file = os.path.join(self.fig_dir, f"{name}.links")
            links.to_csv(link_file, sep='\t', index=None, header=None)
        else:
            link_file = links
        self.config[name] = {
            "file": link_file, 
            "title": name, 
            "color": color, 
            "height": height, 
            **link_kws
        }

    def add_bed(self, peaks, name, display="collapsed", border_color="none", labels=False, **kwargs):
        pass

    def add_bigwig(self, bigwigs, **kwargs):
        for i, (name, bigwig) in enumerate(bigwigs.items()):
            bigwig_kws = self.bigwig_kws.copy()
            bigwig_kws.update(kwargs)

            self.config[name] = {
                "file": bigwig,
                "title": name,
                "color": self.cell_type_colors[i],
                # "height": 1,
                **bigwig_kws
            }

        self.config["spacer2"] = self.spacer_kws

    def add_gene_annotation(self, gene, **kwargs):
        # gene annotation
        self.config["Genes"] = {
            "file": GTF_FILE, 
            "title": "Genes", 
            "prefered_name": "gene_name", 
            "merge_transcripts": True,
            "fontsize": 4, 
            "height": 5, 
            "labels": True, 
            "max_labels": 100,
            "all_labels_inside": True, 
            "labels_in_margin": True,
            "style": "UCSC", 
            "file_type": "gtf",
            # "max_value": max_value
        }

    def generate_track(self, region, extend=50_000, dpi=200, width=12, fontsize=4):
        with open(f"{self.fig_dir}/tracks.ini", "w") as f:
            self.config.write(f)

        if isinstance(region, str):
            region = self.get_region_with_gene(region)
        Chromosome, Start, End = region
        Start -= extend
        End += extend
        cmd = f"pyGenomeTracks --tracks {self.fig_dir}/tracks.ini --region {Chromosome}:{Start}-{End} \
                -t ' ' --dpi {dpi} --width {width} --fontSize {fontsize} \
                --outFileName {self.fig_dir}/tracks.png" # 2> /dev/null"
        import subprocess
        subprocess.run(cmd, shell=True, check=True)

    def get_region_with_gene(self, gene):
        return self.tss.query(f"gene_name == '{gene}'")[['Chromosome', 'Start', 'End']].values[0]
    
    def plot(self, gene, bigwigs, extend=50_000, dpi=200, width=12, fontsize=4, **bigwig_kws):
        self.empty_config()
        self.add_bigwig(bigwigs, **bigwig_kws)
        self.add_gene_annotation(gene)
        self.generate_track(gene, extend, dpi, width, fontsize)
        from IPython.display import Image
        return Image(filename=f"{self.fig_dir}/tracks.png")


def plot_tracks(
        gene, region, 
        peaks=None, 
        bigwigs=None,
        links={}, 
        all_link=None, 
        all_link_kwargs={},
        meta_gr=None, 
        extend=100000, 
        dpi=200,
        width=12,
        fontsize=4,
        fig_dir='./',
        bigwig_max_value='auto'
    ):
    fig_dir = os.path.join(fig_dir, gene)
    os.makedirs(fig_dir, exist_ok=True)

    config = ConfigParser()
    spacer_kws = {"file_type": "spacer", "height": 0.5, "title": ""}
    config.add_section("spacer")
    
    # links
    link_kws = {
        "links_type": "arcs", "line_width": 0.5, "line_style": "solid",
        "compact_arcs_level": 2, "use_middle": False, "file_type": "links"
    }
    for name, link in links.items():
        name = name.replace(' ', '_')
        if not os.path.isfile(link):
            link_file = os.path.join(fig_dir, f"{name}.links")
            link.to_csv(link_file, sep='\t', index=None, header=None)
        else:
            link_file = link
        config[name] = {
            "file": link_file, 
            "title": name, 
            "color": "darkgreen", 
            "height": 1.2, 
            **link_kws
        }

    if all_link is not None:
        all_links = all_link.query(f"gene == '{gene}'").copy()

        for name, kwargs in all_link_kwargs.items():
            link = all_links.loc[:, [*all_link.columns[:6], name]]
            if 'cutoff' in kwargs:
                cutoff = kwargs['cutoff']
                link = link[(link[name] > cutoff) | (link[name] < -cutoff)]
            else:
                link = link[link[name] == True]

            name = name.replace(' ', '_')
            link.to_csv(os.path.join(fig_dir, f'{name}.links'), sep='\t', index=None, header=None)
            config[name] = {
                "file": os.path.join(fig_dir, f"{name}.links"), 
                "title": name, 
                "color": "darkgreen", 
                "height": 1.2, 
                **link_kws,
                **kwargs
            }                        

        

    # peak
    bed_kws = {
        "display": "collapsed", "border_color": "none",
        "labels": False, "file_type": "bed"
    }
    if peaks is not None:
        for name, peak in peaks.items():
            peak.to_csv(os.path.join(fig_dir, f"{name}.bed"), sep='\t', index=None, header=None)
            config[name] = {
                "file": os.path.join(fig_dir, f"{name}.bed"),
                "title": name, 
                **bed_kws
            }
    
    config["spacer1"] = spacer_kws


    # atac bigwig
    cell_type_kws = {
        "file_type": "bigwig"
    }

    # import matplotlib.pyplot as plt
    cell_type_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
                        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',]
    
    if meta_gr is not None:
        cell_types = meta_gr.df.columns[3:]
        chromsizes = get_chromsize()
        chroms = meta_gr.df['Chromosome'].unique()
        if len(set(chroms) - set(chromsizes.df['Chromosome'].values)) > 0:
            import pyranges as pr
            meta_gr = meta_gr.df[meta_gr.df['Chromosome'].isin(chromsizes.df['Chromosome'].values)]
            meta_gr = pr.PyRanges(meta_gr)
        os.makedirs(os.path.join(fig_dir ,'bw'), exist_ok=True)
        for i, c in enumerate(cell_types):
            d = c
            c = c.replace(' ', '_')
            if not os.path.isfile(os.path.join(fig_dir, 'bw', f"{c}.bw")):
                meta_gr.to_bigwig(os.path.join(fig_dir, 'bw', f'{c}.bw'), chromosome_sizes=chromsizes, value_col=d, rpm=False)

            config[f"meta {c}"] = {
                "file": os.path.join(fig_dir, 'bw', f"{c}.bw"),
                "title": f"{c}",
                "color": cell_type_colors[i],
                "height": 1,
                "min_value": 0, 
                # "max_value": meta_max_value, 
                **cell_type_kws
            }

    if bigwigs is not None:
        i = 0
        for name, bigwig in bigwigs.items():
            
            config[name] = {
                "file": bigwig,
                "title": name,
                "color": cell_type_colors[i+1],
                "height": 1,
                "min_value": 0, 
                "max_value": bigwig_max_value, 
                **cell_type_kws
            }
            i+=1

    config["spacer2"] = spacer_kws
    
    if not os.path.isfile(GTF_FILE): 
        if os.path.exists(os.path.dirname(GTF_FILE)):
            os.makedirs(os.path.dirname(GTF_FILE), exist_ok=True)
        import wget
        wget.download("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.annotation.gtf.gz", GTF_FILE)
    # gene annotation
    config["Genes"] = {
        "file": GTF_FILE, 
        "title": "Genes", 
        "prefered_name": "gene_name", 
        "merge_transcripts": True,
        "fontsize": fontsize, 
        "height": 5, 
        "labels": True, 
        "max_labels": 100,
        "all_labels_inside": True, 
        "labels_in_margin": True,
        "style": "UCSC", 
        "file_type": "gtf",
        # "max_value": max_value
    }

    config["x-axis"] = {"fontsize": fontsize}
    with open(f"{fig_dir}/tracks.ini", "w") as f:
        config.write(f)

    Chromosome, Start, End = region
    Start -= extend
    End += extend
    cmd = f"pyGenomeTracks --tracks {fig_dir}/tracks.ini --region {Chromosome}:{Start}-{End} \
        -t 'Target gene: {gene}' --dpi {dpi} --width {width} --fontSize {fontsize} \
        --outFileName {fig_dir}/tracks.png" # 2> /dev/null"
    import subprocess
    subprocess.run(cmd, shell=True, check=True)

    print(f"{fig_dir}/tracks.png")


def plot_radar(df, save=None, vmax=1):
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


from scalex.pl._dotplot import dotplot
from scalex.pl._jaccard import plot_jaccard_heatmap
from scalex.pl._sankey import plot_sankey
from scalex.pl._stackedplot import plot_crosstab, plot_crosstab_stacked
from scalex.pl._heatmap import plot_heatmap
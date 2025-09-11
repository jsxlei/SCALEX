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
        cond2=None, 
        v2=None, 
        save=None, 
        legend_loc='right margin', 
        legend_fontsize=None, 
        legend_fontweight='bold', 
        sep='_', 
        basis='X_umap',
        size=30,
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

    # Create subplots
    n_plots = len(_groups)
    n_rows = (n_plots + n_cols - 1) // n_cols  # Calculate number of rows
    figsize = 4
    # wspace = 0.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize + figsize * wspace * (n_cols - 1), n_rows * figsize)) #(5*n_cols+5, 5*n_rows))

    for j, ax in enumerate(axes.flatten()):
        if j < n_plots:
            b = _groups[j]
            adata.obs['tmp'] = adata.obs[color].astype(str)
            adata.obs.loc[adata.obs[groupby]!=b, 'tmp'] = ''
            if cond2 is not None:
                adata.obs.loc[adata.obs[cond2]!=v2, 'tmp'] = ''
                groups = list(adata[(adata.obs[groupby]==b) & 
                                    (adata.obs[cond2]==v2)].obs[color].astype('category').cat.categories.values)
                size = max(size, 12000/len(adata[(adata.obs[groupby]==b) & (adata.obs[cond2]==v2)]))
            else:
                groups = list(adata[adata.obs[groupby]==b].obs[color].astype('category').cat.categories.values)
                size = max(size, 12000/len(adata[adata.obs[groupby]==b]))
            adata.obs['tmp'] = adata.obs['tmp'].astype('category')
            if color_map is not None:
                palette = [color_map[i] if i in color_map else 'gray' for i in adata.obs['tmp'].cat.categories]
            else:
                palette = None

            title = b if cond2 is None else v2+sep+b

            ax = sc.pl.embedding(adata, color='tmp', basis=basis, groups=groups, ax=ax, title=title, palette=palette, size=size, 
                    legend_loc=legend_loc, legend_fontsize=legend_fontsize, legend_fontweight=legend_fontweight, wspace=wspace, show=False, **kwargs)
            # ax.set_aspect('equal')
            # ax.set_aspect('equal', adjustable='box')
            
            del adata.obs['tmp']
            del adata.uns['tmp_colors']
        else:
            fig.delaxes(ax)

    plt.subplots_adjust(wspace=wspace)


    if save:
        plt.savefig(save, bbox_inches='tight')


def plot_expr(adata, gene, groupby='batch', category=None, n_cols=5, size=40, **kwargs):
    vmax = adata[:, gene].X.max() if adata.raw is None else adata.raw[:, gene].X.max()
    if category is not None:
        batches = np.unique(adata.obs.loc[adata.obs['category'] == category, groupby])
    else:
        batches = np.unique(adata.obs[groupby])
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
        color='cell_type', 
        batch='batch', 
        groupby='cell_type',
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

    if adata.obs[color].dtype != 'category':
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








def plot_radar(df, save=None, vmax=1):
    df = df.clip(lower=-vmax, upper=vmax).copy()
    categories = df.columns.tolist()
    labels = df.index.tolist()
    data = df.values.tolist()
    N = len(categories)
    M = len(labels)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',][:M]
    
    # Compute angle for each axis
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the circle
    
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    
    # Plot each group
    for values, label, color in zip(data, labels, colors):
        values += values[:1]  # close the polygon
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.2)
    
    # Fix axis to go in correct order and labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Add radial labels
    ax.set_rlabel_position(30)
    # plt.yticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], color="grey", size=8)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], color="grey", size=8)
    plt.ylim(0, 1)
    # plt.ylim(-vmax, vmax)
    
    # Legend
    plt.legend(loc="right", bbox_to_anchor=(1.4, 0.5))
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()







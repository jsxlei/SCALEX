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
    def __init__(self, genome='hg38', fig_dir='./', cell_type_colors=None):
        self.config = ConfigParser()
        self.spacer_kws = {"file_type": "spacer", "height": 0.5, "title": ""}
        self.line_kws = {"line_width": 0.5, "line_style": "solid", "file_type": "hlines"} 

        # self.config.add_section("spacer")

        self.link_kws = {
            "links_type": "arcs", "line_width": 0.5, "line_style": "solid",
            "compact_arcs_level": 2, "use_middle": False, "file_type": "links"
        }

        self.bigwig_kws = {
            "file_type": "bigwig",
            "min_value": 0,
            "max_value": 10, #'auto',
            "height": 1,
            "grid": "true",
        }

        self.tss = parse_tss()

        # import matplotlib.pyplot as plt
        self.cell_type_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
                            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',] if cell_type_colors is None else cell_type_colors

        self.fig_dir = fig_dir

    def empty_config(self):
        self.config = ConfigParser()
        # self.config.add_section("spacer")

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
            # **link_kws
        }

    def add_bed(self, peaks, name, display="collapsed", border_color="none", labels=False, **kwargs):
        pass

    def add_bigwig(self, bigwigs, **kwargs):
        for i, (name, bigwig) in enumerate(bigwigs.items()):
            bigwig_kws = self.bigwig_kws.copy()
            bigwig_kws.update(kwargs)
            # print(bigwig_kws)
            self.config[name] = {
                "file": bigwig,
                "title": name,
                "color": self.cell_type_colors[i],
                # "height": 1,
                **bigwig_kws
            }
        # self.config["spacer2"] = self.spacer_kws
        # self.config['hline'] = self.line_kws


    def add_gene_annotation(self, gene, **kwargs):
        # gene annotation
        self.config["Genes"] = {
            "file": GTF_FILE, 
            "title": "Genes", 
            "prefered_name": "gene_name", 
            "merge_transcripts": True,
            "fontsize": 4, 
            "height": 1, 
            "labels": True, 
            "max_labels": 100,
            "all_labels_inside": True, 
            "labels_in_margin": True,
            "style": "UCSC", 
            "file_type": "gtf",
            # "max_value": max_value
        }

    def generate_track(self, region, up=0, down=0, dpi=200, width=8, fontsize=4, save_dir='./'):
        with open(f"{save_dir}/tracks.ini", "w") as f:
            self.config.write(f)

        if isinstance(region, str):
            region = self.get_region_with_gene(region)
        Chromosome, Start, End = region
        Start -= up
        End += down
        cmd = f"pyGenomeTracks --tracks {save_dir}/tracks.ini --region {Chromosome}:{Start}-{End} \
                -t ' ' --dpi {dpi} --width {width} --fontSize {fontsize} \
                --outFileName {save_dir}/tracks.pdf" # 2> /dev/null"
        import subprocess
        subprocess.run(cmd, shell=True, check=True)

    def get_region_with_gene(self, gene):
        return self.tss.query(f"gene_name == '{gene}'")[['Chromosome', 'Start', 'End']].values[0]
    
    def plot_gene(self, gene, bigwigs, up=50_000, down=2000, dpi=200, width=12, fontsize=4, **bigwig_kws):
        save_dir = os.path.join(self.fig_dir, gene)
        os.makedirs(save_dir, exist_ok=True)
        self.empty_config()
        self.add_bigwig(bigwigs, **bigwig_kws)
        self.add_gene_annotation(gene)
        self.generate_track(gene, up, down, dpi, width, fontsize, save_dir=save_dir)
        from IPython.display import Image, IFrame
        # from wand.image import Image as WImage
        return IFrame(f"{save_dir}/tracks.pdf", width=800, height=600)


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
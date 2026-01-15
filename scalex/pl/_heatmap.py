import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools


def get_module_series(topgenes): # topgenes are DataFrame with ranks as index, GEPs as columns
    long_df = topgenes.stack().reset_index()
    long_df.columns = ['Rank', 'GEP', 'Gene']
    long_df = long_df.sort_values('Rank', ascending=True)
    gene_series = long_df.drop_duplicates(subset='Gene', keep='first').set_index('Gene')['GEP']
    return gene_series

def local_correlation_plot(
            local_correlation_z, modules, linkage=None,
            mod_cmap='tab10', vmin=-1, vmax=1,
            z_cmap='RdBu_r', yticklabels=False, save=False,
):
    # 1. ALIGNMENT: Ensure modules Series matches the matrix row order exactly
    # This prevents errors if 'modules' and 'local_correlation_z' are sorted differently
    modules_aligned = modules.reindex(local_correlation_z.index)

    unique_modules = sorted(modules_aligned.unique())
    
    # itertools.cycle ensures we don't run out of colors if we have many modules
    palette_cycle = itertools.cycle(sns.color_palette(mod_cmap))
    
    module_color_lookup = {}
    for mod in unique_modules:
        if mod == -1:
            module_color_lookup[mod] = '#ffffff' # White for noise
        else:
            module_color_lookup[mod] = next(palette_cycle)

    # Map the aligned modules to their colors using the lookup dictionary
    row_colors_series = modules_aligned.map(module_color_lookup)
    row_colors_series.name = "Modules" # This becomes the header in the plot

    # 3. PLOTTING
    cm = sns.clustermap(
        local_correlation_z,
        row_linkage=linkage,
        col_linkage=linkage,
        vmin=vmin,
        vmax=vmax,
        cmap=z_cmap,
        xticklabels=False,
        yticklabels=yticklabels,
        row_colors=row_colors_series.to_frame(), # Convert Series to DataFrame
        rasterized=True,
    )

    fig = plt.gcf()
    plt.sca(cm.ax_heatmap)
    plt.ylabel("")
    plt.xlabel("")
    
    # Hide the dendrogram tree
    cm.ax_row_dendrogram.remove()

    # 4. REORDERING LOGIC
    # Get the row indices used by the clustermap (calculated or provided)
    if cm.dendrogram_row is not None:
        reordered_indices = cm.dendrogram_row.reordered_ind
    else:
        reordered_indices = np.arange(len(modules_aligned))

    # Apply these indices to the ALIGNED modules series
    mod_reordered = modules_aligned.iloc[reordered_indices]

    # Calculate positions for the text labels
    mod_map = {}
    y = np.arange(modules_aligned.size)

    for x in mod_reordered.unique():
        if x == -1:
            continue
        # Find the center position of this module in the reordered list
        mod_map[x] = y[mod_reordered == x].mean()

    # Add text labels
    plt.sca(cm.ax_row_colors)
    for mod, mod_y in mod_map.items():
        plt.text(-0.5, y=mod_y, s="{}".format(mod),
                 horizontalalignment='right',
                 verticalalignment='center')
    plt.xticks([])

    # 5. COLORBAR OPTIMIZATION
    # Access the colorbar axes directly via cm.cax
    if cm.cax:
        cm.cax.set_ylabel('Correlation')
        cm.cax.yaxis.set_label_position("left")

    cbar = cm.ax_heatmap.collections[0].colorbar
    cbar.solids.set_rasterized(True)

    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
 
    return cm
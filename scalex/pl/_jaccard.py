import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scalex.pl._utils import _sort_key


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 0.0

def plot_jaccard_heatmap(dict_a, dict_b, vmax=1.0, figsize_scale=0.4,
                         name_a=None, name_b=None, save_path=None,
                         order_a=None, order_b=None):
    shared_set = set(dict_a) & set(dict_b)
    rows = [c for c in order_a if c in shared_set] if order_a else sorted(shared_set, key=_sort_key)
    cols = [c for c in order_b if c in shared_set] if order_b else sorted(shared_set, key=_sort_key)
    M = pd.DataFrame(index=rows, columns=cols, dtype=float)
    for ca in rows:
        for cb in cols:
            M.loc[ca, cb] = jaccard(dict_a[ca], dict_b[cb])

    fig, ax = plt.subplots(figsize=(figsize_scale * len(cols) + 2,
                                    figsize_scale * len(rows) + 2))
    sns.heatmap(M, cmap='viridis', vmin=0, vmax=vmax, square=True, ax=ax,
                cbar=False, linewidths=0, linecolor='none')
    ax.grid(False)

    # Annotate diagonal with overlap count
    diag = [c for c in rows if c in set(cols)]
    for c in diag:
        i = rows.index(c)
        j = cols.index(c)
        n_overlap = len(set(dict_a[c]) & set(dict_b[c]))
        ax.text(j + 0.5, i + 0.5, str(n_overlap),
                ha='center', va='center', fontsize=9, color='white')

    # Tick labels with gene counts
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
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(vmin=0, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Jaccard index', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.grid(False)

    ax.set_xlabel(name_b if name_b else 'Dict B cell types')
    ax.set_ylabel(name_a if name_a else 'Dict A cell types')
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')

    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_crosstab(
    ct: pd.DataFrame,
    figsize: tuple = None,
    cmap: str = "rocket_r",
    fmt: str = ".2f",
    title: str = None,
    row_order: list = None,
    col_order: list = None,
    save: str = None,
) -> plt.Figure:
    """
    Side-by-side heatmaps of row- and column-normalized crosstab ratios.

    Parameters
    ----------
    row_order : list, optional
        Order of row (annotation) categories on the y-axis.
    col_order : list, optional
        Order of column (condition) categories on the x-axis.
    save : str, optional
        File path to save the figure (e.g. "out.png", "out.pdf").
        Format is inferred from the extension. If None, figure is not saved.
    """
    if row_order is not None:
        ct = ct.loc[row_order]
    if col_order is not None:
        ct = ct[col_order]

    row_norm = ct.div(ct.sum(axis=1), axis=0)
    col_norm = ct.div(ct.sum(axis=0), axis=1)

    n_rows, n_cols = ct.shape
    if figsize is None:
        w = max(5, n_cols * 1.3)
        h = max(3, n_rows * 0.75)
        figsize = (w * 2 + 1, h)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    panels = [
        (row_norm, "Row proportion"),
        (col_norm, "Column proportion"),
    ]
    for ax, (data, label) in zip(axes, panels):
        sns.heatmap(
            data,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt=fmt,
            vmin=0,
            vmax=1,
            linewidths=0,
            square=False,
            annot_kws={"size": 10},
            cbar_kws={"shrink": 0.6, "label": "Proportion"},
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(label, fontsize=12, fontweight="bold", pad=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
        ax.tick_params(axis="y", rotation=0, labelsize=10, length=0)
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel(data.columns.name or "", fontsize=10)
        ax.set_ylabel(data.index.name or "", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    fig.tight_layout()
    if save is not None:
        fig.savefig(save, bbox_inches="tight", dpi=150)
    return fig


def plot_crosstab_stacked(
    ct: pd.DataFrame,
    figsize: tuple = None,
    palette: dict | str = "tab20",
    title: str = None,
    row_order: list = None,
    col_order: list = None,
    save: str = None,
) -> plt.Figure:
    """
    Stacked bar plot: one bar per condition (column), stacked by annotation (row),
    normalized to sum to 1.

    Parameters
    ----------
    palette : dict or str
        Dict maps annotation labels to colors. String is a seaborn palette name.
        Defaults to "tab20".
    row_order : list, optional
        Order of annotation categories in the stack and legend (bottom → top).
    col_order : list, optional
        Order of condition bars on the x-axis (left → right).
    save : str, optional
        File path to save the figure (e.g. "out.png", "out.pdf").
        Format is inferred from the extension. If None, figure is not saved.
    """
    if row_order is not None:
        ct = ct.loc[row_order]
    if col_order is not None:
        ct = ct[col_order]

    col_norm = ct.div(ct.sum(axis=0), axis=1)
    n_annotations = len(col_norm.index)

    if isinstance(palette, dict):
        colors = [palette[k] for k in col_norm.index]
    else:
        colors = sns.color_palette(palette, n_colors=n_annotations)

    if figsize is None:
        figsize = (max(4, len(col_norm.columns) * 0.9 + 2), 4)

    fig, ax = plt.subplots(figsize=figsize)
    col_norm.T.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=colors,
        width=0.55,
        edgecolor="white",
        linewidth=0.8,
    )

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.tick_params(axis="x", length=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Fraction", fontsize=11, labelpad=8)
    ax.set_xlabel(col_norm.columns.name or "Condition", fontsize=11, labelpad=8)
    ax.legend(
        title=col_norm.index.name or "Annotation",
        title_fontsize=10,
        fontsize=9,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=False,
    )
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    if save is not None:
        fig.savefig(save, bbox_inches="tight", dpi=150)
    return fig

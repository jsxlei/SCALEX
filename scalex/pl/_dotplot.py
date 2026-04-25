from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _build_color_map(
    color: Union[List[str], Dict[str, str]],
    groups,
) -> Dict[str, str]:
    """Map group names to colors from a list or dict."""
    if isinstance(color, dict):
        return color
    if hasattr(color, "__len__") and not isinstance(color, str):
        return dict(zip(groups, color))
    return {g: color for g in groups}


def dotplot(
    df: pd.DataFrame,
    column: str = "Adjusted P-value",
    x: Optional[str] = None,
    y: str = "Term",
    x_order: Union[List[str], bool] = False,
    y_order: Union[List[str], bool] = False,
    title: str = "",
    cutoff: float = 0.05,
    top_term: int = 10,
    size: float = 5,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (4, 6),
    cmap: str = "viridis_r",
    ofname: Optional[str] = None,
    xticklabels_rot: Optional[float] = None,
    yticklabels_rot: Optional[float] = None,
    marker: str = "o",
    show_ring: bool = False,
    color: Optional[Union[List[str], Dict[str, str]]] = None,
    margin: float = 0.1,
    ylabel_fontsize: int = 10,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **kwargs,
) -> Optional[plt.Axes]:
    """Visualize GSEApy enrichment results as a categorical dot plot.

    Extends gseapy's dotplot with two additional features:
    - Y-axis is inverted so the most significant terms appear at the top,
      making the diagonal run from upper-left to bottom-right.
    - When ``color`` is given and ``x`` is a categorical column, each group
      gets a flat color for its dots and the matching y-axis term labels are
      colored accordingly.

    Parameters
    ----------
    df
        GSEApy DataFrame results.
    column
        Column in ``df`` used for dot color (colormap) or threshold filtering.
        Default: ``"Adjusted P-value"``.
    x
        Categorical column in ``df`` mapped to the x-axis (e.g. dataset name).
        When ``None``, the numeric value of ``column`` is used as x.
    y
        Categorical column in ``df`` mapped to the y-axis. Default: ``"Term"``.
    x_order
        ``True`` for hierarchical clustering on x-axis, or a list of ordered
        category levels.
    y_order
        ``True`` for hierarchical clustering on y-axis, or a list of ordered
        category levels.
    title
        Figure title.
    cutoff
        Only terms with ``column`` value <= cutoff are shown (applies to
        p-value/FDR columns).
    top_term
        Number of top enriched terms to display.
    size
        Scale factor for dot size.
    ax
        Existing matplotlib axes to draw on. Creates a new figure when ``None``.
    figsize
        Figure size, used only when ``ax`` is ``None``.
    cmap
        Matplotlib colormap used when ``color`` is not provided.
    ofname
        Output file path. When given, saves the figure and returns ``None``.
    xticklabels_rot
        Rotation angle for x-axis tick labels.
    yticklabels_rot
        Rotation angle for y-axis tick labels.
    marker
        Matplotlib marker style.
    show_ring
        Draw a thin outer ring around each dot.
    margin
        Fractional whitespace added on each side of the x-axis. Overrides
        gseapy's hardcoded 0.25. Default: ``0.05``.
    color
        Per-group flat colors, used only when ``x`` is a categorical column.
        Accepts a list of colors (matched to groups in order) or a dict
        mapping group names to colors. When provided, replaces the colormap
        with flat per-group colors and colors the y-axis term labels to match.

    Returns
    -------
    matplotlib.axes.Axes or None
        Returns the axes when ``ofname`` is ``None``, otherwise saves and
        returns ``None``.
    """
    from gseapy.plot import DotPlot

    # --- Compute y_order: column-by-column, high→low within each group ---
    # When the user hasn't supplied an explicit y_order, do a dry-run first to get
    # the processed data, then build the order group by group (left→right), placing
    # each group's terms high→low below any terms already claimed by earlier groups.
    # After invert_yaxis() this puts Group A's top term at the top, then Group A's
    # remaining terms, then Group B's unique terms, etc.
    if not y_order:
        _probe = DotPlot(
            df=df, x=x, y=y, hue=column, x_order=x_order,
            thresh=cutoff, n_terms=int(top_term),
            dot_scale=size, figsize=figsize, cmap=cmap, marker=marker,
        )
        col = _probe.colname  # may be 'p_inv' after log-transform

        if x is not None and x in _probe.data.columns:
            # Respect x_order (clustering or custom list) for group traversal order
            group_order = _probe.get_x_order() if x_order else list(_probe.data[x].unique())
            seen: set = set()
            y_order = []
            for grp in group_order:
                grp_terms = (
                    _probe.data[_probe.data[x] == grp]
                    .sort_values(by=col, ascending=False)[y]
                    .tolist()
                )
                for term in grp_terms:
                    if term not in seen:
                        y_order.append(term)
                        seen.add(term)
        else:
            # Single group: sort by value descending
            y_order = (
                _probe.data.sort_values(by=col, ascending=False)[y].tolist()
            )

    dot = DotPlot(
        df=df,
        x=x,
        y=y,
        hue=column,
        x_order=x_order,
        y_order=y_order,
        thresh=cutoff,
        n_terms=int(top_term),
        dot_scale=size,
        ax=ax,
        figsize=figsize,
        cmap=cmap,
        ofname=ofname,
        marker=marker,
    )
    ax = dot.scatter(outer_ring=show_ring)

    # Override gseapy's auto color range with user-supplied p-value bounds.
    # gseapy uses log10(1/p) for color; vmax p-value → low end of scale (least significant),
    # vmin p-value → high end of scale (most significant).
    if vmin is not None or vmax is not None:
        sc = ax.collections[-1]
        clim_lo, clim_hi = sc.get_clim()
        if vmax is not None:
            clim_lo = vmax  # log10(1 / 10^-vmax) = vmax
        if vmin is not None:
            clim_hi = vmin  # log10(1 / 10^-vmin) = vmin
        sc.set_clim(clim_lo, clim_hi)

    # Invert y-axis: most significant terms at top, diagonal upper-left → bottom-right
    ax.invert_yaxis()
    # Override gseapy's hardcoded 0.25 x-margin
    ax.margins(x=margin, y=0.02)
    # Move x-axis ticks and label to the top; center labels over their ticks
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("")
    for label in ax.get_xticklabels():
        label.set_ha("center")
    # Move y-axis ticks and labels to the right
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_tick_params(labelsize=ylabel_fontsize)

    # Move colorbar to the left of the plot and flip ticks/label to face left
    p = ax.get_position()
    for cbar_ax in dot.fig.axes:
        if cbar_ax is not ax:
            cbar_ax.set_position([
                p.x0 - 0.10,
                p.y0,
                0.015,
                p.height * 0.2,
            ])
            cbar_ax.yaxis.set_ticks_position("left")
            cbar_ax.yaxis.set_label_position("left")
            # Move colorbar title from top to bottom.
            # gseapy sets it with loc="left", so clear that and use xlabel instead.
            cbar_ax.set_title("", loc="left")
            cbar_ax.set_xlabel(dot.cbar_title, fontweight="bold")

    # Move size legend to the left
    size_legend = ax.get_legend()
    if size_legend is not None:
        handles = size_legend.legend_handles
        labels = [t.get_text() for t in size_legend.get_texts()]
        title_text = size_legend.get_title().get_text()
        ax.legend(
            handles, labels,
            title=title_text,
            bbox_to_anchor=(-0.02, 0.5),
            loc="center right",
            frameon=False,
            labelspacing=2,
        )

    if color is not None and x is not None and x in dot.data.columns:
        groups = dot.data[x].unique()
        group_color = _build_color_map(color, groups)

        # Recolor the main scatter (last PathCollection, zorder=2)
        main_scatter = ax.collections[-1]
        dot_colors = [group_color.get(g, "black") for g in dot.data[x]]
        main_scatter.set_facecolors(dot_colors)

        # Color x-axis tick labels to match their group
        for label in ax.get_xticklabels():
            label.set_color(group_color.get(label.get_text(), "black"))

        # Color y-axis term labels: each term uses the color of the first group
        # that claims it (same column-by-column priority as y_order).
        term_primary_group: dict = {}
        for grp in groups:
            for term in dot.data[dot.data[x] == grp][y]:
                if term not in term_primary_group:
                    term_primary_group[term] = grp
        for label in ax.get_yticklabels():
            grp = term_primary_group.get(label.get_text())
            if grp:
                label.set_color(group_color.get(grp, "black"))

    if xticklabels_rot:
        for label in ax.get_xticklabels():
            label.set_ha("left")  # 'left' aligns rotated labels over ticks at top
            label.set_rotation(xticklabels_rot)

    if yticklabels_rot:
        for label in ax.get_yticklabels():
            label.set_ha("right")
            label.set_rotation(yticklabels_rot)

    if ofname is None:
        return ax
    dot.fig.savefig(ofname, bbox_inches="tight", dpi=300)

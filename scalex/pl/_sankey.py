"""Non-interactive Sankey / alluvial flow plot built on matplotlib.

The previous implementation rendered with plotly; this one produces a
publication-friendly static figure using matplotlib only, while keeping
the same call signature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle


_TAB20 = [plt.cm.tab20(i / 20) for i in range(20)]


def _numeric_key(v):
    try:
        return (0, float(v))
    except (ValueError, TypeError):
        return (1, str(v))


def _resolve_node_order(labels, orders):
    return [
        list(order) if order is not None else sorted(set(col), key=_numeric_key)
        for col, order in zip(labels, orders)
    ]


def _resolve_colors(col_nodes, colormaps):
    out = []
    for nodes, cmap in zip(col_nodes, colormaps):
        per_col, auto_idx = {}, 0
        for lbl in nodes:
            if cmap and lbl in cmap:
                per_col[lbl] = cmap[lbl]
            else:
                per_col[lbl] = _TAB20[auto_idx % len(_TAB20)]
                auto_idx += 1
        out.append(per_col)
    return out


def _node_layout(col_nodes, pair_counts, gap_frac=0.02):
    """Stack nodes per column proportional to flow, with small gaps between them.

    Returns a list of dicts ``{label: (y_bottom, y_top)}`` per column, with the
    total height for every column normalised to 1.
    """
    n_cols = len(col_nodes)
    col_totals = []
    for c, nodes in enumerate(col_nodes):
        totals = {lbl: 0.0 for lbl in nodes}
        if c == 0 and n_cols > 1:
            for (la, _), v in pair_counts[0].items():
                totals[la] = totals.get(la, 0.0) + v
        elif c == n_cols - 1:
            for (_, lb), v in pair_counts[-1].items():
                totals[lb] = totals.get(lb, 0.0) + v
        else:
            out_sum = {lbl: 0.0 for lbl in nodes}
            in_sum = dict(out_sum)
            for (la, _), v in pair_counts[c].items():
                out_sum[la] = out_sum.get(la, 0.0) + v
            for (_, lb), v in pair_counts[c - 1].items():
                in_sum[lb] = in_sum.get(lb, 0.0) + v
            for lbl in nodes:
                totals[lbl] = max(in_sum.get(lbl, 0.0), out_sum.get(lbl, 0.0))
        col_totals.append(totals)

    layouts = []
    for c, nodes in enumerate(col_nodes):
        totals = col_totals[c]
        s = sum(totals.values()) or 1.0
        gap = gap_frac
        usable = max(0.0, 1.0 - gap * max(0, len(nodes) - 1))
        y = 1.0
        bands = {}
        for lbl in nodes:
            h = (totals[lbl] / s) * usable
            bands[lbl] = (y - h, y)
            y = y - h - gap
        layouts.append(bands)
    return layouts


def _ribbon(x0, y0a, y0b, x1, y1a, y1b, color, alpha):
    """Cubic-bezier ribbon between (x0, [y0a, y0b]) and (x1, [y1a, y1b])."""
    cx0, cx1 = x0 + (x1 - x0) * 0.5, x0 + (x1 - x0) * 0.5
    verts = [
        (x0, y0a),
        (cx0, y0a), (cx1, y1a), (x1, y1a),
        (x1, y1b),
        (cx1, y1b), (cx0, y0b), (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    return PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha)


def plot_sankey(
    *labels,
    names=None,
    orders=None,
    colormaps=None,
    link_alpha: float = 0.4,
    width: float = 9.0,
    height: float = 6.0,
    node_width: float = 0.02,
    save_path: str | None = None,
    ax=None,
):
    """Plot a Sankey/alluvial diagram connecting N parallel label arrays.

    The diagram is rendered with matplotlib (non-interactive) so it can be
    saved as a static figure (PNG/PDF/SVG) without a browser.

    Parameters
    ----------
    *labels : array-like
        Two or more arrays of the same length (e.g. cell-type annotations from
        different methods). Each array becomes one column in the diagram.
    names : list of str or None, optional
        One title per column; ``None`` for a column uses the default
        ``'Label i'``. Example: ``['Method A', None, 'Method C']``.
    orders : list of list or None, optional
        One ordered list of labels per column controlling top-to-bottom node
        order. ``None`` keeps the default sorted order.
    colormaps : list of dict or None, optional
        One dict per column mapping ``label → color`` (any matplotlib colour
        spec). ``None`` for a column uses the automatic tab20 palette.
    link_alpha : float, default 0.4
        Ribbon opacity (0–1).
    width, height : float
        Figure dimensions in inches (when ``ax`` is ``None``).
    node_width : float, default 0.02
        Width of each node bar in axis fractions.
    save_path : str, optional
        File path. Format inferred from extension (``.png``/``.pdf``/``.svg``).
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. Creates a new figure when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    assert len(labels) >= 2, "Need at least 2 label arrays"
    labels = [list(l) for l in labels]
    assert all(len(l) == len(labels[0]) for l in labels), \
        "All label arrays must have the same length"

    n_cols = len(labels)
    names = list(names) if names else [None] * n_cols
    orders = list(orders) if orders else [None] * n_cols
    colormaps = list(colormaps) if colormaps else [None] * n_cols

    col_nodes = _resolve_node_order(labels, orders)
    names = [n if n is not None else f"Label {i}" for i, n in enumerate(names)]
    color_maps = _resolve_colors(col_nodes, colormaps)

    pair_counts = []
    for c in range(n_cols - 1):
        pairs = list(zip(labels[c], labels[c + 1]))
        pair_counts.append(pd.Series(pairs).value_counts().to_dict())

    layouts = _node_layout(col_nodes, pair_counts)

    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.figure

    col_xs = [0.05 + 0.90 * c / max(n_cols - 1, 1) for c in range(n_cols)]

    # Draw ribbons first, then nodes on top.
    for c in range(n_cols - 1):
        x0 = col_xs[c] + node_width
        x1 = col_xs[c + 1]
        # Track how much of each node has been consumed top-down.
        out_consumed = {lbl: layouts[c][lbl][1] for lbl in col_nodes[c]}
        in_consumed = {lbl: layouts[c + 1][lbl][1] for lbl in col_nodes[c + 1]}
        # Iterate in node order to keep ribbons stable.
        ordered_pairs = [
            ((la, lb), pair_counts[c][(la, lb)])
            for la in col_nodes[c]
            for lb in col_nodes[c + 1]
            if (la, lb) in pair_counts[c]
        ]
        total_per_la = {la: sum(v for (a, _), v in ordered_pairs if a == la)
                        for la in col_nodes[c]}
        total_per_lb = {lb: sum(v for (_, b), v in ordered_pairs if b == lb)
                        for lb in col_nodes[c + 1]}
        for (la, lb), v in ordered_pairs:
            la_band = layouts[c][la]
            lb_band = layouts[c + 1][lb]
            la_h = (la_band[1] - la_band[0]) * (v / max(total_per_la[la], 1))
            lb_h = (lb_band[1] - lb_band[0]) * (v / max(total_per_lb[lb], 1))
            y0_top = out_consumed[la]
            y0_bot = y0_top - la_h
            y1_top = in_consumed[lb]
            y1_bot = y1_top - lb_h
            ax.add_patch(_ribbon(x0, y0_top, y0_bot, x1, y1_top, y1_bot,
                                 color=color_maps[c][la], alpha=link_alpha))
            out_consumed[la] = y0_bot
            in_consumed[lb] = y1_bot

    # Node bars + labels.
    for c, nodes in enumerate(col_nodes):
        x = col_xs[c]
        for lbl in nodes:
            y0, y1 = layouts[c][lbl]
            ax.add_patch(Rectangle((x, y0), node_width, y1 - y0,
                                   facecolor=color_maps[c][lbl], edgecolor="none"))
            ha = "right" if c == 0 else "left"
            tx = x - 0.005 if c == 0 else x + node_width + 0.005
            ax.text(tx, (y0 + y1) / 2, str(lbl), ha=ha, va="center", fontsize=9)

    # Column titles.
    for c, name in enumerate(names):
        ax.text(col_xs[c] + node_width / 2, 1.04, name,
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.08)
    ax.set_axis_off()

    if save_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=200)

    return fig

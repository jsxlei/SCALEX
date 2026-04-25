import matplotlib.pyplot as plt
import pandas as pd

_TAB20 = [
    '#%02x%02x%02x' % tuple(int(v * 255) for v in plt.cm.tab20(i / 20)[:3])
    for i in range(20)
]


def _to_rgba(color_str, alpha):
    """Convert '#rrggbb' or 'rgb(r,g,b)' to 'rgba(r,g,b,alpha)'."""
    color_str = color_str.strip()
    if color_str.startswith('#'):
        h = color_str.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    elif color_str.startswith('rgb'):
        parts = color_str.replace('rgb(', '').replace(')', '').split(',')
        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        r, g, b = 100, 100, 100
    return f'rgba({r},{g},{b},{alpha})'


def plot_sankey(
    *labels,
    names=None,
    orders=None,
    colormaps=None,
    link_alpha=0.4,
    width=900,
    height=600,
    save_path=None,
):
    """Plot a Sankey diagram connecting N parallel label arrays.

    Parameters
    ----------
    *labels : array-like
        Two or more arrays of the same length (e.g. cell-type annotations from
        different methods). Each array becomes one column in the diagram.
    names : list of str or None, optional
        One title per column; use None for a column to get the default
        'Label i' title. E.g. ['Method A', None, 'Method C'].
    orders : list of list or None, optional
        One ordered list of labels per column controlling top-to-bottom node
        order. Use None for a column to keep the default sorted order.
        E.g. [['T cell', 'B cell'], None, ['Plasma', 'DC']].
    colormaps : list of dict or None, optional
        One dict per column mapping label → hex/rgb color string. Use None
        for a column to use the automatic tab20 palette.
        E.g. [{'T cell': '#e41a1c'}, None, {'Plasma': '#377eb8'}].
    link_alpha : float
        Ribbon opacity (0–1). Default 0.4.
    width, height : int
        Figure dimensions in pixels.
    save_path : str, optional
        Save as .html (interactive) or image (.png/.svg/.pdf).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    assert len(labels) >= 2, "Need at least 2 label arrays"
    labels = [list(l) for l in labels]
    assert all(len(l) == len(labels[0]) for l in labels), \
        "All label arrays must have the same length"

    n_cols = len(labels)
    names     = list(names)     if names     else [None] * n_cols
    orders    = list(orders)    if orders    else [None] * n_cols
    colormaps = list(colormaps) if colormaps else [None] * n_cols

    # ── per-column node lists (respect per-column order if given) ─────────────
    def _numeric_key(v):
        try:
            return (0, float(v))
        except (ValueError, TypeError):
            return (1, str(v))

    col_nodes = [
        list(order) if order is not None else sorted(set(col), key=_numeric_key)
        for col, order in zip(labels, orders)
    ]

    # fill in column title defaults after resolving orders
    names = [
        name if name is not None else f'Label {i}'
        for i, name in enumerate(names)
    ]

    # ── assign colors (per-column dict + global auto fallback) ────────────────
    node_color_map = []
    for nodes, cmap in zip(col_nodes, colormaps):
        per_col = {}
        auto_idx = 0
        for lbl in nodes:
            if cmap and lbl in cmap:
                per_col[lbl] = cmap[lbl]
            else:
                per_col[lbl] = _TAB20[auto_idx % len(_TAB20)]
                auto_idx += 1
        node_color_map.append(per_col)

    # ── flat node list + global index map ─────────────────────────────────────
    all_nodes, all_colors, node_global_idx = [], [], []
    offset = 0
    for nodes, color_map in zip(col_nodes, node_color_map):
        idx_map = {}
        for lbl in nodes:
            idx_map[lbl] = offset
            all_nodes.append(lbl)
            all_colors.append(color_map[lbl])
            offset += 1
        node_global_idx.append(idx_map)

    # ── links for each consecutive column pair ────────────────────────────────
    sources, targets, values, link_colors = [], [], [], []
    for c in range(n_cols - 1):
        counts = pd.Series(list(zip(labels[c], labels[c + 1]))).value_counts()
        for (la, lb), cnt in counts.items():
            src = node_global_idx[c].get(la)
            tgt = node_global_idx[c + 1].get(lb)
            if src is not None and tgt is not None:
                sources.append(src)
                targets.append(tgt)
                values.append(int(cnt))
                link_colors.append(_to_rgba(all_colors[src], link_alpha))

    # ── node positions: columns evenly spaced, nodes evenly spaced per column ─
    node_x, node_y = [], []
    for c, nodes in enumerate(col_nodes):
        x = 0.01 if n_cols == 1 else 0.01 + 0.98 * c / (n_cols - 1)
        n = len(nodes)
        for i in range(n):
            node_x.append(x)
            node_y.append(0.05 + 0.90 * i / max(n - 1, 1))

    # ── assemble figure ───────────────────────────────────────────────────────
    fig = go.Figure(go.Sankey(
        arrangement='fixed',
        node=dict(
            label=all_nodes,
            color=all_colors,
            x=node_x,
            y=node_y,
            pad=15,
            thickness=20,
            line=dict(color='white', width=0.5),
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors),
    ))

    col_xs = [0.01 + 0.98 * c / max(n_cols - 1, 1) for c in range(n_cols)]
    fig.update_layout(
        template='simple_white',
        width=width,
        height=height,
        font_size=11,
        annotations=[
            dict(x=col_xs[c], y=1.04, xref='paper', yref='paper',
                 text=f'<b>{names[c]}</b>', showarrow=False, font_size=13)
            for c in range(n_cols)
        ],
    )

    # ── save ──────────────────────────────────────────────────────────────────
    if save_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        if save_path.endswith('.html'):
            fig.write_html(save_path, include_plotlyjs='cdn')
        else:
            fig.write_image(save_path)

    return fig

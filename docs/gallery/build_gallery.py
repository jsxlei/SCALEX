"""Build small example figures for the SCALEX plotting gallery.

Run this script to (re)generate the PNGs referenced from
``docs/gallery/README.md`` and the project root ``README.md``::

    cd SCALEX
    python docs/gallery/build_gallery.py

Uses publicly available scanpy demo data (PBMC3k) so the gallery is
reproducible without proprietary inputs. Output goes to
``docs/gallery/_assets/``.

Track plots (TrackSpec / compose_tracks / trackplot_*) need real
genomic data files and are documented separately in the gallery
README; they are not auto-rendered here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scanpy as sc

import scalex
from scalex import pl
from scalex.pl._heatmap import plot_corr, local_correlation_plot

ASSETS = Path(__file__).resolve().parent / "_assets"
ASSETS.mkdir(parents=True, exist_ok=True)
DPI = 120


def _save(name: str) -> Path:
    p = ASSETS / f"{name}.png"
    plt.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close("all")
    return p


def _load_pbmc():
    """Return PBMC3k AnnData with louvain labels, UMAP, X_pca, and a synthetic batch."""
    adata = sc.datasets.pbmc3k_processed()
    adata.obs["batch"] = np.where(np.arange(adata.n_obs) % 2 == 0, "donor1", "donor2")
    adata.obsm["latent"] = adata.obsm["X_pca"][:, :20]
    return adata


def _top_markers(adata, n_per_group: int = 3) -> list[str]:
    if "rank_genes_groups" not in adata.uns:
        sc.tl.rank_genes_groups(adata, "louvain", n_genes=20, method="wilcoxon")
    names = adata.uns["rank_genes_groups"]["names"]
    return list({g for grp in names.dtype.names for g in names[grp][:n_per_group]})


def fig_embedding(adata):
    pl.embedding(adata, color="louvain", show=False)
    return _save("embedding")


def fig_meta(adata):
    pl.plot_meta(adata, use_rep="latent", color="louvain", batch="batch")
    return _save("plot_meta")


def fig_meta2(adata):
    pl.plot_meta2(adata, use_rep="latent", color="louvain", batch="batch", figsize=(6, 6))
    return _save("plot_meta2")


def fig_corr_clustermap(adata):
    pl.plot_corr_clustermap(adata, use_rep="latent", color="louvain", figsize=(6, 6))
    return _save("plot_corr_clustermap")


def fig_corr(adata):
    plot_corr(adata, groupby="louvain", obsm_key="latent",
              subsample=True, subsample_n=20)
    return _save("plot_corr")


def fig_corr_genelabels(adata):
    """Gene–gene correlation with stickout labels on selected markers."""
    genes = _top_markers(adata, n_per_group=4)
    expr = adata.raw[:, genes].X if adata.raw is not None else adata[:, genes].X
    expr = expr.toarray() if hasattr(expr, "toarray") else np.asarray(expr)
    corr = pd.DataFrame(np.corrcoef(expr.T), index=genes, columns=genes)
    from sklearn.cluster import KMeans
    modules = pd.Series(
        KMeans(n_clusters=min(6, len(genes)), random_state=0, n_init=10)
        .fit_predict(corr.values),
        index=genes,
    )
    highlights = genes[::3]  # every third gene gets a stickout label
    local_correlation_plot(corr, modules=modules, size=6,
                           row_label_ticks=highlights,
                           col_label_ticks=highlights,
                           label_fontsize=7)
    return _save("plot_corr_genelabels")


def fig_agg_heatmap(adata):
    pl.plot_agg_heatmap(adata, _top_markers(adata, n_per_group=3),
                        cell_type="louvain", show=False)
    return _save("plot_agg_heatmap")


def fig_local_correlation(adata):
    """Module-coloured gene–gene correlation clustermap for top marker genes."""
    genes = _top_markers(adata, n_per_group=4)
    expr = adata.raw[:, genes].X if adata.raw is not None else adata[:, genes].X
    expr = expr.toarray() if hasattr(expr, "toarray") else np.asarray(expr)
    corr = pd.DataFrame(np.corrcoef(expr.T), index=genes, columns=genes)
    # Build a synthetic module assignment from k-means on the corr matrix.
    from sklearn.cluster import KMeans
    modules = pd.Series(
        KMeans(n_clusters=min(6, len(genes)), random_state=0, n_init=10)
        .fit_predict(corr.values),
        index=genes,
    )
    local_correlation_plot(corr, modules=modules, size=6)
    return _save("local_correlation_plot")


def fig_heatmap(adata):
    """Unified RNA heatmap (one panel) using top marker genes."""
    pl.plot_heatmap(rna=adata, genes=_top_markers(adata, n_per_group=2),
                    groupby="louvain")
    return _save("plot_heatmap")


def fig_dotplot_markers(adata):
    """Marker-gene dotplot via scanpy (scalex.pl.dotplot is for GO results)."""
    markers = {
        "T":    ["CD3D", "CD3E", "CD8A"],
        "B":    ["MS4A1", "CD79A"],
        "NK":   ["NKG7", "GNLY", "PRF1"],
        "Mono": ["LYZ", "S100A9", "FCGR3A"],
    }
    sc.pl.dotplot(adata, markers, groupby="louvain", show=False, standard_scale="var")
    return _save("dotplot_markers")


def fig_sankey(adata):
    pl.plot_sankey(adata.obs["batch"].values, adata.obs["louvain"].values,
                   names=["batch", "louvain"])
    return _save("plot_sankey")


def fig_jaccard(adata):
    """Jaccard heatmap between two marker dicts."""
    sc.tl.rank_genes_groups(adata, "louvain", n_genes=20, method="wilcoxon")
    res = adata.uns["rank_genes_groups"]["names"]
    groups = res.dtype.names
    dict_a = {g: list(res[g][:20]) for g in groups[:5]}
    dict_b = {g: list(res[g][:20]) for g in groups[:5]}
    pl.plot_jaccard_heatmap(dict_a, dict_b, name_a="set A", name_b="set B")
    return _save("plot_jaccard_heatmap")


def fig_crosstab(adata):
    ct = pd.crosstab(adata.obs["louvain"], adata.obs["batch"])
    pl.plot_crosstab(ct)
    return _save("plot_crosstab")


def fig_crosstab_stacked(adata):
    ct = pd.crosstab(adata.obs["louvain"], adata.obs["batch"])
    pl.plot_crosstab_stacked(ct)
    return _save("plot_crosstab_stacked")


def fig_confusion(adata):
    """Confusion matrix between true louvain and a shuffled subset."""
    rng = np.random.default_rng(0)
    y = adata.obs["louvain"].cat.codes.to_numpy()
    y_pred = y.copy()
    flip = rng.choice(len(y), size=len(y) // 5, replace=False)
    y_pred[flip] = rng.integers(0, y.max() + 1, size=flip.size)
    pl.plot_confusion(y, y_pred)
    return _save("plot_confusion")


def fig_radar(adata):
    """Radar chart of per-cluster mean of a few marker genes."""
    genes = ["CD3D", "MS4A1", "NKG7", "LYZ", "FCGR3A"]
    genes = [g for g in genes if g in adata.var_names or
             (adata.raw is not None and g in adata.raw.var_names)]
    src = adata.raw if adata.raw is not None else adata
    expr = src[:, genes].X
    expr = expr.toarray() if hasattr(expr, "toarray") else np.asarray(expr)
    df = pd.DataFrame(expr, columns=genes, index=adata.obs["louvain"].values)
    df = df.groupby(level=0).mean()
    df = df.div(df.max(axis=0)).fillna(0)  # scale each gene to [0, 1]
    pl.plot_radar(df.iloc[:5])
    return _save("plot_radar")


GALLERY = [
    ("embedding",            fig_embedding),
    ("plot_meta",            fig_meta),
    ("plot_meta2",           fig_meta2),
    ("plot_corr_clustermap", fig_corr_clustermap),
    ("plot_corr",            fig_corr),
    ("plot_corr_genelabels", fig_corr_genelabels),
    ("plot_agg_heatmap",     fig_agg_heatmap),
    ("local_correlation",    fig_local_correlation),
    ("plot_heatmap",         fig_heatmap),
    ("dotplot_markers",      fig_dotplot_markers),
    ("plot_sankey",          fig_sankey),
    ("plot_jaccard_heatmap", fig_jaccard),
    ("plot_crosstab",        fig_crosstab),
    ("plot_crosstab_stacked", fig_crosstab_stacked),
    ("plot_confusion",       fig_confusion),
    ("plot_radar",           fig_radar),
]


def main() -> None:
    print(f"SCALEX {scalex.__version__} — building gallery into {ASSETS}")
    adata = _load_pbmc()
    for name, fn in GALLERY:
        try:
            path = fn(adata)
            print(f"  [ok] {name:25s} -> {path.name}")
        except Exception as e:
            print(f"  [!!] {name:25s} -> {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

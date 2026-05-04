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
from scalex.pl._heatmap import plot_corr

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


def fig_embedding(adata):
    pl.embedding(adata, color="louvain", show=False)
    return _save("embedding")


def fig_meta2(adata):
    pl.plot_meta2(adata, use_rep="latent", color="louvain", batch="batch", figsize=(6, 6))
    return _save("plot_meta2")


def fig_corr_clustermap(adata):
    pl.plot_corr_clustermap(adata, use_rep="latent", color="louvain", figsize=(6, 6))
    return _save("plot_corr_clustermap")


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


def fig_corr(adata):
    plot_corr(adata, groupby="louvain", obsm_key="latent",
              subsample=True, subsample_n=20)
    return _save("plot_corr")


GALLERY = [
    ("embedding",            fig_embedding),
    ("plot_meta2",           fig_meta2),
    ("plot_corr_clustermap", fig_corr_clustermap),
    ("dotplot_markers",      fig_dotplot_markers),
    ("plot_sankey",          fig_sankey),
    ("plot_jaccard_heatmap", fig_jaccard),
    ("plot_corr",            fig_corr),
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

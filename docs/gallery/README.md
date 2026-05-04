# SCALEX plotting gallery

A reference for every public plot function in `scalex.pl`. Each entry shows what the function produces, the minimal call signature, and a short description.

> **Reproduce these figures locally:**
> ```bash
> python docs/gallery/build_gallery.py
> ```
> All cell-type figures use scanpy's PBMC3k demo data, so no proprietary inputs are needed. Track-plot examples need genomic files (bigwig / GTF / loop tables) and are shown with code snippets only.

## Cells: embeddings, meta plots, heatmaps

### `embedding` — UMAP/PCA scatter

![embedding](_assets/embedding.png)

```python
import scanpy as sc
from scalex import pl

adata = sc.datasets.pbmc3k_processed()
pl.embedding(adata, color="louvain")
```

Colours by any `.obs` column. Supports diagonal-ordered cluster placement and per-group subsampling for crowded plots.

---

### `plot_meta2` — stacked composition bar

![plot_meta2](_assets/plot_meta2.png)

```python
pl.plot_meta2(adata, batch="batch", cell_type="louvain")
```

Stacked-bar composition of one categorical label within another (e.g. cell-type fraction per donor).

---

### `plot_corr_clustermap` — cell-type correlation clustermap

![plot_corr_clustermap](_assets/plot_corr_clustermap.png)

```python
pl.plot_corr_clustermap(adata, groupby="louvain", use_rep="X_pca")
```

Hierarchical-clustered correlation between group means in a chosen `.obsm` representation.

---

### `plot_corr` — pairwise correlation heatmap

![plot_corr](_assets/plot_corr.png)

```python
from scalex.pl._heatmap import plot_corr
plot_corr(adata, groupby="louvain", obsm_key="X_pca",
          subsample=True, subsample_n=20)
```

Within-dataset (or two-dataset) pairwise correlations, optionally subsampled per group.

---

### `local_correlation_plot` & `get_module_series`

```python
from scalex.pl._heatmap import local_correlation_plot, get_module_series

corr_df = adata.to_df().T.corr()
modules = get_module_series(corr_df, n_clusters=8)
local_correlation_plot(corr_df, modules=modules)
```

Module-coloured local correlation heatmap (gene–gene or peak–peak). `get_module_series` returns the cluster labels used for the side-strip.

---

### `plot_heatmap` — generic gene/peak heatmap

```python
pl.plot_heatmap(adata, var_names=top_markers, groupby="louvain")
```

Single-anchor expression / accessibility heatmap with z-score, dendrogram, and colour-bar control.

---

### `dotplot` — marker dotplot

![dotplot](_assets/dotplot_markers.png)

```python
import scanpy as sc

markers = {"T": ["CD3D", "CD3E"], "B": ["MS4A1", "CD79A"]}
# scalex.pl.dotplot is for GO enrichment results — use scanpy for marker dotplots:
sc.pl.dotplot(adata, markers, groupby="louvain", standard_scale="var")

# scalex.pl.dotplot — for GO enrichment DataFrames:
from scalex.pp.enrichment import enrich_and_plot
enrich_and_plot(gene_dict, organism="hsapiens")  # internally calls pl.dotplot
```

Image above shows scanpy's `sc.pl.dotplot` for marker genes. `scalex.pl.dotplot`
is specialised for plotting GO/KEGG enrichment results (term × group, sized
by gene-set size, coloured by adjusted p-value).

---

## Categorical relationships

### `plot_sankey` — alluvial flow

![plot_sankey](_assets/plot_sankey.png)

```python
pl.plot_sankey(adata.obs, source="batch", target="louvain")
```

Maps one categorical label onto another (e.g. cluster ↔ celltype assignment).

---

### `plot_jaccard_heatmap` — overlap matrix

![plot_jaccard_heatmap](_assets/plot_jaccard_heatmap.png)

```python
pl.plot_jaccard_heatmap(adata.obs, key1="louvain_v1", key2="louvain_v2")
```

Pairwise Jaccard similarity between two categorical label sets.

---

### `plot_crosstab` & `plot_crosstab_stacked`

```python
pl.plot_crosstab(adata.obs, x="batch", y="louvain")
pl.plot_crosstab_stacked(adata.obs, x="batch", y="louvain", normalize="x")
```

Heatmap (`plot_crosstab`) or stacked-bar (`plot_crosstab_stacked`) of two-way label tables, with optional row/column normalization.

---

## Genomic tracks

### `compose_tracks` + `TrackSpec` — multi-panel locus view

```python
from scalex.pl.trackplot import TrackSpec, compose_tracks

tracks = [
    TrackSpec("scalebar"),
    TrackSpec("coverage", data=frag_file, cell_type="Foamy_TypeI", label="Foamy"),
    TrackSpec("coverage", data=frag_file, cell_type="Resident",    label="Resident"),
    TrackSpec("loop",     data=loops_df,  label="scE2G",
              params={"score_column": "ENCODE-rE2G.Score"}),
    TrackSpec("gene",     data=transcripts, label="Genes"),
]
fig = compose_tracks(tracks, region="chr3:69729463-69978332",
                     cell_groups=adata.obs["cell_type"], save="locus.png")
```

Composable, declarative locus plotting. Track types:
`scalebar`, `coverage`, `gene`, `annotation`, `loop`, `shap`, `bigwig`, `sce2g`, `scglue`.

Register a custom type via `@register_track("my_track")`.

---

### Function-based track helpers

```python
from scalex.pl import (
    trackplot_coverage, trackplot_gene, trackplot_loop,
    trackplot_scalebar, trackplot_genome_annotation, trackplot_combine,
)

p1 = trackplot_scalebar(region)
p2 = trackplot_coverage(coverage_df, region)
p3 = trackplot_gene(transcripts, region)
fig = trackplot_combine(p1, p2, p3)
```

One panel per call; useful when assembling figures cell-by-cell in a notebook. Same panels as `compose_tracks`, just imperative.

---

### `plot_tracks` — pyGenomeTracks-based browser

```python
pl.plot_tracks(region="chr1:1000000-1200000",
               bigwigs={"ATAC": "atac.bw", "H3K27ac": "h3k27ac.bw"},
               output="locus.png")
```

Wraps the pyGenomeTracks CLI for IGV-style multi-track browser images. Use `trackplot/compose_tracks` instead when you want pure-Python figures.

---

## Variant interpretation

### `plot_variant_effect` (`scalex.pl.variant`)

```python
from scalex.pl.variant import plot_variant_effect

plot_variant_effect(
    chrom="chr1", pos=1000000, ref="A", alt="G",
    model="chrombpnet.h5", fasta="hg38.fa",
)
```

ChromBPNet-based ref/alt sequence prediction with stacked profile and contribution-score (SHAP) tracks. Requires `chrombpnet`, `tangermeme`, and `pyfaidx`.

---

## Backward-compat reference

* `scalex.pl._legacy_snapatac2` — original snapatac2-style plotly QC plots (`tsse`, `frag_size_distr`, `umap`, `regions`, `motif_enrichment`). Kept for users who depended on the old `scalex.pl.umap` / `scalex.pl.tsse` paths.

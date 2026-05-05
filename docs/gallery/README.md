# SCALEX plotting gallery

A reference for every public plot function in `scalex.pl`. Each entry shows what the function produces, the minimal call signature, and a short description.

> **Reproduce these figures locally:**
> ```bash
> python docs/gallery/build_gallery.py
> ```
> All cell-type figures use scanpy's PBMC3k demo data, so no proprietary inputs are needed. Track-plot and variant examples need genomic files (bigwig / GTF / loop tables / fasta / model) and are shown with code snippets only.

---

## Cells: embeddings

### `plot_embedding` — per-batch UMAP/PCA scatter

![embedding](_assets/embedding.png)

```python
import scanpy as sc
from scalex import pl

adata = sc.datasets.pbmc3k_processed()
pl.plot_embedding(adata, color="louvain", groupby="batch")
```

Splits cells by `groupby` into one panel per category, colouring by `color` and showing the rest of the dataset as light-grey context. `embedding` is kept as an alias for backward compatibility.

---

### `plot_subplots` — utility subplot grid

```python
fig, axes = pl.plot_subplots(n_panels=7, ncols=3)
```

Returns `(fig, axes)` with empty cells removed. Convenient when assembling custom multi-panel figures.

---

## Cells: pseudobulk correlation

### `plot_pseudobulk_corr` — per-batch × cell-type pseudobulk heatmap

![plot_meta](_assets/plot_meta.png)

```python
pl.plot_pseudobulk_corr(adata, use_rep="latent", color="louvain", batch="batch")
# legacy name still works:
pl.plot_meta(adata, use_rep="latent", color="louvain", batch="batch")
```

Builds one pseudobulk vector per `(batch, cell-type)` and shows their Pearson correlation as a clustered (lower-triangle) heatmap. Tick labels are tinted by batch.

---

### `plot_pseudobulk_corr_cross` — cross-batch composition heatmap

![plot_meta2](_assets/plot_meta2.png)

```python
pl.plot_pseudobulk_corr_cross(adata, use_rep="latent",
                              color="louvain", batch="batch", figsize=(6, 6))
# legacy name still works:
pl.plot_meta2(adata, use_rep="latent", color="louvain", batch="batch")
```

Cross-batch Pearson correlation between cell-types in two batches; rows = batch B, cols = batch A. Useful for sanity-checking cell-type alignment across donors / conditions.

---

### `plot_corr_clustermap` — single cell × cell hierarchical clustermap

![plot_corr_clustermap](_assets/plot_corr_clustermap.png)

```python
pl.plot_corr_clustermap(adata, use_rep="latent", color="louvain")
```

Stratified-subsamples cells per group, computes the **N×N cell–cell** Pearson correlation in `obsm[use_rep]`, then orders rows/cols either by `cluster_by_corr` (hierarchical clustering) **or** by `_diagonal_order` (cell-type aligned along the diagonal). Adds left/bottom colour strips for cell-type and batch and a single shared colour bar. With `compare=True`, switches to a cross-batch N1×N2 mode.

---

### `plot_corr` — cell × cell correlation with module strips

![plot_corr](_assets/plot_corr.png)

```python
from scalex.pl import plot_corr

plot_corr(adata, groupby="louvain", obsm_key="latent",
          subsample=True, subsample_n=20)
```

Within-dataset (or two-dataset) cell–cell correlation in `obsm[obsm_key]`. Pass `adata2=` for explicit cross-correlation, or `batch=<obs col>` to split a single AnnData. Internally dispatches to `local_correlation_plot`, so all of its options (custom ordering, `cluster=False`, colour map, etc.) are forwarded.

#### With stickout gene labels

![plot_corr_genelabels](_assets/plot_corr_genelabels.png)

```python
from scalex.pl import local_correlation_plot
from scalex.pl._heatmap import plot_corr  # for cell-cell mode

# gene-gene example: pass any DataFrame; row/column names are the labels
local_correlation_plot(corr_df, modules=modules,
                       row_label_ticks=highlights,
                       col_label_ticks=highlights,
                       label_fontsize=7)

# cell-cell example: forward kwargs through plot_corr
plot_corr(adata, groupby="louvain", obsm_key="latent",
          row_label_ticks=cell_ids_to_highlight, label_fontsize=7)
```

`row_label_ticks` / `col_label_ticks` accept lists of values matching the input matrix's index / columns (gene names, cell ids, ...) and draw leader-line "stickout" labels next to the heatmap, similar to `plot_heatmap`. `label_fontsize` controls the text size.

---

### `local_correlation_plot` & `get_module_series`

![local_correlation_plot](_assets/local_correlation_plot.png)

```python
from scalex.pl import local_correlation_plot, get_module_series

corr_df = adata.to_df().T.corr()
modules = get_module_series(top_genes_dataframe)
local_correlation_plot(corr_df, modules=modules,
                       row_label_ticks=["CD3D", "MS4A1"],
                       label_fontsize=7)
```

Module-coloured local correlation clustermap (gene–gene or peak–peak). `get_module_series` collapses a `(rank × GEP)` top-genes DataFrame to a `gene → GEP` Series suitable for use as the side strip. Supports `row_label_ticks` / `col_label_ticks` for selectively annotating items.

---

## Heatmaps & dotplots

### `plot_agg_heatmap` — per-cell-type mean expression

![plot_agg_heatmap](_assets/plot_agg_heatmap.png)

```python
pl.plot_agg_heatmap(adata, marker_genes, cell_type="louvain")
```

Aggregates expression by `cell_type` (mean), wraps the result back into an AnnData, and dispatches to `sc.pl.heatmap` with gene labels shown.

---

### `plot_heatmap` — unified RNA / ATAC heatmap

![plot_heatmap](_assets/plot_heatmap.png)

```python
pl.plot_heatmap(rna=adata, genes=marker_genes, groupby="louvain")
# Multi-panel with peak2gene links:
pl.plot_heatmap(rna=rna_avg, atac=atac_avg, links=links_df,
                groupby="cell_type", gene_ticks=["GATA1", "FOXP3"])
```

Publication-quality multi-panel heatmap with shared row ordering. Supports any combination of RNA and ATAC panels, K-means row grouping, leader-line `gene_ticks`, and a single side colour bar. Returns ordered matrices when `return_matrices=True` so the layout can be replayed.

---

### `dotplot` — GO/KEGG enrichment dotplot

![dotplot](_assets/dotplot_markers.png)

```python
# scalex.pl.dotplot — for GSEApy enrichment DataFrames:
from scalex.pl import dotplot
dotplot(enrich_df, x="dataset", y="Term", column="Adjusted P-value",
        top_term=10)

# For marker dotplots use scanpy directly:
sc.pl.dotplot(adata, marker_dict, groupby="louvain", standard_scale="var")
```

`scalex.pl.dotplot` is specialised for plotting GSEApy enrichment results (term × group, sized by gene-set size, coloured by adjusted p-value). The image above is `sc.pl.dotplot` for marker genes, shown for comparison.

---

## Categorical relationships

### `plot_sankey` — alluvial flow (matplotlib, non-interactive)

![plot_sankey](_assets/plot_sankey.png)

```python
pl.plot_sankey(
    adata.obs["batch"].values,
    adata.obs["louvain"].values,
    names=["batch", "louvain"],
)
```

Static matplotlib Sankey/alluvial diagram connecting two or more categorical label arrays. Pass `orders=` to fix node order per column and `colormaps=` to pin colours.

---

### `plot_jaccard_heatmap` — set overlap matrix

![plot_jaccard_heatmap](_assets/plot_jaccard_heatmap.png)

```python
pl.plot_jaccard_heatmap(dict_a, dict_b, name_a="set A", name_b="set B")
```

Pairwise Jaccard similarity between two `{label: list_of_genes}` dictionaries, with diagonal overlap counts and per-set sizes in the tick labels.

---

### `plot_crosstab` & `plot_crosstab_stacked`

![plot_crosstab](_assets/plot_crosstab.png)
![plot_crosstab_stacked](_assets/plot_crosstab_stacked.png)

```python
import pandas as pd

ct = pd.crosstab(adata.obs["louvain"], adata.obs["batch"])
pl.plot_crosstab(ct)
pl.plot_crosstab_stacked(ct)
```

Both take a pre-computed `pd.crosstab` DataFrame. `plot_crosstab` shows row- and column-normalised heatmaps side-by-side; `plot_crosstab_stacked` shows one stacked bar per condition.

---

## Evaluation

### `plot_confusion` — clustering confusion matrix

![plot_confusion](_assets/plot_confusion.png)

```python
pl.plot_confusion(y_true, y_pred)
```

Normalised confusion matrix between two integer label arrays (e.g. ground-truth vs. predicted clusters). Returns the (F1, NMI, ARI) scalars.

---

### `plot_radar` — radar chart

![plot_radar](_assets/plot_radar.png)

```python
pl.plot_radar(df, vmax=1)
```

Spider chart for a `(samples × axes)` DataFrame; one polygon per row. Useful for comparing per-cluster scores or per-method metrics on a fixed set of axes.

---

### `plot_expr` — per-batch UMAP for one gene

```python
pl.plot_expr(adata, gene="CD3D", groupby="batch", category="my_condition")
```

Grid of UMAP panels (one per batch) coloured by `gene` expression with the rest of the cells faded. Requires an `obs['category']` column for filtering — keep this in mind when adapting to your own data.

---

### `reassign_cluster_with_ref`

```python
y_aligned, ind = pl.reassign_cluster_with_ref(y_pred, y_true)
```

Hungarian-algorithm relabelling so that predicted cluster integers match the reference. Not a plotting function — pair it with `plot_confusion` to produce a sane confusion matrix.

---

## Genomic tracks (code-only — needs genomic inputs)

### `compose_tracks` + `TrackSpec` — declarative multi-panel locus view

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

Track types: `scalebar`, `coverage`, `gene`, `annotation`, `loop`, `shap`, `bigwig`, `sce2g`, `scglue`. Register custom types via `@register_track("my_track")`.

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

Imperative equivalent of `compose_tracks`. Same panels, just one call per panel.

---

### `plot_tracks` — pyGenomeTracks-based browser

```python
pl.plot_tracks(region="chr1:1000000-1200000",
               bigwigs={"ATAC": "atac.bw", "H3K27ac": "h3k27ac.bw"},
               output="locus.png")
```

Wraps the pyGenomeTracks CLI for IGV-style multi-track browser images. Use `compose_tracks` instead when you want pure-Python figures.

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

* `embedding`, `plot_meta`, `plot_meta2` — kept as aliases for `plot_embedding`, `plot_pseudobulk_corr`, `plot_pseudobulk_corr_cross`.
* `scalex.pl._legacy_snapatac2` — original snapatac2-style plotly QC plots (`tsse`, `frag_size_distr`, `umap`, `regions`, `motif_enrichment`).

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

from scalex.atac.snapatac2._diff import aggregate_X
from scalex.data import aggregate_data

from scalex.atac.bedtools import intersect_bed, bed_to_df, df_to_bed
from scalex.pp.annotation import format_rna, format_atac
from scalex.analysis import get_markers, flatten_dict
# from scalex.linkage.utils import row_wise_correlation

class Linkage:
    def __init__(self, rna=None, atac=None, rna_avg=None, atac_avg=None, groupby='cell_type', gtf=None, up=100_000, down=100_000):
        self.rna = rna
        self.atac = atac
        self.gtf = gtf if gtf is not None else os.path.expanduser('~/.scalex/gencode.v38.annotation.gtf.gz')
        self.up = up
        self.down = down

        self.format_data()
        if rna_avg is None or atac_avg is None:
            self.aggregate_data(groupby=groupby)
        else:
            self.rna_avg = rna_avg
            self.atac_avg = atac_avg
        self.get_edges(up=up, down=down)
        self.get_peak_gene_corr()


    def format_data(self):
        self.rna = format_rna(self.rna.copy(), gtf=self.gtf)
        self.atac = format_atac(self.atac.copy())
        self.rna_var = self.rna.var
        self.atac_var = self.atac.var

    def aggregate_data(self, groupby="cell_type"):
        self.rna_avg = aggregate_data(self.rna, groupby=groupby)
        self.atac_avg = aggregate_data(self.atac, groupby=groupby)

        common = set(self.rna_avg.obs[groupby].values) & set(self.atac_avg.obs[groupby].values)
        self.rna_avg = self.rna_avg[self.rna_avg.obs[groupby].isin(common)]
        self.atac_avg = self.atac_avg[self.atac_avg.obs[groupby].isin(common)]
        print(common)

    def get_cell_type_links(self, marker_genes, marker_peaks, groupby="cell_type"):
        links = {}
        shared_keys = set(marker_genes.keys()) & set(marker_peaks.keys())   
        for cell_type in shared_keys:
            rna_avg_ct = self.rna_avg[:, marker_genes[cell_type]].copy()
            atac_avg_ct = self.atac_avg[:, marker_peaks[cell_type]].copy()
            if rna_avg_ct.shape[0] == 0 or atac_avg_ct.shape[0] == 0:
                continue
            edge = intersect_bed(rna_avg_ct.var, atac_avg_ct.var, up=self.up, down=self.down, add_distance=True)
            # print(len(edge[0]), len(edge[1]))
            index_to_gene = {i: gene for i, gene in enumerate(rna_avg_ct.var.index)}
            index_to_peak = {i: peak for i, peak in enumerate(atac_avg_ct.var.index)}
            gene_list = [index_to_gene[i] for i in edge[0]]
            peak_list = [index_to_peak[i] for i in edge[1]]
            links[cell_type] = list(zip(gene_list, peak_list))
        return links
    
    def link_peak_to_gene(self, links):
        links = flatten_dict(links)
        peak_to_gene = dict(zip(links[1], links[0]))
        return peak_to_gene

    def get_edges(self, up=100_000, down=100_000):
        # assert self.rna_agg.obs[self.groupby] == self.atac_agg.obs[self.groupby], "groupby should have the same order"
        ## peak to gene linkage
        self.edges = intersect_bed(self.rna_var, self.atac_var, up=up, down=down, add_distance=True)
        self.gene_to_index = {gene: i for i, gene in enumerate(self.rna_var.index)}
        self.peak_to_index = {peak: i for i, peak in enumerate(self.atac_var.index)}    

    def get_peak_gene_corr(self):
        peak_vec = self.atac_avg.to_df().iloc[:, self.edges[1]].T.values
        genes_vec = self.rna_avg.to_df().iloc[:, self.edges[0]].T.values

        self.peak_gene_corr = row_wise_correlation(genes_vec, peak_vec) 

    def filter_peak_gene_linkage(self, threshold=0.7):
        indices = np.where(self.peak_gene_corr > threshold)[0]
        gene_indices = self.edges[0][indices]
        peak_indices = self.edges[1][indices]

        genes = self.rna_var.iloc[gene_indices].index.values
        peaks = self.atac_var.iloc[peak_indices].index.values

        return genes, peaks

    def find_overlapping_peak_gene_linkage(self, genes_deg, peaks_deg, threshold=0.7):
        genes, peaks = self.filter_peak_gene_linkage(threshold=threshold)
        if isinstance(genes_deg, dict):
            genes_deg = flatten_dict(genes_deg)
        if isinstance(peaks_deg, dict):
            peaks_deg = flatten_dict(peaks_deg)

        gene_peak_pair = [(genes[i], peaks[i]) for i, _ in enumerate(genes) if genes[i] in genes_deg and peaks[i] in peaks_deg]

        return gene_peak_pair

    def get_cell_type_peak_gene_links(self, marker_genes, marker_peaks, correlation_threshold=0.7):
        """
        Identify linked peaks and genes for each cell type.
        
        Parameters:
        -----------
        cell_type_genes : dict
            Dictionary mapping cell types to lists of cell type specific genes
        cell_type_peaks : dict
            Dictionary mapping cell types to lists of cell type specific peaks
        correlation_threshold : float, optional (default=0.7)
            Minimum correlation threshold for peak-gene pairs
            
        Returns:
        --------
        dict
            Dictionary mapping cell types to lists of (gene, peak) tuples
        """
        cell_type_links = {}
        
        # Get all potential peak-gene pairs based on genomic distance and correlation
        genes, peaks = self.filter_peak_gene_linkage(threshold=correlation_threshold)
        
        # Create lookup dictionaries for faster membership testing
        gene_to_peak = {gene: peak for gene, peak in zip(genes, peaks)}
        peak_to_gene = {peak: gene for gene, peak in zip(genes, peaks)}
        
        # Process each cell type
        for cell_type in marker_genes.keys():
            if cell_type not in marker_peaks:
                continue
                
            # Get cell type specific genes and peaks
            ct_genes = set(marker_genes[cell_type])
            ct_peaks = set(marker_peaks[cell_type])
            
            # Find overlapping genes and peaks that are linked
            linked_pairs = []
            for gene in ct_genes:
                if gene in gene_to_peak and gene_to_peak[gene] in ct_peaks:
                    linked_pairs.append((gene, gene_to_peak[gene]))
            
            cell_type_links[cell_type] = linked_pairs
            
        return cell_type_links

def row_wise_correlation(arr1, arr2, epsilon=1e-8):
    """
    Calculates the Pearson correlation coefficient between corresponding rows of two NumPy arrays,
    with robustness to small values and division by zero.

    Parameters:
    - arr1: NumPy array of shape (m, n)
    - arr2: NumPy array of shape (m, n)
    - epsilon: Small constant to avoid division by zero (default: 1e-8)

    Returns:
    - correlations: NumPy array of shape (m,)
    """
    assert arr1.shape == arr2.shape, "Arrays must have the same shape."

    # Compute means
    mean1 = np.mean(arr1, axis=1, keepdims=True)
    mean2 = np.mean(arr2, axis=1, keepdims=True)

    # Compute standard deviations
    std1 = np.std(arr1, axis=1, ddof=1, keepdims=True)
    std2 = np.std(arr2, axis=1, ddof=1, keepdims=True)

    # Avoid division by zero by adding epsilon
    safe_std1 = np.where(std1 < epsilon, np.nan, std1)
    safe_std2 = np.where(std2 < epsilon, np.nan, std2)

    # Standardize the data (z-scores)
    z1 = (arr1 - mean1) / safe_std1
    z2 = (arr2 - mean2) / safe_std2

    # Compute sum of products of z-scores
    sum_of_products = np.nansum(z1 * z2, axis=1)

    # Degrees of freedom
    n = arr1.shape[1]
    degrees_of_freedom = n - 1

    # Compute Pearson correlation coefficients
    correlations = sum_of_products / degrees_of_freedom

    return correlations
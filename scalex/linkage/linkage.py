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
from scalex.analysis import get_markers
# from scalex.linkage.utils import row_wise_correlation

class Linkage:
    def __init__(self, rna, atac, groupby='cell_type', gtf=None, up=100_000, down=100_000):
        self.rna = rna
        self.atac = atac
        self.gtf = gtf
        self.up = up
        self.down = down

        self.format_data()
        self.aggregate_data(groupby=groupby)
        self.get_edges(up=up, down=down)
        self.get_peak_gene_corr()


    def format_data(self):
        self.rna = format_rna(self.rna)
        format_atac(self.atac)
        self.rna_var = self.rna.var
        self.atac_var = self.atac.var

    def aggregate_data(self, groupby="cell_type"):
        self.rna_agg = aggregate_data(self.rna, groupby=groupby)
        self.atac_agg = aggregate_X(self.atac, groupby=groupby, normalize="RPKM")

    def get_edges(self, up=100_000, down=100_000):
        ## peak to gene linkage
        self.edges = intersect_bed(self.rna_var, self.atac_var, up=up, down=down, add_distance=True)
        self.gene_to_index = {gene: i for i, gene in enumerate(self.rna_var.index)}
        self.peak_to_index = {peak: i for i, peak in enumerate(self.atac_var.index)}    

    def get_peak_gene_corr(self):
        peak_vec = self.atac_agg.to_df().iloc[:, self.edges[1]].T.values
        genes_vec = self.rna_agg.to_df().iloc[:, self.edges[0]].T.values

        self.peak_gene_corr = row_wise_correlation(genes_vec, peak_vec) 

    def filter_peak_gene_linkage(self, threshold=0.7):
        indices = np.where(self.peak_gene_corr > threshold)[0]
        gene_indices = self.edges[0][indices]
        peak_indices = self.edges[1][indices]

        genes = self.rna_var.iloc[gene_indices].index.values
        peaks = self.atac_var.iloc[peak_indices].index.values

        return genes, peaks



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
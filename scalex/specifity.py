#!/usr/bin/env python
"""
# Optimized Specificity Score Calculation
# Author: Xiong Lei (original), optimized by Xuxin Tang
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

def jsd(p, q, base=np.e):
    """
    Compute Jensen-Shannon divergence.
    """
    p, q = np.asarray(p), np.asarray(q)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (stats.entropy(p, m, base=base) + stats.entropy(q, m, base=base))

def jsd_sp(p, q, base=np.e):
    """
    Specificity score: 1 - sqrt(jsd)
    """
    return 1 - np.sqrt(jsd(p, q, base))

def log2norm(e):
    """
    Log2 normalization: log2(e+1) and scaling.
    """
    loge = np.log2(e + 1)
    return loge / loge.sum()

def predefined_pattern(t, labels):
    """
    Generate predefined binary pattern for cluster t.
    """
    return (labels == t).astype(int)

def vec_specificity_score(e, t, labels):
    """
    Compute specificity score for a given cluster.
    """
    return jsd_sp(log2norm(e), log2norm(predefined_pattern(t, labels)))

def mat_specificity_score(mat, labels):
    """
    Compute specificity scores for all genes/peaks across clusters.
    Returns a DataFrame of genes/peaks (rows) x clusters (columns).
    """
    unique_labels = np.unique(labels)
    return pd.DataFrame(
        {t: mat.apply(lambda x: vec_specificity_score(x, t, labels), axis=1) for t in unique_labels},
        index=mat.index
    )

def compute_pvalues(score_mat, labels, num_permutations=1000):
    """
    Compute p-values for cluster-specificity scores using permutation testing.
    """
    shuffled_scores = np.zeros((score_mat.shape[0], score_mat.shape[1], num_permutations))

    for i in range(num_permutations):
        shuffled_labels = np.random.permutation(labels)
        shuffled_scores[:, :, i] = mat_specificity_score(score_mat, shuffled_labels).values

    p_values = (np.sum(shuffled_scores >= score_mat.values[:, :, None], axis=2) + 1) / (num_permutations + 1)
    return pd.DataFrame(p_values, index=score_mat.index, columns=score_mat.columns)

def filter_significant_clusters(score_mat, p_values, alpha=0.05):
    """
    Filter cluster-specific genes/peaks based on significance threshold (p-value < alpha).
    """
    significant_mask = p_values < alpha
    return score_mat.where(significant_mask)

def cluster_specific(score_mat, classes=None, top=0):
    """
    Identify top specific genes/peaks for each cluster.
    """
    max_scores = score_mat.max(axis=1)
    peak_labels = score_mat.idxmax(axis=1)

    if classes is None:
        classes = peak_labels.unique()

    top_indices = {
        cls: max_scores[peak_labels == cls].nlargest(top).index for cls in classes
    }
    
    return top_indices

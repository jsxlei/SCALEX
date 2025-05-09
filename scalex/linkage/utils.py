import pandas as pd
import numpy as np


def correlation_between_df(df1, df2):
    df1_standardized = (df1 - df1.mean(axis=1).values.reshape(-1, 1)) / df1.std(axis=1).values.reshape(-1, 1)
    df2_standardized = (df2 - df2.mean(axis=1).values.reshape(-1, 1)) / df2.std(axis=1).values.reshape(-1, 1)
    
    # Calculate the correlation matrix
    correlation_matrix = np.dot(df2_standardized.values, df1_standardized.values.T) / df1.shape[1]
    
    # Convert the result to a DataFrame
    correlation_df = pd.DataFrame(correlation_matrix, index=df2.index, columns=df1.index)
    return correlation_df


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


"""Lightweight helpers shared across ``scalex.pl`` modules.

Keep this file dependency-free beyond numpy: anything heavier (matplotlib,
seaborn, scanpy) belongs in the module that needs it.
"""
import numpy as np


def _sort_key(x):
    """Sort key: numeric values first (ascending), then alphabetic strings."""
    try:
        return (0, float(x))
    except (ValueError, TypeError):
        return (1, str(x))


def _pearson_corr(X: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix between rows of ``X`` (float32)."""
    X = X.astype(np.float32)
    X_c = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (X_c / norms) @ (X_c / norms).T


def _pearson_cross(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson cross-correlation between rows of ``A`` and rows of ``B`` (float32)."""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    A_c = A - A.mean(axis=1, keepdims=True)
    B_c = B - B.mean(axis=1, keepdims=True)
    A_n = A_c / (np.linalg.norm(A_c, axis=1, keepdims=True) + 1e-10)
    B_n = B_c / (np.linalg.norm(B_c, axis=1, keepdims=True) + 1e-10)
    return A_n @ B_n.T


def _row_zscore(X: np.ndarray) -> np.ndarray:
    """Z-score each row; rows with zero variance are left as zeros."""
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True)
    sigma[sigma == 0] = 1
    return (X - mu) / sigma

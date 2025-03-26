import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.utils.extmath import randomized_svd
import logging
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)

def normalize(matrix: sp.csr_matrix, feature_weights: np.ndarray) -> None:
    """
    Applies feature weighting and L2 normalization on the rows of a sparse matrix.
    """
    matrix = matrix.copy()
    matrix.data *= feature_weights[matrix.indices]

    row_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    row_norms[row_norms == 0] = 1.0  # Avoid division by zero
    matrix = matrix.multiply(sp.diags(1.0 / row_norms))

    return matrix

def compute_idf(matrix: sp.csr_matrix) -> np.ndarray:
    """
    Computes the inverse document frequency (IDF) for feature weighting.
    """
    doc_count = np.diff(matrix.indptr)
    n_docs = matrix.shape[0]
    idf_values = np.log(n_docs / (doc_count + 1))  # Avoid division by zero
    return idf_values

def spectral_embedding(
    anndata: sp.csr_matrix,
    selected_features: np.ndarray,
    n_components: int,
    random_state: int,
    feature_weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform spectral embedding using matrix factorization.
    """
    logging.info("Performing spectral embedding using matrix factorization...")

    # Select features
    matrix = anndata[:, selected_features]

    # Apply feature weighting
    if feature_weights is None:
        feature_weights = compute_idf(matrix)
    matrix = normalize(matrix, feature_weights)

    # Compute eigenvalues and eigenvectors
    evals, evecs = eigsh(matrix, k=n_components, which="LM", random_state=random_state)
    return evals, evecs

def spectral_embedding_nystrom(
    anndata: sp.csr_matrix,
    selected_features: np.ndarray,
    n_components: int,
    sample_size: int,
    weighted_by_degree: bool,
    chunk_size: int,
    feature_weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform spectral embedding using the Nystrom method.
    """
    logging.info("Performing spectral embedding using the Nystrom algorithm...")

    matrix = anndata[:, selected_features]

    # Compute feature weighting
    if feature_weights is None:
        feature_weights = compute_idf(matrix)
    
    matrix = normalize(matrix, feature_weights)

    # Sample landmarks
    n_samples = matrix.shape[0]
    rng = np.random.default_rng(2023)
    
    if weighted_by_degree:
        degree_weights = np.array(matrix.sum(axis=1)).flatten()
        degree_weights /= degree_weights.sum()
        selected_indices = rng.choice(n_samples, size=sample_size, p=degree_weights, replace=False)
    else:
        selected_indices = rng.choice(n_samples, size=sample_size, replace=False)

    seed_matrix = matrix[selected_indices, :]

    # Compute spectral decomposition of the sample matrix
    evals, evecs = eigsh(seed_matrix, k=n_components, which="LM")

    logging.info("Applying Nystrom extension...")
    full_matrix = matrix @ evecs @ np.diag(1 / evals)

    return evals, full_matrix

def multi_spectral_embedding(
    anndatas: list,
    selected_features: list,
    weights: list,
    n_components: int,
    random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform multi-view spectral embedding by concatenating feature spaces.
    """
    logging.info("Computing normalized views...")

    weighted_matrices = []
    for data, features in zip(anndatas, selected_features):
        matrix = data[:, features]
        feature_weights = compute_idf(matrix)
        matrix = normalize(matrix, feature_weights)

        norm_factor = np.linalg.norm(matrix.data)  # Frobenius norm approximation
        weighted_matrices.append(matrix * (weights / norm_factor))

    combined_matrix = sp.hstack(weighted_matrices)

    logging.info("Computing embedding...")
    evals, evecs = eigsh(combined_matrix, k=n_components, which="LM", random_state=random_state)

    return evals, evecs

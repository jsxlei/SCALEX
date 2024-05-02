import pytest
import anndata as ad
import numpy as np
import pandas as pd

np.random.seed(42)

@pytest.fixture(scope='session')
def adata_test(n_obs=16, n_vars=32, n_categories=3):
    """
    Creates a virtual AnnData object with random binary data for testing.

    Parameters:
    - n_obs (int): Number of observations (cells)
    - n_vars (int): Number of variables (genes)
    - n_categories (int): Number of categories for the categorical annotation

    Returns:
    - An AnnData object populated with binary data and annotations.
    """
    # Generate random binary data
    X = np.random.randint(0, 2, size=(n_obs, n_vars))  # Random binary matrix (0s and 1s)

    # Generate observation names and variable names
    obs_names = [f"Cell_{i}" for i in range(n_obs)]
    var_names = [f"Gene_{j}" for j in range(n_vars)]

    # Create observation (cell) metadata
    obs = pd.DataFrame({
        'condition': np.random.choice([f"Condition_{i}" for i in range(1, n_categories + 1)], n_obs),
        'batch': np.random.choice([i for i in range(1, n_categories + 1)], n_obs)
    }, index=obs_names)

    # Create variable (gene) metadata
    var = pd.DataFrame({
        'Gene_ID': var_names
    }, index=var_names)

    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs['batch'] = adata.obs['batch'].astype('category')

    return adata
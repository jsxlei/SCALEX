"""Clustering tools for SCALEX latent embeddings."""

from anndata import AnnData

from scalex.utils import DEFAULT_RESOLUTION, DEFAULT_N_NEIGHBORS


def cluster(
    adata: AnnData,
    resolution: float = DEFAULT_RESOLUTION,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    key_added: str = 'leiden',
    use_rep: str = 'latent',
    random_state: int = 0,
) -> AnnData:
    """Cluster cells on the SCALEX latent embedding using the Leiden algorithm.

    Builds a k-NN graph on ``adata.obsm[use_rep]``, then applies Leiden
    community detection. Results are stored in ``adata.obs[key_added]``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must have ``adata.obsm[use_rep]`` populated
        (e.g. by running :func:`scalex.SCALEX`).
    resolution : float, default 0.3
        Leiden resolution parameter. Higher values produce more clusters.
    n_neighbors : int, default 15
        Number of nearest neighbours for the k-NN graph.
    key_added : str, default 'leiden'
        Key in ``adata.obs`` where cluster labels are stored.
    use_rep : str, default 'latent'
        Key in ``adata.obsm`` to use as the cell embedding.
    random_state : int, default 0
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        The input ``adata`` with ``adata.obs[key_added]`` populated.

    Raises
    ------
    ValueError
        If ``use_rep`` is not found in ``adata.obsm``.

    Examples
    --------
    >>> import scalex
    >>> adata = scalex.SCALEX(['data1.h5ad', 'data2.h5ad'])
    >>> scalex.tl.cluster(adata, resolution=0.5)
    >>> adata.obs['leiden'].value_counts()
    """
    import scanpy as sc

    if use_rep not in adata.obsm:
        raise ValueError(
            f"Representation '{use_rep}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}. "
            "Run scalex.SCALEX() first to populate the latent embedding."
        )

    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors, random_state=random_state)
    sc.tl.leiden(adata, resolution=resolution, key_added=key_added, random_state=random_state)
    return adata

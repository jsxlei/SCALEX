"""PyTorch Dataset and DataLoader wrappers for single-cell data."""

import numpy as np
from anndata import AnnData
import scanpy as sc
from sklearn.preprocessing import MaxAbsScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from scalex.utils import DEFAULT_CHUNK_SIZE
from scalex.io.load import concat_data
from scalex.io.preprocess import preprocessing

CHUNK_SIZE = DEFAULT_CHUNK_SIZE


class BatchSampler(Sampler):
    """Batch-specific sampler ensuring each mini-batch is from one dataset.

    Parameters
    ----------
    batch_size : int
        Number of samples per mini-batch.
    batch_id : pd.Series
        Batch label for every sample in the dataset.
    drop_last : bool, default False
        Drop incomplete final batches.
    """

    def __init__(self, batch_size: int, batch_id, drop_last: bool = False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.batch_id = batch_id

    def __iter__(self):
        batch = {}
        sampler = np.random.permutation(len(self.batch_id))
        for idx in sampler:
            c = self.batch_id.iloc[idx]
            if c not in batch:
                batch[c] = []
            batch[c].append(idx)
            if len(batch[c]) == self.batch_size:
                yield batch[c]
                batch[c] = []

        for c in batch.keys():
            if len(batch[c]) > 0 and not self.drop_last:
                yield batch[c]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.batch_id) // self.batch_size
        return (len(self.batch_id) + self.batch_size - 1) // self.batch_size


class SingleCellDataset(Dataset):
    """PyTorch Dataset wrapping an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with ``obs['batch']`` set.
    use_layer : str, default 'X'
        Layer to use as input features. ``'X'`` uses ``adata.X``;
        otherwise looks in ``adata.layers`` then ``adata.obsm``.
    """

    def __init__(self, adata: AnnData, use_layer: str = 'X'):
        self.adata = adata
        self.shape = adata.shape
        self.use_layer = use_layer

    def __len__(self) -> int:
        return self.adata.shape[0]

    def __getitem__(self, idx):
        if self.use_layer == 'X':
            if isinstance(self.adata.X[idx], np.ndarray):
                x = self.adata.X[idx].squeeze().astype(float)
            else:
                x = self.adata.X[idx].toarray().squeeze().astype(float)
        elif self.use_layer in self.adata.layers:
            x = self.adata.layers[self.use_layer][idx]
        else:
            x = self.adata.obsm[self.use_layer][idx]

        domain_id = self.adata.obs['batch'].cat.codes.iloc[idx]
        return x, domain_id, idx


def load_data(
    data_list,
    batch_categories=None,
    profile: str = 'RNA',
    join: str = 'inner',
    batch_key: str = 'batch',
    batch_name: str = 'batch',
    groupby=None,
    subsets=None,
    min_features: int = 600,
    min_cells: int = 3,
    target_sum=None,
    n_top_features=None,
    min_cell_per_batch: int = 10,
    keep_mt: bool = False,
    backed: bool = False,
    batch_size: int = 64,
    chunk_size: int = CHUNK_SIZE,
    fraction=None,
    n_obs=None,
    processed: bool = False,
    log=None,
    num_workers: int = 4,
    use_layer: str = 'X',
):
    """Load and preprocess single-cell data, returning AnnData and DataLoaders.

    Parameters
    ----------
    data_list : list[str | AnnData]
        Paths or AnnData objects to load and concatenate.
    batch_categories : list[str] | None
        Batch labels, one per element of ``data_list``.
    profile : {'RNA', 'ATAC', 'ADT'}, default 'RNA'
        Single-cell data type.
    join : {'inner', 'outer'}, default 'inner'
        How to merge variable spaces across batches.
    batch_key : str, default 'batch'
        Key used to tag the batch origin in ``obs``.
    batch_name : str, default 'batch'
        Column in ``obs`` to use as the batch label for training.
    groupby : str | None
        Column in ``obs`` used together with ``subsets`` to filter cells.
    subsets : list | None
        Specific groups to keep (filtered via ``groupby``).
    min_features : int, default 600
        Minimum detected features per cell.
    min_cells : int, default 3
        Minimum cells per feature.
    target_sum : int | None
        Library size for normalization.
    n_top_features : int or list[str] | None
        Top features to retain.
    min_cell_per_batch : int, default 10
        Minimum cells per batch for HVG selection.
    keep_mt : bool, default False
        Keep mitochondrial genes.
    backed : bool, default False
        Open H5AD files in backed mode.
    batch_size : int, default 64
        Mini-batch size for the training DataLoader.
    chunk_size : int, default 20000
        Internal chunk size.
    fraction : float | None
        Fraction of cells to subsample.
    n_obs : int | None
        Fixed number of cells to subsample.
    processed : bool, default False
        Skip preprocessing and only apply MaxAbs scaling.
    log : logging.Logger | None
        Optional logger.
    num_workers : int, default 4
        Number of DataLoader worker processes.
    use_layer : str, default 'X'
        Layer to use as model input.

    Returns
    -------
    adata : AnnData
        Preprocessed annotated data.
    trainloader : DataLoader
        Shuffled DataLoader for training.
    testloader : DataLoader
        Batch-ordered DataLoader for inference.
    """
    import os
    adata = concat_data(data_list, batch_categories, join=join, batch_key=batch_key)

    if subsets is not None and groupby is not None:
        adata = adata[adata.obs[groupby].isin(subsets)].copy()
        if log:
            log.info('Subsets dataset shape: {}'.format(adata.shape))

    if log:
        log.info('Raw dataset shape: {}'.format(adata.shape))

    if batch_name != 'batch':
        if ',' in batch_name:
            names = batch_name.split(',')
            adata.obs['batch'] = adata.obs[names[0]].astype(str) + '_' + adata.obs[names[1]].astype(str)
        else:
            adata.obs['batch'] = adata.obs[batch_name]
    if 'batch' not in adata.obs:
        adata.obs['batch'] = 'batch'
    adata.obs['batch'] = adata.obs['batch'].astype('category')

    if isinstance(n_top_features, str):
        if os.path.isfile(n_top_features):
            n_top_features = np.loadtxt(n_top_features, dtype=str)
        else:
            n_top_features = int(n_top_features)

    if n_obs is not None or fraction is not None:
        sc.pp.subsample(adata, fraction=fraction, n_obs=n_obs)

    if not processed and use_layer == 'X':
        adata = preprocessing(
            adata,
            profile=profile,
            min_features=min_features,
            min_cells=min_cells,
            target_sum=target_sum,
            n_top_features=n_top_features,
            min_cell_per_batch=min_cell_per_batch,
            keep_mt=keep_mt,
            chunk_size=chunk_size,
            backed=backed,
            log=log,
        )
    else:
        if use_layer == 'X':
            adata.X = MaxAbsScaler().fit_transform(adata.X)
        elif use_layer in adata.layers:
            adata.layers[use_layer] = MaxAbsScaler().fit_transform(adata.layers[use_layer])
        elif use_layer in adata.obsm:
            adata.obsm[use_layer] = MaxAbsScaler().fit_transform(adata.obsm[use_layer])
        else:
            raise ValueError(
                f"Layer {use_layer!r} not found in adata.layers or adata.obsm. "
                f"Available layers: {list(adata.layers.keys())}, "
                f"obsm keys: {list(adata.obsm.keys())}."
            )

    scdata = SingleCellDataset(adata, use_layer=use_layer)
    trainloader = DataLoader(
        scdata,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers,
    )
    batch_sampler = BatchSampler(batch_size, adata.obs['batch'], drop_last=False)
    testloader = DataLoader(scdata, batch_sampler=batch_sampler, num_workers=num_workers)

    return adata, trainloader, testloader

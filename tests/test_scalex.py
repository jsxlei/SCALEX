import pytest

import scalex
from scalex.function import SCALEX
from scalex.net.vae import VAE
from scalex.data import preprocessing_rna

import torch

def test_preprocess_rna(adata_test):
    adata = adata_test.copy()
    adata = preprocessing_rna(adata, min_cells=4, min_features=0)
    assert adata.raw.shape == adata_test.shape


def test_scalex_forward(adata_test):
    n_domain = len(adata_test.obs['batch'].astype('category').cat.categories)
    x_dim = adata_test.X.shape[1]

    # model config
    enc = [['fc', 1024, 1, 'relu'],['fc', 10, '', '']]  # TO DO
    dec = [['fc', x_dim, n_domain, 'sigmoid']]
    model = VAE(enc, dec, n_domain=n_domain)
    
    x = torch.Tensor(adata_test.X)
    y = torch.LongTensor(adata_test.obs['batch'].values)
    z, recon_x, loss = model(x, y)
    # Print a summary of the AnnData object
    assert z.shape == (adata_test.shape[0], 10)
    assert recon_x.shape == adata_test.shape

    # Load the file


def test_full_model(adata_test):
    out = SCALEX(
        adata_test, processed=True, min_cells=0, min_features=0, batch_size=2, max_iteration=10,
    )
    assert 'distances' in out.obsp
    assert 'X_scalex_umap' in out.obsm


if __name__ == '__main__':
    pytest.main([__file__])

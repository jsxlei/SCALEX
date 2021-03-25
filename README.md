[![Stars](https://img.shields.io/github/stars/jsxlei/SCALE_v2?logo=GitHub&color=yellow)](https://github.com/jsxlei/scale_v2/stargazers)
[![PyPI](https://img.shields.io/pypi/v/scale-v2.svg)](https://pypi.org/project/scale-v2)
[![Documentation Status](https://readthedocs.org/projects/scale-v2/badge/?version=latest)](https://scale-v2.readthedocs.io/en/latest/?badge=stable)
[![Downloads](https://pepy.tech/badge/scale_v2)](https://pepy.tech/project/scale_v2)
# SCALE v2: Single-cell integrative Analysis via latent Feature Extraction 

## [Documentation](https://scale-v2.readthedocs.io/en/latest/index.html) 

## Installation  	
#### install from PyPI

    pip install scale-v2
    
#### install from GitHub

	git clone git://github.com/jsxlei/scale_v2.git
	cd scale_v2
	python setup.py install
    
SCALE v2 is implemented in [Pytorch](https://pytorch.org/) framework.  
Running SCALE v2 on CUDA is recommended if available.   
Installation only requires a few minutes.  

## Quick Start

SCALE v2 can both used under command line and API function in jupyter notebook


### 1. Command line

    SCALE.py --data_list data1 data2 --batch_categories batch1 batch2 
    
#### Parameters

    data_list
        A list of matrices file (each as a `batch` or a single batch/batch-merged file.
    batch_categories
        Categories for the batch annotation. By default, use increasing numbers.
    profile
        Specify the single-cell profile, RNA or ATAC. Default: RNA.
    join
        Use intersection ('inner') or union ('outer') of variables of different batches. 
    batch_key
        Add the batch annotation to obs using this key. By default, batch_key='batch'.
    batch_name
        Use this annotation in obs as batches for training model. Default: 'batch'.
    min_features
        Filtered out cells that are detected in less than min_features. Default: 600.
    min_cells
        Filtered out genes that are detected in less than min_cells. Default: 3.
    n_top_features
        Number of highly-variable genes to keep. Default: 2000.
    batch_size
        Number of samples per batch to load. Default: 64.
    lr
        Learning rate. Default: 2e-4.
    max_iteration
        Max iterations for training. Training one batch_size samples is one iteration. Default: 30000.
    seed
        Random seed for torch and numpy. Default: 124.
    gpu
        Index of GPU to use if GPU is available. Default: 0.
    outdir
        Output directory. Default: 'output/'.
    projection
        Use for new dataset projection. Input the folder containing the pre-trained model. Default: None. 
    impute
        If True, calculate the imputed gene expression and store it at adata.layers['impute']. Default: False.
    chunk_size
        Number of samples from the same batch to transform. Default: 20000.
    ignore_umap
        If True, do not perform UMAP for visualization and leiden for clustering. Default: False.
    verbose
        Verbosity, True or False. Default: False.
    

#### Output
Output will be saved in the output folder including:
* **checkpoint**:  saved model to reproduce results cooperated with option --checkpoint or -c
* **[adata.h5ad](https://anndata.readthedocs.io/en/stable/anndata.AnnData.html#anndata.AnnData)**:  preprocessed data and results including, latent, clustering and imputation
* **umap.png**:  UMAP visualization of latent representations of cells 
* **log.txt**:  log file of training process

     
#### Useful options  
* output folder for saveing results: [-o] or [--outdir] 
* filter rare genes, default 3: [--min_cell]
* filter low quality cells, default 600: [--min_gene]  
* select the number of highly variable genes, keep all genes with -1, default 2000: [--n_top_genes]
	
    
#### Help
Look for more usage of SCALE v2

	SCALE.py --help 
    
    
### 2. API function

    from scale import SCALE
    adata = SCALE(data_list, batch_categories)
    
Function of parameters are similar to command line options.
Output is a Anndata object for further analysis with scanpy.
    
    
## [Tutorial](https://scale-v2.readthedocs.io/en/latest/tutorial/index.html) 


## Previous version [SCALE](https://github.com/jsxlei/SCALE)

Previous SCALE for single-cell ATAC-seq analysis is still available in SCALE v2 by command line or api.

### Command line

    SCALE.py -d data --version 1
    
### API

    from scale.extensions import SCALE_v1
    SCALE_v1(data)
    
    
All the usage is the same with previous SCALE version 1.

[![Stars](https://img.shields.io/github/stars/jsxlei/scalex?logo=GitHub&color=yellow)](https://github.com/jsxlei/scalex/stargazers)
[![PyPI](https://img.shields.io/pypi/v/scalex.svg)](https://pypi.org/project/scalex)
[![Documentation Status](https://readthedocs.org/projects/scalex/badge/?version=latest)](https://scalex.readthedocs.io/en/latest/?badge=stable)
[![Downloads](https://pepy.tech/badge/scalex)](https://pepy.tech/project/scalex)
# SCALEX: Single-cell integrative Analysis via latent Feature Extraction 

## [Documentation](https://scalex.readthedocs.io/en/latest/index.html) 

## Installation  	
#### install from PyPI

    pip install scalex
    
#### install from GitHub

	git clone git://github.com/jsxlei/scalex.git
	cd scalex
	python setup.py install
    
SCALEX is implemented in [Pytorch](https://pytorch.org/) framework.  
Running SCALEX on CUDA is recommended if available.   
Installation only requires a few minutes.  

## Quick Start

SCALEX can both used under command line and API function in jupyter notebook


### 1. Command line

    SCALE.py --data_list data1 data2 dataN --batch_categories batch1 batch2 batchN 
    
#### Option

* --**data_list**  
        A list of matrices file (each as a `batch`) or a single batch/batch-merged file.
* --**batch_categories**  
        Categories for the batch annotation. By default, use increasing numbers if not given
* --**profile**  
        Specify the single-cell profile, RNA or ATAC. Default: RNA.
* --**min_features**  
        Filtered out cells that are detected in less than min_features. Default: 600 for RNA, 100 for ATAC.
* --**min_cells**  
        Filtered out genes that are detected in less than min_cells. Default: 3.
* --**n_top_features**  
        Number of highly-variable genes to keep. Default: 2000 for RNA, 30000 for ATAC.
* --**outdir**  
        Output directory. Default: 'output/'.
* --**projection**  
        Use for new dataset projection. Input the folder containing the pre-trained model. Default: None. 
* --**impute**  
        If True, calculate the imputed gene expression and store it at adata.layers['impute']. Default: False.
* --**chunk_size**  
        Number of samples from the same batch to transform. Default: 20000.
* --**ignore_umap**  
        If True, do not perform UMAP for visualization and leiden for clustering. Default: False.
* --**join**  
        Use intersection ('inner') or union ('outer') of variables of different batches. 
* --**batch_key**  
        Add the batch annotation to obs using this key. By default, batch_key='batch'.
* --**batch_name**  
        Use this annotation in obs as batches for training model. Default: 'batch'.
* --**batch_size**  
        Number of samples per batch to load. Default: 64.
* --**lr**  
        Learning rate. Default: 2e-4.
* --**max_iteration**  
        Max iterations for training. Training one batch_size samples is one iteration. Default: 30000.
* --**seed**  
        Random seed for torch and numpy. Default: 124.
* --**gpu**  
        Index of GPU to use if GPU is available. Default: 0.
* --**verbose**  
        Verbosity, True or False. Default: False.
    

#### Output
Output will be saved in the output folder including:
* **checkpoint**:  saved model to reproduce results cooperated with option --checkpoint or -c
* **[adata.h5ad](https://anndata.readthedocs.io/en/stable/anndata.AnnData.html#anndata.AnnData)**:  preprocessed data and results including, latent, clustering and imputation
* **umap.png**:  UMAP visualization of latent representations of cells 
* **log.txt**:  log file of training process

     
#### Useful options  
* output folder for saveing results: [-o] or [--outdir] 
* filter rare genes, default 3: [--min_cells]
* filter low quality cells, default 600: [--min_features]  
* select the number of highly variable genes, keep all genes with -1, default 2000: [--n_top_featuress]
	
    
#### Help
Look for more usage of SCALEX

	SCALEX.py --help 
    
    
### 2. API function

    from scalex import SCALEX
    adata = SCALEX(data_list, batch_categories)
    
Function of parameters are similar to command line options.
Output is a Anndata object for further analysis with scanpy.
    
    
## [Tutorial](https://scalex.readthedocs.io/en/latest/tutorial/index.html) 


## Previous version [SCALE](https://github.com/jsxlei/SCALE)

Previous SCALE for single-cell ATAC-seq analysis is still available in SCALEX by command line (--version 1) or api (SCALE_v1).

### Command line

    SCALEX.py -d data --version 1
    
### API

    from scale.extensions import SCALE_v1
    SCALE_v1(data)
    
    
All the usage is the same with previous SCALE version 1.

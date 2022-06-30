[![Stars](https://img.shields.io/github/stars/jsxlei/scalex?logo=GitHub&color=yellow)](https://github.com/jsxlei/scalex/stargazers)
[![PyPI](https://img.shields.io/pypi/v/scalex.svg)](https://pypi.org/project/scalex)
[![Documentation Status](https://readthedocs.org/projects/scalex/badge/?version=latest)](https://scalex.readthedocs.io/en/latest/?badge=stable)
[![Downloads](https://pepy.tech/badge/scalex)](https://pepy.tech/project/scalex)
# [Online single-cell data integration through projecting heterogeneous datasets into a common cell-embedding space](https://www.biorxiv.org/content/10.1101/2021.04.06.438536v2)

![](docs/source/_static/img/scalex.jpg)


## [Documentation](https://scalex.readthedocs.io/en/latest/index.html) 

## Installation  	
#### install from PyPI

    pip install scalex
    
#### install from GitHub

	git clone git://github.com/jsxlei/scalex.git
	cd scalex
	python setup.py install
    
SCALEX is implemented in [Pytorch](https://pytorch.org/) framework.  
SCALEX can be run on CPU devices, and running SCALEX on GPU devices if available is recommended.   

## Quick Start

SCALEX can both used under command line and API function in jupyter notebook


### 1. Command line
#### Standard usage


    SCALEX.py --data_list data1 data2 dataN --batch_categories batch_name1 batch_name2 batch_nameN 
    
    
`--data_list`: data path of each batch of single-cell dataset, use `-d` for short

`--batch_categories`: name of each batch, batch_categories will range from 0 to N if not specified

    

#### Use h5ad file storing `anndata` as input, one or multiple separated files

    SCALEX.py --data_list <filename.h5ad>

#### Specify batch in `anadata.obs` using `--batch_name` if only one concatenated h5ad file provided, batch_name can be e.g. conditions, samples, assays or patients, default is `batch`

    SCALEX.py --data_list <filename.h5ad> --batch_name <specific_batch_name>
    
    
#### Integrate heterogenous scATAC-seq datasets, add option `--profile` ATAC
        
    SCALEX.py --data_list <filename.h5ad> --profile ATAC
    
#### Inputation simultaneously along with Integration, add option `--impute`, results are stored at anndata.layers['impute']

    SCALEX.py --data_list <atac_filename.h5ad> --profile ATAC --impute
    
    
#### Custom features through `--n_top_features` a filename contains features in one column format read

    SCALEX.py --data_list <filename.h5ad> --n_top_features features.txt
    
    
#### Use preprocessed data `--processed`

    SCALEX.py --data_list <filename.h5ad> --processed

#### Option

* --**data_list**  
        A list of matrices file (each as a `batch`) or a single batch/batch-merged file.
* --**batch_categories**  
        Categories for the batch annotation. By default, use increasing numbers if not given
* --**batch_name**  
        Use this annotation in anndata.obs as batches for training model. Default: 'batch'.
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


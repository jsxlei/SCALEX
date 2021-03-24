# SCALE v2: Single-cell integrative Analysis via latent Feature Extraction 

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


### 1. Command line

    SCALE.py --data_list data1 data2 --batch_categories batch1 batch2 
    
    data_list: data path of each batch of single-cell dataset
    batch_categories: name of each batch
    

#### Output
Output will be saved in the output folder including:
* **checkpoint**:  saved model to reproduce results cooperated with option --checkpoint or -c
* **adata.h5ad**:  preprocessed data and results including, latent, clustering and imputation
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
    
#### Tutorial

See document 


## Previous version [SCALE](https://github.com/jsxlei/SCALE)

SCALE is still available in SCALE v2

### Command line

    SCALE.py --d data --version 1
    
### API

    from scale.extensions import SCALE_v1
    SCALE_v1(data)
    
    
All the usage is the same with previous SCALE version 1.
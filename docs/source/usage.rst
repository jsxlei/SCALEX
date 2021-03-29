Usage
----------------

SCALEX provide both commanline tool and api function used in jupyter notebook   

Command line
^^^^^^^^^^^^
Run SCALEX after installation::

    SCALEX.py --data_list data1 data2 --batch_categories batch1 batch2 
    
``data_list``: data path of each batch of single-cell dataset  

``batch_categories``: name of each batch, batch_categories will range from 0 to N if not specified
    
Input
~~~~~
Input can be one of following:  

* single file of format h5ad, csv, txt, mtx or their compression file  
* multiple files of above format  

.. note:: h5ad file input
* SCALEX will use the ``batch`` column in the obs of adata format read from h5ad file as batch information  
* Users can specify any columns in the obs with option: ``--batch_name`` name
* If multiple inputs are given, SCALEX can take each file as individual batch by default, and overload previous batch information, users can change the concat name via option ``--batch_key`` other_name

Output
~~~~~~~~~~~
Output will be saved in the output folder including:  

* **checkpoint**:  saved model to reproduce results cooperated with option ``--checkpoint`` or -c
* **adata.h5ad**:  preprocessed data and results including, latent, clustering and imputation
* **umap.png**:  UMAP visualization of latent representations of cells 
* **log.txt**:  log file of training process

     
Useful options 
~~~~~~~~~~~~~~
* output folder for saveing results: [-o] or [--outdir] 
* filter rare genes, default 3: [--min_cell]
* filter low quality cells, default 600: [--min_gene]  
* select the number of highly variable genes, keep all genes with -1, default 2000: [--n_top_genes]
	
    
Help
~~~~
Look for more usage of SCALEX::

	SCALEX.py --help 
    
    
API function
^^^^^^^^^^^^
Use SCALEX in jupyter notebook::

    from scalex.function import SCALEX
    adata = SCALEX(data_list, batch_categories)
    
Function of parameters are similar to command line options.
Output is a Anndata object for further analysis with scanpy.
    



AnnData
^^^^^^^
SCALEX supports :mod:`scanpy` and :mod:`anndata`, which provides the :class:`~anndata.AnnData` class.

.. image:: http://falexwolf.de/img/scanpy/anndata.svg
   :width: 300px

At the most basic level, an :class:`~anndata.AnnData` object `adata` stores
a data matrix `adata.X`, annotation of observations
`adata.obs` and variables `adata.var` as `pd.DataFrame` and unstructured
annotation `adata.uns` as `dict`. Names of observations and
variables can be accessed via `adata.obs_names` and `adata.var_names`,
respectively. :class:`~anndata.AnnData` objects can be sliced like
dataframes, for example, `adata_subset = adata[:, list_of_gene_names]`.
For more, see this `blog post`_.

.. _blog post: http://falexwolf.de/blog/171223_AnnData_indexing_views_HDF5-backing/

To read a data file to an :class:`~anndata.AnnData` object, call::

    import scanpy as sc
    adata = sc.read(filename)

to initialize an :class:`~anndata.AnnData` object. Possibly add further annotation using, e.g., `pd.read_csv`::

    import pandas as pd
    anno = pd.read_csv(filename_sample_annotation)
    adata.obs['cell_groups'] = anno['cell_groups']  # categorical annotation of type pandas.Categorical
    adata.obs['time'] = anno['time']                # numerical annotation of type float
    # alternatively, you could also set the whole dataframe
    # adata.obs = anno

To write, use::

    adata.write(filename)
    adata.write_csvs(filename)
    adata.write_loom(filename)


.. _Seaborn: http://seaborn.pydata.org/
.. _matplotlib: http://matplotlib.org/
.. module:: scalex
.. automodule:: scalex
   :noindex:
   
   
API
====


Import SCALEX::

   import scalex
   
   
Function
--------
.. module:: scalex
.. currentmodule:: scalex

.. autosummary::
    :toctree: .
    
    SCALEX
    label_transfer


Data
-------------------
.. module:: scalex.data
.. currentmodule:: scalex


Load data
~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: .
    
    data.load_data
    data.concat_data
    data.load_files
    data.load_file
    data.read_mtx
    
   
Preprocessing
~~~~~~~~~~~~~
.. autosummary::
    :toctree: .
   
    data.preprocessing
    data.preprocessing_rna
    data.preprocessing_atac
    data.batch_scale
    data.reindex
   
   
DataLoader
~~~~~~~~~~
.. autosummary::
    :toctree: .
    
    data.SingleCellDataset
    data.BatchSampler
    
    
    
Net
-----------
.. module:: scalex.net
.. currentmodule:: scalex


Model
~~~~~
.. autosummary::
    :toctree: .
    
    net.vae.VAE
    
    
Layer
~~~~~
.. autosummary::
    :toctree: .
    
    net.layer.DSBatchNorm
    net.layer.Block
    net.layer.NN
    net.layer.Encoder
    
    
Loss
~~~~
.. autosummary::
    :toctree: .
    
    net.loss.kl_div
    net.loss.binary_cross_entropy
    
    
Utils
~~~~~
.. autosummary::
    :toctree: .
    
    net.utils.onehot
    net.utils.EarlyStopping
    
    

Plot
------------
.. module:: scalex.plot
.. currentmodule:: scalex


.. autosummary::
    :toctree: .
    
    plot.embedding
    plot.plot_meta
    plot.plot_meta2
    plot.plot_confusion


Metric
----------------
.. module:: scalex.metric
.. currentmodule:: scalex


Collections of useful measurements for evaluating results.

.. autosummary::
   :toctree: .

   metrics.batch_entropy_mixing_score
   metrics.silhouette_score


Logger
------
.. module:: scalex.logger
.. currentmodule:: scalex

.. autosummary::
    :toctree: .
    
    logger.create_logger
   

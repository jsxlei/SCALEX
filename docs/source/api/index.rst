.. module:: scale
.. automodule:: scale
   :noindex:
   
   
API
====


Import SCALE as::

   import scale
   
   
Function
--------
.. module:: scale
.. currentmodule:: scale

.. autosummary::
    :toctree: .
    
    SCALE
    label_transfer


Data
-------------------
.. module:: scale.data
.. currentmodule:: scale


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
.. module:: scale.net
.. currentmodule:: scale


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
.. module:: scale.plot
.. currentmodule:: scale


.. autosummary::
    :toctree: .
    
    plot.embedding
    plot.plot_meta
    plot.plot_meta2
    plot.plot_confusion


Metric
----------------
.. module:: scale.metric
.. currentmodule:: scale


Collections of useful measurements for evaluating results.

.. autosummary::
   :toctree: .

   metrics.batch_entropy_mixing_score
   metrics.silhouette_score


Logger
------
.. module:: scale.logger
.. currentmodule:: scale

.. autosummary::
    :toctree: .
    
    logger.create_logger
   

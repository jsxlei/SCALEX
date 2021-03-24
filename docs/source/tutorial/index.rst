Tutorial
=========

`Integration <../tutorial/Integration_PBMC.ipynb>`_
--------------------

before integration

.. image:: ../_static/img/pbmc_before_integration.png
   :width: 500px

after SCALE v2 integration

.. image:: ../_static/img/pbmc_integration.png
   :width: 600px


`Projection <../tutorial/Projection_pancreas.ipynb>`_
-------------

Map new data to the embeddings of reference

A pancreas reference was created by integrating eight batches.

Here, map pancreas_gse81547, pancreas_gse83139 and pancreas_gse114297 to the embeddings of pancreas reference.

.. image:: ../_static/img/pancreas_projection.png
    :width: 600px


Label transfer
---------------
Annotate cells in new data through label transfer

Label transfer tabula muris data and mouse kidney data from mouse atlas reference

mouse atlas reference

.. image:: ../_static/img/mouse_atlas_reference.png
    :width: 400px
     
query tabula muris aging and query mouse kidney
    
.. image:: ../_static/img/query_tabula_muris_aging.png
    :width: 300px

.. image:: ../_static/img/query_kidney.png
    :width: 300px
    
    
`Integration scATAC-seq data <Integration_scATAC-seq.ipynb>`_
---------------

.. image:: ../_static/img/scATAC_integration.png
    :width: 600px


Integration cross-modality data
-------------------------------
Integrate scRNA-seq and scATAC-seq dataset

.. image:: ../_static/img/cross-modality_integration.png
    :width: 600px
    

Spatial data (To be updated)
------------
Integrating spatial data with scRNA-seq


Examples
--------

.. toctree::
    :maxdepth: 2
    :hidden:
    
    Integration_PBMC
    Projection_pancreas
    Integration_scATAC-seq
    
    

   







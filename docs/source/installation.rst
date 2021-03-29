Installation
------------

PyPI install
~~~~~~~~~~~~

Pull SCALE from `PyPI <https://pypi.org/project/scalex>`__ (consider using ``pip3`` to access Python 3)::

    pip install scalex

.. _from PyPI: https://pypi.org/project/scalex


Pytorch
~~~~~~~
If you have cuda devices, consider install Pytorch_ cuda version.::

    conda install pytorch torchvision torchaudio -c pytorch
    
.. _Pytorch: https://pytorch.org/

Troubleshooting
~~~~~~~~~~~~~~~


Anaconda
~~~~~~~~
If you do not have a working installation of Python 3.6 (or later), consider
installing Miniconda_ (see `Installing Miniconda`_). 


Installing Miniconda
~~~~~~~~~~~~~~~~~~~~
After downloading Miniconda_, in a unix shell (Linux, Mac), run

.. code:: shell

    cd DOWNLOAD_DIR
    chmod +x Miniconda3-latest-VERSION.sh
    ./Miniconda3-latest-VERSION.sh

and accept all suggestions.
Either reopen a new terminal or `source ~/.bashrc` on Linux/ `source ~/.bash_profile` on Mac.
The whole process takes just a couple of minutes.

.. _Miniconda: http://conda.pydata.org/miniconda.html


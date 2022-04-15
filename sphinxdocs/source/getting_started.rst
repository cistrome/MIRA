Getting Started
===============

------------

System Requirements
-------------------

* Linux or MacOS
* Python >=3.5, <3.8
* (optional) CUDA-enabled GPU

Data
----

.. image :: /_static/data_example.png
    :width: 350
    :align: center

MIRA takes scRNA-seq and scATAC-seq count matrices from a single-cell multiomics experiment,
where each cell is measured using both assays, and measurements are linked by a shared cell
barcode. We demonstrated MIRA using SHARE-seq data and commercial 10X genomics multiome data, 
but MIRA's assumptions and models are extensible to other multiome protocols.

Since MIRA starts from count matrices, one can use any method for read preprocessing and 
cell QC. Of note, we find that CellRanger's ATAC-seq peak-calling method finds fewer
and less-specific peaks than `MACS2 <https://github.com/macs3-project/MACS>`_. This contributes to lower resolution manifolds
during topic modeling. 

Installation
------------

MIRA can be installed from either `Conda <https://anaconda.org/liulab-dfci/mira-multiome>`_ 
or  `PyPI <https://pypi.org/project/mira-multiome>`_:

.. code-block :: bash
    
    conda install -c conda-forge -c bioconda -c pytorch mira-multiome

or

.. code-block:: bash

    pip install mira-multiome

Installation will take about a minute.

Please note, currently MIRA may only be installed on Python <3.8 due to some dependencies' requirements. 
We are working to make it accessible on newer Python versions. 
To set up an a new analysis, we recommend starting with a fresh environment:

.. code-block:: bash

    conda create --name mira-env -c conda-forge -c pytorch -c bioconda mira-multiome scanpy jupyter leidenalg
    conda activate mira-env
    python -m ipykernel install --user --name mira-env

To use the environment in a jupyter notebook, start the notebook server, then go to Kernel > Change kernel > mira-env.


Installing with GPU support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training on a GPU reduces the training time of MIRA topic models.
To install MIRA with PyTorch compiled with GPU support, first install MIRA, as above. Then, follow instructions 
at `pytorch.org <https://pytorch.org/get-started/locally/>`_ to find the version of PyTorch that suits your system.

Learning Curve
--------------

.. image:: /_static/code_example.png
    :width: 600
    :align: center

If you have experience with Scanpy, we structured MIRA to follow similar conventions 
so that it would feel familiar and intuitive. In fact, most MIRA analyses
seamlessly weave between MIRA and Scanpy functionalities for cleaning, slicing,
and plotting the data. In general, the first positional argument of a MIRA 
function is an AnnData object, and the following keyword arguments change 
how the function transforms that object. 


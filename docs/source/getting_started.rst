Getting Started
===============

------------

.. image:: https://badge.fury.io/py/mira-multiome.svg
    :target: https://badge.fury.io/py/mira-multiome

.. image:: https://readthedocs.org/projects/mira-multiome/badge/?version=latest&style=plastic

.. image:: https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg
    :target: https://codeocean.com/capsule/6761625/tree

.. image:: https://img.shields.io/conda/pn/liulab-dfci/mira-multiome
    :target: https://anaconda.org/bioconda/mira-multiome

.. image:: https://zenodo.org/badge/DOI/10.1101/2021.12.06.471401.svg
    :target: https://www.nature.com/articles/s41592-022-01595-z


MIRA versus CODAL
-----------------

If you are looking to use the CODAL method for comparative single-cell atlas construction,
you're in the right place. MIRA's topic modeling methods were originally developed for single-cell
multiomic analysis, but work identically for non-multiome single-cell experiments. Now, MIRA's topic models
perform inference using the CODAL objective for disentangling biological effects from technical
effects in batched data. 


System Requirements
-------------------

* Linux or MacOS
* Python >=3.5
* (optional) CUDA-enabled GPU

Data
----

**Single-cell Multiome**

.. image :: /_static/data_example.png
    :width: 350
    :align: center

MIRA takes scRNA-seq and scATAC-seq count matrices from a single-cell multiomics experiment or experiments (batched),
where each cell is measured using both assays, and measurements are linked by a shared cell
barcode. We demonstrated MIRA using SHARE-seq data and commercial 10X genomics multiome data, 
but MIRA's assumptions and models are extensible to other multiome protocols.

**scRNA-seq or scATAC-seq only**

When working with non-multiomic data, some of MIRA's functionalities are limited. However, one can use MIRA's 
topic models to analyze single-mode datasets. Again, MIRA needs a count matrix as input.

**Notes on Preprocessing**

Since MIRA starts from count matrices, one can use any method for read preprocessing and 
cell QC. Of note, we find that CellRanger's ATAC-seq peak-calling method finds fewer
and less-specific peaks than `MACS2 <https://github.com/macs3-project/MACS>`_. This contributes to lower resolution manifolds
during topic modeling. 

Installation
------------

MIRA can be installed from `PyPI <https://pypi.org/project/mira-multiome>`_:

.. code-block:: bash

    pip install mira-multiome

Or from bioconda:

.. code-block:: bash

    conda install -c bioconda mira-multiome

Installation will take about a minute.

To set up an a new analysis, we recommend starting with a fresh environment:

.. code-block:: bash

    conda create --name mira-env -c conda-forge -c bioconda scanpy jupyter leidenalg
    conda activate mira-env
    conda install -c bioconda mira-multiome
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


.. MIRA documentation master file, created by
   sphinx-quickstart on Tue Jan  4 12:00:13 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/mira_logo.png
  :width: 400
  :alt: MIRA
  :align: center

------------

**Multimodal models for Integrated Regulatory Analysis**, or MIRA,  is a python package for analyzing
the dynamic processes of gene regulation using single-cell multiomics datasets. 

MIRA works on top of `Scanpy <https://scanpy.readthedocs.io/en/stable/>`_ and `Anndata <https://anndata.readthedocs.io/en/latest/>`_
to provide a rich framework integrating accessibility and expression data for more insights
into your data. MIRA includes methods for:

* Topic modeling of expression and accessibility
* Construction of a joint representation
* Pseudotime trajectory inference
* *Cis*-regulatory modeling

Install
-------

Install MIRA using:

**Anaconda**

.. code-block:: python
   
   >>> conda install -c conda-forge -c bioconda -c liulab-dfci mira-multiome

**PyPI**

.. code-block:: python

   >>> pip install mira-multiome


Documentation
-------------

.. toctree::
   :maxdepth: 2

   getting_started
   tutorials
   api
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

User Guide
==========

------------

Setting up
----------

To use MIRA, you need a single-cell multiomics, scRNA-seq, or scATAC-seq experiment preprocessed and formatted
as a **peaks by cells count matrix for scATAC-seq** and **genes by cells count matrix
for scRNA-seq**. Before starting with MIRA, make sure to filter low quality cells, doublets, etc.

These tutorials also depend on `scanpy`. To install:

.. code-block :: bash

   $ conda install -c conda-forge -c bioconda scanpy


Tutorials
---------

The following tutorials cover the main functionalities of MIRA, starting from
initial steps - topic modeling - and culminating in the identification of 
divergences between local chromatin accessibility and gene expression across
genes and cell states. 

.. toctree::
   :maxdepth: 2

   notebooks/tutorial_CODAL.ipynb
   notebooks/tutorial_atlas_integration.ipynb
   notebooks/tutorial_joint_representation.ipynb
   notebooks/tutorial_topic_analysis.ipynb
   notebooks/tutorial_mira.time.ipynb
   notebooks/tutorial_cisregulatory_modeling.ipynb
   notebooks/tutorial_NITE_LITE_modeling.ipynb

Other Features
--------------

These tutorials cover functionalities of MIRA that are useful, but
outside of the main analysis track. Learn how to make beautiful
and informative streamgraphs and use MIRA plotting functions with velocity data.

.. toctree::
   :maxdepth: 2

   notebooks/tutorial_streamgraphs.ipynb
   notebooks/tutorial_cellrank.ipynb
   tutorials/enrichr.rst
   notebooks/tutorial_assemble_gene_annotations.ipynb


Examples
--------

We will add more examples of MIRA analyses in the future! These examples
may be run in Google Colab for an interactive experience.

.. toctree::
   :maxdepth: 1

   notebooks/tutorial_mouse_brain.ipynb
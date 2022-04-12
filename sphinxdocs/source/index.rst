.. MIRA documentation master file, created by
   sphinx-quickstart on Tue Jan  4 12:00:13 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/mira_logo.png
  :width: 400
  :alt: MIRA
  :align: center

------------

.. _home:

.. code-block :: bash
   
   conda install -c conda-forge -c bioconda -c pytorch mira-multiome

**Multimodal models for Integrated Regulatory Analysis**, or MIRA,  is a python package for analyzing
the dynamic processes of gene regulation using single-cell multiomics datasets. 

MIRA works on top of `Scanpy <https://scanpy.readthedocs.io/en/stable/>`_ and `Anndata <https://anndata.readthedocs.io/en/latest/>`_
to provide a rich, comprehensive framework integrating accessibility and expression data for more insights
into your data. MIRA includes methods for:

.. panels::
   :card: + text-center
   
   ---
   :img-top: _static/tensorboard_hparams.png

   .. link-button:: notebooks/tutorial_topic_model_tuning_full
        :type: ref
        :text: Multimodel Topic Modeling
        :classes: btn-link stretched-link font-weight-bold

   ---
   :img-top: _static/pseudotime/mira.time.trace_differentiation.gif

   .. link-button:: notebooks/tutorial_mira.time
        :type: ref
        :text: Constructing a Joint Representation
        :classes: btn-link stretched-link font-weight-bold

   ---
   :img-top: _static/mira.topics.AccessibilityModel.plot_compare_topic_enrichments.svg

   .. link-button:: notebooks/tutorial_mira.time
        :type: ref
        :text: Regulator and Functional Enrichment
        :classes: btn-link stretched-link font-weight-bold

   ---
   :img-top: _static/pseudotime/mira.time.trace_differentiation.gif

   .. link-button:: notebooks/tutorial_mira.time
        :type: ref
        :text: Pseudotime Trajectory inference
        :classes: btn-link stretched-link font-weight-bold

   ---
   :img-top:  _static/mira.pl.plot_chromatin_differential.png

   .. link-button:: notebooks/tutorial_mira.time
        :type: ref
        :text: Regulatory Potential Modeling
        :classes: btn-link stretched-link font-weight-bold

   ---
   :img-top:  _static/mira.pl.plot_chromatin_differential.png

   .. link-button:: notebooks/tutorial_mira.time
        :type: ref
        :text: Finding NITE regulation: Divergent Cis-Accessibility and Gene Expression
        :classes: btn-link stretched-link font-weight-bold

\.\.\. And more! Check out the `MIRA preprint <https://www.biorxiv.org/content/10.1101/2021.12.06.471401v1.full.pdf>`_ on bioarxiv. 
Also, see :doc:`getting_started` for installation and data preparation instructions.

------------

Docs
----

.. toctree::
   :maxdepth: 2

   home
   getting_started
   tutorials
   api
   faq

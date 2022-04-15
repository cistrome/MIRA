Enrichr Tutorial
----------------

Functions for posting genelists, retreiving ontologies, and
plotting results from `Enrichr <https://maayanlab.cloud/Enrichr/>`_.
The basic workflow is:

#. Post genelist
#. Download results from ontologies
#. Plot and analyze

Example
~~~~~~~

First, post your genelist to the Enrichr website:

.. code-block:: python

    >>> mira.tl.post_genelist(mira.tools.enrichr_enrichments.example_genelist)
    44149645

This function returns a post key, which may be used to retrieve
results:

.. code-block:: python

    >>> results = mira.tl.fetch_ontologies(44149645, 
    ... ontologies=['WikiPathways_2019_Mouse','BioPlanet_2019'])

You can manually provide ontologies of interest, or use MIRA's
default: `mira.tl.LEGACY_ONTOLOGIES`. The results are given as a 
python dictionary, and may be plotted using:

.. code-block:: python
    
    >>> mira.pl.plot_enrichments(results, plots_per_row = 1)

.. image:: /_static/mira.topics.ExpressionTopicModel.plot_enrichments.svg
            :width: 1200

Functions
~~~~~~~~~

.. autofunction:: mira.tl.post_genelist

.. autofunction:: mira.tl.fetch_ontologies

.. autofunction:: mira.tl.fetch_ontology

.. autofunction:: mira.pl.plot_ontologies
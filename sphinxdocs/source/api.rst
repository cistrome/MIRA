API
===

.. module:: mira
.. currentmodule:: mira

Topic Modeling
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: topicmodeling
   :template: topic_model.rst

   mira.topics.ExpressionTopicModel

.. autosummary::
   :toctree: topicmodeling
   :template: atac_topic_model.rst

   mira.topics.AccessibilityTopicModel

.. autosummary::
   :toctree: topicmodeling

   mira.topics.TopicModelTuner


Plotting
~~~~~~~~

.. autosummary::
   :toctree: plotting

   mira.pl.plot_stream

Tools
~~~~~

.. autosummary::
   :toctree: Tools

   mira.tl.get_motif_hits_in_peaks
   mira.tl.get_ChIP_hits_in_peaks
   mira.tl.get_distance_to_TSS
   mira.tl.post_genelist
   mira.tl.fetch_ontology
   mira.tl.fetch_ontologies
   mira.tl.get_NITE_score_genes
   mira.tl.get_NITE_score_cells
   mira.tl.get_chromatin_differential
   mira.tl.get_chromatin_differential
API
===

------------

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
   

Regulatory Potential Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: rpmodeling
   :template: lite_model.rst

   mira.rp.LITE_Model

.. autosummary::
   :toctree: rpmodeling
   :template: rp_model.rst

   mira.rp.NITE_Model

.. autosummary:: 
   :toctree: rpmodeling
   :template: gene_rp_model.rst

   mira.rp_model.rp_model.GeneModel

Pseudotime
~~~~~~~~~~

.. autosummary::
   :toctree: time
   :template: pseudotime.rst

   mira.time.normalize_diffmap
   mira.pl.plot_eigengap
   mira.time.get_connected_components
   mira.time.get_transport_map
   mira.time.find_terminal_cells
   mira.time.get_branch_probabilities
   mira.time.get_tree_structure
   mira.time.trace_differentiation

Plotting
~~~~~~~~

.. autosummary::
   :toctree: plotting

   mira.pl.plot_stream
   mira.pl.plot_chromatin_differential
   mira.pl.plot_scatter_chromatin_differential
   mira.pl.plot_enrichments
   mira.pl.compare_driver_TFs_plot

Tools
~~~~~

.. autosummary::
   :toctree: tools

   mira.tl.get_motif_hits_in_peaks
   mira.tl.get_ChIP_hits_in_peaks
   mira.tl.post_genelist
   mira.tl.fetch_ontology
   mira.tl.fetch_ontologies
   mira.tl.get_distance_to_TSS
   mira.tl.get_NITE_score_genes
   mira.tl.get_NITE_score_cells
   mira.tl.get_chromatin_differential


Utils/Accessors
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: utils

   mira.utils.make_joint_representation
   mira.utils.wide_view
   mira.utils.pretty_sderr
   mira.utils.subset_factors
   mira.utils.fetch_TSS_data
   mira.utils.fetch_gene_TSS_distances
   mira.utils.fetch_factor_meta
   mira.utils.fetch_factor_hits
   mira.utils.fetch_binding_sites
   mira.utils.show_gif


Datasets
~~~~~~~~

.. autosummary::
   :toctree: datasets

   mira.datasets.ShareseqSkin_Ma2020
   mira.datasets.FrankenCell_RNA
   mira.datasets.mm10_chrom_sizes
   mira.datasets.mm10_tss_data
   mira.datasets.hg38_chrom_sizes
   mira.datasets.hg38_tss_data

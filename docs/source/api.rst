API
===

------------

.. module:: mira
.. currentmodule:: mira

Topic Modeling
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: topicmodeling

   mira.topics.make_model


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
   :template: tuner.rst

   mira.topics.BayesianTuner

.. autosummary::
   :toctree: topicmodeling
   
   mira.topics.gradient_tune

.. autosummary::
   :toctree: topicmodeling
   
   mira.topics.Redis
   

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
   mira.pl.plot_topic_contributions
   mira.pl.plot_disentanglement

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

Joint Representation
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: tools

   mira.utils.make_joint_representation
   mira.tl.get_cell_pointwise_mutual_information
   mira.tl.summarize_mutual_information
   mira.tl.get_relative_norms
   mira.tl.get_topic_cross_correlation

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
   mira.datasets.StreamGraphTutorial
   mira.datasets.PseudotimeTrajectoryInferenceTutorial
   mira.datasets.ShareseqTopicModels
   mira.datasets.ShareseqBaseData
   mira.datasets.ShareseqAnnotatedData
   mira.datasets.ShareseqRPModels
   mira.datasets.MouseBrainDataset
   mira.datasets.FrankenCell_RNA
   mira.datasets.mm10_chrom_sizes
   mira.datasets.mm10_tss_data
   mira.datasets.hg38_chrom_sizes
   mira.datasets.hg38_tss_data

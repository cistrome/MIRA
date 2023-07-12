
FAQ
===

------------

I have data in separate batches. Does MIRA perform batch correction?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Yes** - using the novel CODAL (COvariate Disentangling Augmented Loss) model.
MIRA's topic models can disentangle biological from technical effects across batches
in both scRNA-seq and scATAC-seq data while faithfully representing batch-confounded
cell types. This makes MIRA particularly useful for analyzing batched perturbations,
where the cell-state distribution of different batches may be different due to treatment.


I have only one modality. Can I still use MIRA?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Certain features of MIRA can be used without multiomic data. MIRA's topic
models do not depend on the other mode for training or inference, so one
could train a MIRA expression topic model for a single-cell RNA-seq experiment,
for instance. Also, the pseudotime trajectory inference module only needs 
a k-nearest neighbors graph over cells, and is agnostic about what data 
the KNN graph is derived from.

I have separate scATAC-seq and scRNA-seq data from the same sample. Can I use MIRA?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the cells measured by scATAC-seq and scRNA-seq are not joined by a barcode,
there are limitations to what MIRA can do. MIRA's RP models, NITE analysis, 
and joint representation **absolutely** depend on the shared cell barcode because
these methods are designed to help researchers find interesting differences
between accessibility and expression modalities. Integration techniques for 
matching cells between scATAC-seq and scRNA-seq experiments assume 
that chromatin accessibility and gene expression are correlated. 
This assumption is directly at odds with MIRA's analysis paradigm. 

See the question above for which of MIRA's models work for non-multiomic data.

I am working with an different organism (not human or mouse). Can I use MIRA?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MIRA has ready-to-use annotations for hg38 and mm10, but MIRA is flexible
on the organism. The main annotations that one must provide are:

* a fasta file of the genome
* gene TSS locations
* chromosome sizes

If you can get these annotations for your organism, you can use MIRA. The one
fixed aspect of the analysis for now is the motif database. MIRA uses the
JASPAR 2020 vertabrates collection. In the future, we will allow users to
download other databases or provide their own position weight matrices.

Does MIRA work with spatial or protein data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No, MIRA's models are built to compare chromatin accessibility and gene expression.
Additional modalties pose interesting questions, but we do not address them. 


I want to make streamgraphs but I don't have multiome data. Can I do this?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! See the first question on this page. Additionally, MIRA's stream graph 
can be used with upstream processing by `CellRank` or MIRA's own pseudotime 
trajectory inference API. In turn, these workflows only need some form of 
nearest-neighbors graph over cells. 

If you have an ongoing project and you don't want to start from scatch
with topic modeling (though we think you'll find topic models elucidate
quite a lot about your data), you can install MIRA and immediately
proceed to making streams.

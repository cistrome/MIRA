<p align="center">
  <img src="https://github.com/AllenWLynch/Kladi/blob/adata/docs/graphics/mira_logo.png" width="250" />
</p>

**MIRA (Probabilistic Multimodal <ins>M</ins>odels for <ins>I</ins>ntegrated <ins>R</ins>egulatory <ins>A</ins>nalysis)** is a comprehensive methodology that systematically contrasts single cell transcription and accessibility to infer the regulatory circuitry driving cells along developmental trajectories. 

MIRA leverages joint topic modeling of cell states and regulatory potential modeling at individual gene loci to:
- jointly represent cell states in an efficient and interpretable latent space
- infer high fidelity lineage trees
- determine key regulators of fate decisions at branch points
- expose the variable influence of local accessibility on transcription at distinct loci

See [our manuscript](https://www.biorxiv.org/content/10.1101/2021.12.06.471401v1) for details.

## Getting Started

MIRA takes count matrices of transcripts and accessible regions measured by single cell multimodal RNA-seq and ATAC-seq from any platform as input data. MIRA output integrates with AnnData data structure for interoperability with Scanpy. The initial model training is faster with GPU hardware but can be accomplished with CPU computation.

Please refer to [our tutorial](https://colab.research.google.com/drive/1dtBMWNlkf58yGKylsJUFMxtNwHAo0h04?usp=sharing) for an overview of analyses that can be achieved with MIRA using an example 10x Multiome embryonic brain dataset.


## System Requirements

* Linux or MacOS
* Python >=3.5, <3.8
* (optional) CUDA-enable GPU


## Install

MIRA can be installed from either [PyPI](https://pypi.org/project/mira-multiome) or [conda-forge](https://anaconda.org/liulab-dfci/mira-multiome):

<pre>
pip install mira-multiome
</pre>
or
<pre>
conda install -c conda-forge -c liulab-dfci -c bioconda mira-multiome
</pre>

Installation will take about a minute.

Please note, currently MIRA may only be installed on Python < 3.8 due to some dependencies' requirements. We are working to make it accessible on newer Python versions. To set up an a new analysis, it is recommended to start with a fresh environment:

<pre>
conda create --name mira-env -c conda-forge -c liulab-dfci -c bioconda mira-multiome scanpy jupyter leidenalg
conda activate mira-env
python -m ipykernel install --user --name mira-env
</pre>

To use the environment in a jupyter notebook, start the notebook server, then go to Kernel > Change kernel > mira-env.


## Dependencies

* pytorch
* pyro-ppl
* tqdm
* moods
* pyfaidx
* matplotlib
* lisa2
* requests
* networkx
* numpy
* scipy
* optuna
* anndata

See [environment.yaml](https://github.com/cistrome/MIRA/blob/main/environment.yaml) for versioning details.

## Changelog

See [changelog](https://github.com/cistrome/MIRA/blob/main/docs/changelog.md) for details.


## Gallery

**With MIRA, you can analyze single cell multimodal transcriptional (RNA-seq) and accessibility (ATAC-seq) to:**

Construct biologically meaningful joint representations of cells progressing through developmental trajectories<sup>1</sup>:

<p align="center">
  <img src="https://github.com/AllenWLynch/Kladi/blob/adata/docs/graphics/mira_joint_rep.png"/>
</p>

<p>&nbsp;</p>

Infer high fidelity lineage trees defining developmental fate decisions<sup>1</sup>:

<p align="center">
  <img src="https://github.com/AllenWLynch/Kladi/blob/adata/docs/graphics/mira_lineage_tree.png"/ width=650>
</p>

<p>&nbsp;</p>

Learn the "topics" describing cell transcriptional and accessibility states<sup>1</sup>:

<p align="center">
  <img src="https://github.com/AllenWLynch/Kladi/blob/adata/docs/graphics/mira_topics.png"/ width=750>
</p>

<p>&nbsp;</p>

Contrast transcriptional and accessibility topics on stream graphs and determine the pathways and regulators governing in each cell state<sup>1</sup>:

<p align="center">
  <img src="https://github.com/AllenWLynch/Kladi/blob/adata/docs/graphics/mira_streams.png"/>
</p>

<p>&nbsp;</p>

Identify the transcription factors driving poised genes down diverging developmental paths, predict transcription factor targets via in silico deletion of putative regulatory elements, plot heatmaps of transcriptional and accessibility dynamics, and compare expression and motif scores of key factors on MIRA's joint representation<sup>1</sup>:

<p align="center">
  <img src="https://github.com/AllenWLynch/Kladi/blob/adata/docs/graphics/mira_tf_drivers.png"/>
</p>

<p>&nbsp;</p>

Explore gene expression within lineage trajectories and compare expression to motif score of key factors with stream graphs<sup>1</sup>:

<p align="center">
  <img src="https://github.com/AllenWLynch/Kladi/blob/adata/docs/graphics/mira_stream_variations.png"/>
</p>

<p>&nbsp;</p>

Determine the transcription factors driving fate decisions at key lineage branch points<sup>2</sup>:

<p align="center">
  <img src="https://github.com/AllenWLynch/Kladi/blob/adata/docs/graphics/mira_fate_drivers.png"/>
</p>

<p>&nbsp;</p>

Elucidate genes with local chromatin accessibility-influenced transcriptional expression (LITE) versus non-local chromatin accessibility-influenced transcriptional expression (NITE) and plot "chromatin differential" to highlight cells where transcription is decoupled from shifts in local chromatin accessibility<sup>2</sup>:

<p align="center">
  <img src="https://github.com/AllenWLynch/Kladi/blob/adata/docs/graphics/mira_chrom_diff.png"/ width=500>
</p>

<p>&nbsp;</p>

Quantify NITE regulation of topics or cells across the developmental continuum to reveal how variable circuitry regulates fate commitment and terminal identity.<sup>1,2</sup>:

<p align="center">
  <img src="https://github.com/AllenWLynch/Kladi/blob/adata/docs/graphics/mira_nite_stream.png"/>
</p>

<p>&nbsp;</p>

Overall, MIRA leverages principled probabilistic cell-level topic modeling and gene-level RP modeling to expose the key regulators driving fate decisions at lineage branch points and to precisely contrast the spatiotemporal dynamics of transcription and local chromatin accessibility at unprecedented resolution to reveal the distinct circuitry regulating fate commitment versus terminal identity.  

<p>&nbsp;</p>

## Methodology

<p align="center">
  <img src="https://github.com/AllenWLynch/Kladi/blob/adata/docs/graphics/mira_schematic.png"/>
</p>

### MIRA Topic Model
MIRA harnesses a variational autoencoder approach to model both transcription and chromatin accessibility topics defining each cell’s identity while accounting for their distinct statistical properties and employing a sparsity constraint to ensure topics are coherent and interpretable. MIRA’s hyperparameter tuning scheme learns the appropriate number of topics needed to comprehensively yet non-redundantly describe each dataset. MIRA next combines the expression and accessibility topics into a joint representation used to calculate a k-nearest neighbors (KNN) graph. This output can then be leveraged for visualization and clustering, construction of high fidelity lineage trajectories, and rigorous topic analysis to determine regulators driving key fate decisions at lineage branch points. 

### MIRA RP Model
MIRA’s regulatory potential (RP) model integrates transcriptional and chromatin accessibility data at each gene locus to determine how regulatory elements surrounding each gene influence its expression. Regulatory influence of enhancers is modeled to decay exponentially with genomic distance at a rate learned by the MIRA RP model from the joint multimodal data. MIRA learns independent upstream and downstream decay rates and includes parameters to weigh upstream, downstream, and promoter effects. The RP of each gene is scored as the sum of the contribution of individual regulatory elements. MIRA predicts key regulators at each locus by examining transcription factor motif enrichment or occupancy (if provided chromatin immunoprecipitation (ChIP-seq) data) within elements predicted to highly influence transcription at that locus using probabilistic in silico deletion (ISD).

### MIRA LITE vs NITE Models
MIRA quantifies the regulatory influence of local chromatin accessibility by comparing the local RP model with a second, expanded model that augments the local RP model with genome-wide accessibility states encoded by MIRA’s chromatin accessibility topics. Genes whose expression is significantly better described by this expanded model are defined as non-local chromatin accessibility-influenced transcriptional expression (NITE) genes. Genes whose transcription is sufficiently predicted by the RP model based on local accessibility alone are defined as local chromatin accessibility-influenced transcriptional expression (LITE) genes. While LITE genes appear tightly regulated by local chromatin accessibility, the transcription of NITE genes appears to be titrated without requiring extensive local chromatin remodeling. MIRA defines the extent to which the LITE model over- or under-estimates expression in each cell as “chromatin differential”, highlighting cells where transcription is decoupled from shifts in local chromatin accessibility. MIRA examines chromatin differential across the developmental continuum to reveal how variable circuitry regulates fate commitment and terminal identity.

## Citations

MIRA was created by researchers in the X. Shirley Liu Lab at Dana-Farber Cancer Institute. If you use MIRA in your research, we would appreciate citation of [our manuscript](https://www.biorxiv.org/content/10.1101/2021.12.06.471401v1) ([bibtex](https://github.com/AllenWLynch/Kladi/blob/adata/docs/references/mira_bioarxiv.bib)).

<p>&nbsp;</p>

Public datasets used for analyses in gallery and tutorial:

1. Ma, S. et al. Chromatin Potential Identified by Shared Single-Cell Profiling of RNA and Chromatin. _Cell_ (2020).
2. Datasets - 10x Genomics. https://support.10xgenomics.com/single-cell-multiome-atac-gex/datasets.

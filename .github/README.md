MIRA-dev
--------

A repository for developmental updates and changes to the MIRA multiome analysis package.
Currently, this repo contains code for batch correction of multiome data,
which will be included with the main MIRA package at a later date.

The APIs are the same as the currently-released MIRA version, except for the topic model
tuning API. A tutorial on this will be added soon. Please see [the MIRA documentation](https://mira-multiome.readthedocs.io/en/latest/).

For now, one must use

```
model = mira.topics.load_model('path/to/topicmodel.pth')
```

To load a model from disk instead of the old method. Every other method regarding the model is unchanged. 

Installation
------------

**Requirements and data**

*System*
* Python >= 3.7
* Conda

*Data*

QC-filtered count matrix of raw expression or accessibility feature counts per cell.

**Setup**

*Conda env*

Please create a new conda environment in which to install MIRA and other packages useful for analysis, then activate that environment:

```
$ conda create --name mira-analysis -c conda-forge -c bioconda -c liulab-dfci scanpy mira-multiome
$ conda activate mira-analysis
```

*Installing*

Install via pip from github:

```
(mira-analysis) $ pip install git+https://github.com/AllenWLynch/MIRA-dev.git
```

This can also be used to update the package.

More Info
---------

Please see the [main MIRA repository](https://github.com/cistrome/MIRA), and see the [paper](https://www.biorxiv.org/content/10.1101/2021.12.06.471401v1.full.pdf).
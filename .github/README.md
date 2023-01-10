CODAL
-----

A repository for developmental updates and changes to the MIRA multiome analysis package.
Currently, this repo contains code for batch correction of multiome data,
which will be included with the main MIRA package at a later date.

The APIs are the same as the currently-released MIRA version, except for the topic model
tuning API.

Tutorial
--------

[Click here](/docs/source/notebooks/CODAL_tutorial.ipynb) to view or download the CODAL tutorial. The tutorial demonstrates CODAL on a simulated
perturbation of a differentiation system, and should take ~10 minutes to run on a laptop.

Code and notebooks to reproduce figures may be found [here](https://github.com/AllenWLynch/CODA-reproduction/tree/main).


Installation
------------

Installation should take about a minute on most machines. First, set up a Conda environment with the 
requisite dependencies, then install the CODAL repository using pip.

**System Requirements**

* Linux or Mac OS
* Python >= 3.7
* Conda


**Conda Env**

Please create a new conda environment in which to install MIRA and other packages useful for analysis, then activate that environment:

```
$ conda create --name codal -c conda-forge -c bioconda -c liulab-dfci scanpy mira-multiome jupyterlab ipywidgets
$ conda activate codal
```

Then, configure the jupyterlab kernel:

```
(codal) $ python -m ipykernel install --user --name=codal
(codal) $ jupyter nbextension enable --py widgetsnbextension
```

**Installation**

Install via pip from github:

```
(codal) $ pip install git+https://github.com/AllenWLynch/MIRA-dev.git
```

This can also be used to update the package.


Dependencies
------------

* torch>=1.8.0,<2
* pyro-ppl>=1.5.2,<2
* networkx>=2.3,<3
* optuna>=2.8,<3
* anndata>=0.7.6,<1
* MOODS-python>=1.9.4.1
* pyfaidx>=0.5,<1
* matplotlib>=3.4,<4
* lisa2>=2.3.0
* requests>=2,<3
* tqdm
* tensorboard
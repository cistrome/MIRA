from setuptools import setup
from setuptools import find_packages

long_text='''
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

Installation will usually take about a minute.

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
'''


setup(name='mira-multiome',
      version='0.0.7',
      description='Single-cell multiomics data analysis package',
      long_description=long_text,
      long_description_content_type="text/markdown",
      url='https://github.com/cistrome/MIRA',
      author='Allen W Lynch',
      author_email='alynch@ds.dfci.harvard.edu',
      license='MIT',
      classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
      ],
      packages=find_packages(),
      install_requires = [
        'torch>=1.8.0,<2',
        'tqdm',
        'MOODS-python>=1.9.4.1',
        'pyfaidx>=0.5,<1',
        'matplotlib>=3.4,<4',
        'lisa2>=2.2.5,<2.3',
        'requests>=2,<3',
        'pyro-ppl>=1.5.2,<2',
        'networkx>=2.3,<3',
        'numpy>=1.19.0,<2',
        'scipy>=1.5,<2',
        'optuna>=2.8,<3',
        'anndata>=0.7.6,<1',
      ],
      include_package_data = True,
      zip_safe=True)
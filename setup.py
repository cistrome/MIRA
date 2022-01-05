from setuptools import setup
from setuptools import find_packages
import configparser
import os

configpath = os.path.join(os.path.dirname(__file__), 'mira', 'config.ini')

print(configpath)
config = configparser.ConfigParser()
config.read(configpath)

#print(info, info.sections)
info = config['MIRA']

long_text='''
**MIRA (Probabilistic Multimodal Models for Integrated Regulatory Analysis)** is a comprehensive methodology that systematically contrasts single cell transcription and accessibility to infer the regulatory circuitry driving cells along developmental trajectories. 

MIRA leverages joint topic modeling of cell states and regulatory potential modeling at individual gene loci to:
- jointly represent cell states in an efficient and interpretable latent space
- infer high fidelity lineage trees
- determine key regulators of fate decisions at branch points
- expose the variable influence of local accessibility on transcription at distinct loci

See [our manuscript]({bioarxiv}) for details. View code at the [MIRA github repository]({github}).

## Getting Started

MIRA takes count matrices of transcripts and accessible regions measured by single cell multimodal RNA-seq and ATAC-seq from any platform as input data. MIRA output integrates with AnnData data structure for interoperability with Scanpy. The initial model training is faster with GPU hardware but can be accomplished with CPU computation.

Please refer to [our tutorial]({tutorial}) for an overview of analyses that can be achieved with MIRA using an example 10x Multiome embryonic brain dataset.


## System Requirements

* Linux or MacOS
* Python >=3.5, <3.8
* (optional) CUDA-enabled GPU

## Install

MIRA can be installed from either [PyPI]({pypi_site}) or [conda-forge]({conda_site}):

<pre>
{pypi_install}
</pre>
or
<pre>
{conda_install}
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

See [environment.yaml]({github}/blob/main/environment.yaml) for versioning details.
'''.format(**info)


setup(name='mira-multiome',
      version=info['version'],
      description='Single-cell multiomics data analysis package',
      long_description=long_text,
      long_description_content_type="text/markdown",
      url = info['github'],
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
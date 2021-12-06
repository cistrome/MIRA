from setuptools import setup
from setuptools import find_packages

long_text='''
**MIRA (Probabilistic Multimodal Models for Integrated Regulatory Analysis)** is a comprehensive methodology that systematically contrasts single cell transcription and accessibility to infer the regulatory circuitry driving cells along developmental trajectories. 

MIRA leverages joint topic modeling of cell states and regulatory potential modeling at individual gene loci to:
- jointly represent cell states in an efficient and interpretable latent space
- infer high fidelity lineage trees
- determine key regulators of fate decisions at branch points
- expose the variable influence of local accessibility on transcription at distinct loci

## Install

MIRA can be installed from either [PyPI](https://pypi.org/project/mira-multiome) or [conda](https://anaconda.org/liulab-dfci/mira-multiome):

<pre>
pip install mira-multiome
</pre>
or
<pre>
conda install -c conda-forge -c liulab-dfci -c bioconda mira-multiome
</pre>

## Getting Started

MIRA takes count matrices of transcripts and accessible regions measured by single cell multimodal RNA-seq and ATAC-seq from any platform as input data. MIRA output integrates with AnnData data structure for interoperability with Scanpy. The initial model training is faster with GPU hardware but can be accomplished with CPU computation.

Please refer to [our tutorial](https://colab.research.google.com/drive/1dtBMWNlkf58yGKylsJUFMxtNwHAo0h04?usp=sharing) for an overview of analyses that can be achieved with MIRA using an example 10x Multiome embryonic brain dataset.
'''


setup(name='mira-multiome',
      version='0.0.6',
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
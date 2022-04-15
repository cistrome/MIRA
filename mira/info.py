
### MIRA attributes and configuration###
#======================================#
version = '0.1.1'

github = 'https://github.com/cistrome/MIRA'

bioarxiv = 'https://www.biorxiv.org/content/10.1101/2021.12.06.471401v1'

tutorial = 'https://colab.research.google.com/drive/1m3PUc_X4IaddPyHfGhwakNXFPtKEEbgp?usp=sharing'

pypi_site = 'https://pypi.org/project/mira-multiome'

conda_site = 'https://anaconda.org/liulab-dfci/mira-multiome'

conda_install = 'conda install -c conda-forge -c liulab-dfci -c bioconda mira-multiome'

pypi_install = 'pip install mira-multiome'

dependencies = '''
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
'''

requirements = '''
* Linux or MacOS
* Python >=3.5, <3.8
* (optional) CUDA-enable GPU
'''
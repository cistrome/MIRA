from setuptools import setup
from setuptools import find_packages

setup(name='mira-multiome',
      version='0.0.0',
      description='single-cell multiomics data analysis package',
      url='https://github.com/cistrome/MIRA',
      author='Allen W Lynch',
      author_email='alynch@ds.dfci.harvard.edu',
      license='MIT',
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
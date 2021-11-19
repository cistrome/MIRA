# Kladi

To install Kladi's dependencies, I've included an environment.yaml with the repository. If you haven't already, install miniconda on your system:

```
$ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
```

Then, create a directory to work in and move to that directory:

```
$ mkdir ~/kladi && cd ~/kladi
```

Clone the repository:

```
$ git clone <url>
```

Then install dependencies using:

```
$ conda env create -f environment.yaml
```

Once, created, use:

```
$ conda activate kladi
```

To enter the environment. For analysis, I recommend adding jupyter and scanpy. If you will be working with a GPU, also install the requesite CUDA toolkit:

```
(kladi) $ conda install jupyter scanpy
(kladi) $ conda install cudatoolkit==<version>
```

Lastly, install a kernel for your environment:

```
(kladi) $ python -m ipykernel install --user --name kladi
```

And start the notebook server:

```
(kladi) $ jupyter notebook
```
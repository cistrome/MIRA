
import anndata
import pandas as pd
import os
from scipy.io import mmread
import numpy as np

def read_raw(prefix, suffix = ''):

    def read_obs(prefix):
        return pd.read_csv(prefix + 'barcodes.tsv' + suffix, 
                    sep = '\t', names = ['barcodes']).set_index('barcodes')

    def read_var(prefix):
        var = pd.read_csv(prefix + 'features.tsv' + suffix,
            sep = '\t', names = ['id', 'symbol', 'feature_type'])

        var['symbol'] = var.symbol.str.upper()

        return var

    def read_matrix(prefix):
        return mmread(prefix + 'matrix.mtx' + suffix).T.tocsr()

    return anndata.AnnData(
        obs = read_obs(prefix),
        var = read_var(prefix),
        X = read_matrix(prefix)
    )


def combine_adatas(*adatas, suffix = ''):

    adata_loaded = {}
    for batch, sample, path in adatas:

        adata = read_raw(path, suffix = suffix)
        adata.var = adata.var.set_index('id')
        adata.obs['batch'] = batch
        adata.obs['sample'] = sample
        adata_loaded['@' + batch + ':' + sample] = adata

    if len(adata_loaded) == 1:
        return list(adata_loaded.values())[0]
    
    return anndata.concat(
        adata_loaded, 
        merge = 'same',
        index_unique=':',
        label = '@batch:sample'
    )


def add_arguments(parser):

    parser.add_argument('--adata','-ad', action = 'append', type = str, nargs = 3,
        help = 'Add an Adata to format as <batch> <sample> <path/to/raw>')
    parser.add_argument('--outfile','-o', type = str, required=True)


def main(args):

    assert(len(args.adata) > 0)

    adata = combine_adatas(*args.adata)
    adata.write_h5ad(args.outfile)

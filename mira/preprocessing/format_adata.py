
import anndata
import pandas as pd
import os
from scipy.io import mmread
import numpy as np

def read_star_rawmatrix(dir_path):

    def read_obs(dir_path):
        return pd.read_csv(os.path.join(dir_path, 'barcodes.tsv'), 
                    sep = '\t', names = ['barcodes']).set_index('barcodes')

    def read_var(dir_path):
        var = pd.read_csv(os.path.join(dir_path, 'features.tsv'),
            sep = '\t', names = ['id', 'symbol', 'feature_type'])

        var['symbol'] = var.symbol.str.upper()

        return var

    def read_matrix(dir_path):
        return mmread(os.path.join(dir_path, 'matrix.mtx')).T.tocsr()

    return anndata.AnnData(
        obs = read_obs(dir_path),
        var = read_var(dir_path),
        X = read_matrix(dir_path)
    )


def combine_adatas(*adatas):

    adata_list = []
    for batch, sample, path in adatas:

        adata = read_star_rawmatrix(path)
        adata.obs_names = np.array(['@{}:{}:{}'.format(batch, sample, bc)
                for bc in adata.obs_names
            ])
        
        adata.var = adata.var.set_index('id')
            
        adata_list.append( adata )

    if len(adata_list) == 1:
        return adata_list[0]
    
    return anndata.concat(adata_list, merge = 'same')


def add_arguments(parser):

    parser.add_argument('--adata','-ad', action = 'append', type = str, nargs = 3,
        help = 'Add an Adata to format as <batch> <sample> <path/to/raw>')
    parser.add_argument('--outfile','-o', type = str, required=True)


def main(args):

    assert(len(args.adata) > 0)

    adata = combine_adatas(*args.adata)
    adata.write_h5ad(args.outfile)

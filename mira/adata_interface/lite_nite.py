import numpy as np
from mira.adata_interface.core import fetch_layer, get_dense_columns, project_matrix
import logging

logger = logging.getLogger(__name__)

def fetch_logp_data(self, adata, counts_layer = None):

    try:
        lite_gene_mask = get_dense_columns(self, adata, 'LITE_logp')
        lite_logp = fetch_layer(self, adata, 'LITE_logp')
    except KeyError:
        raise KeyError('User must run "get_logp" using a trained lite_model object before running this function')

    try:
        nite_gene_mask = get_dense_columns(self, adata, 'NITE_logp')
        nite_logp = fetch_layer(self, adata, 'NITE_logp')
    except KeyError:
        raise KeyError('User must run "get_logp" using a trained nite_model before running this function')

    overlapped_genes = np.logical_and(lite_gene_mask, nite_gene_mask)
    expression = fetch_layer(self, adata, counts_layer)
    
    return dict(
        lite_logp = lite_logp[:, overlapped_genes].toarray(),
        nite_logp = nite_logp[:, overlapped_genes].toarray(),
        gene_expr = expression[:, overlapped_genes].toarray(),
        genes = adata.var_names[overlapped_genes].values,
    )
    

def add_NITE_score_gene(adata, output):

    genes, nite_score, nonzero_counts = output
    
    adata.var['NITE_score'] = \
        project_matrix(adata.var_names.values, genes, nite_score[np.newaxis,:]).reshape(-1)

    adata.var['nonzero_counts'] = \
        project_matrix(adata.var_names.values, genes, nonzero_counts[np.newaxis,:]).reshape(-1)

    logger.info('Added keys to var: NITE_score, nonzero_counts')


def add_NITE_score_cell(adata, output):

    nite_score, nonzero_counts = output
    
    adata.obs['NITE_score'] = nite_score
    adata.obs['nonzero_counts'] = nonzero_counts

    logger.info('Added keys to obs: NITE_score, nonzero_counts')


def fetch_NITE_score_gene(self, adata):

    try:
        nite_score = adata.var_vector('NITE_score')
    except KeyError:
        raise KeyError(
            'User must run "global_local_test" function to calculate test_statistic before running this function'
        )

    genes = adata.var_names
    mask = np.isfinite(nite_score)

    return dict(
        genes = genes[mask],
        nite_score = nite_score[mask],
    )


def fetch_lite_nite_prediction(self, adata):

    try:
        lite_gene_mask = get_dense_columns(self, adata, 'LITE_prediction')
        lite_prediction = fetch_layer(self, adata, 'LITE_prediction')
    except KeyError:
        raise KeyError('User must run "predict" using a trained lite_model object before running this function')

    try:
        nite_gene_mask = get_dense_columns(self, adata, 'NITE_prediction')
        nite_prediction = fetch_layer(self, adata, 'NITE_prediction')
    except KeyError:
        raise KeyError('User must run "predict" using a trained nite_model before running this function')

    overlapped_genes = np.logical_and(lite_gene_mask, nite_gene_mask)

    return dict(
        nite_prediction = nite_prediction[:, overlapped_genes].toarray(),
        lite_prediction = lite_prediction[:, overlapped_genes].toarray(),
        genes = adata.var_names[overlapped_genes]
    )
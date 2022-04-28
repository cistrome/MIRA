import logging
import inspect
from functools import partial, wraps
import numpy as np
from mira.adata_interface.core import project_matrix, add_layer
from mira.adata_interface.regulators import fetch_peaks, fetch_factor_hits
from tqdm.auto import tqdm
from scipy import sparse
from joblib import Parallel, delayed
import pandas as pd
logger = logging.getLogger(__name__)

def add_predictions(adata, output, model_type = 'LITE', sparse = True):

    features, predictions = output
    expr_predictions, logp_data = list(zip(*predictions))
    
    #(adata, output, add_layer = 'imputed', sparse = False):
    add_layer(adata, (features, np.hstack(expr_predictions)), 
        add_layer = model_type + '_prediction', sparse = True)

    add_layer(adata, (features, np.hstack(logp_data)), 
        add_layer = model_type + '_logp', sparse = True)


def get_peak_and_tss_data(self, adata, tss_data = None, peak_chrom = 'chr', peak_start = 'start', peak_end = 'end', 
        gene_id = 'geneSymbol', gene_chrom = 'chrom', gene_start = 'txStart', gene_end = 'txEnd', gene_strand = 'strand',
        sep = '\t'):

    if tss_data is None:
        raise Exception('User must provide dataframe of tss data to "tss_data" parameter.')

    if isinstance(tss_data, str):
        tss_data = pd.read_csv(tss_data, sep = sep)
        tss_data.columns = tss_data.columns.str.strip('#')

    return_dict = fetch_peaks(self, adata, chrom = peak_chrom, start = peak_start, end = peak_end)
    
    try:
        return_dict.update(
            {
                'gene_id' : tss_data[gene_id].values,
                'chrom' : tss_data[gene_chrom].values,
                'start' : tss_data[gene_start].values,
                'end' : tss_data[gene_end].values,
                'strand' : tss_data[gene_strand].values
            }
        )
    except KeyError as err:
        raise KeyError('Missing column in TSS annotation. Please make sure you indicate the correct names for columns through the keyword arguments.')


    return return_dict


def add_peak_gene_distances(adata, output):

    distances, gene, chrom, start, end, strand = output

    adata.varm['distance_to_TSS'] = distances.tocsc()
    adata.uns['distance_to_TSS_genes'] = list(gene)

    adata.uns['TSS_metadata'] = {
        'gene' : list(gene), 'chromosome' : list(chrom), 'txStart' : list(start),
        'txEnd' : list(end), 'strand' : list(strand),
    }

    logger.info('Added key to var: distance_to_TSS')
    logger.info('Added key to uns: distance_to_TSS_genes')


def fetch_TSS_data(self, adata):

    try:
        tss_metadata = adata.uns['TSS_metadata']
    except KeyError:
        raise KeyError('Adata does not have .uns["TSS_metadata"], user must run mira.tl.get_distance_to_TSS first.')

    return {
        gene : {
            'gene_chrom' : chrom, 'gene_start' : start, 'gene_end' : end, 'gene_strand' : strand
        }
        for gene, chrom, start, end, strand in list(zip(
            tss_metadata['gene'], tss_metadata['chromosome'], 
            tss_metadata['txStart'], tss_metadata['txEnd'], tss_metadata['strand']
        ))
    }


def fetch_TSS_from_adata(self, adata):
    
    tss_metadata = fetch_TSS_data(self, adata)

    try:
        return tss_metadata[self.gene]
    except KeyError:
        raise KeyError('Gene {} not in TSS annotation.'.format(self.gene))


def set_up_model(gene_name, atac_adata, expr_adata, 
    distance_matrix, read_depth, expr_softmax_denom,
    NITE_features, atac_softmax_denom, include_factor_data,
    self):

    try:
        gene_idx = np.argwhere(np.array(atac_adata.uns['distance_to_TSS_genes']) == gene_name)[0,0]
    except IndexError:
        raise IndexError('Gene {} does not appear in peak annotation'.format(gene_name))

    try:
        gene_expr = expr_adata.obs_vector(gene_name, layer = self.counts_layer)

        assert(np.isclose(gene_expr.astype(np.int64), gene_expr, 1e-2).all()), 'Input data must be raw transcript counts, represented as integers. Provided data contains non-integer values.'
        gene_expr = gene_expr.astype(int)
        
    except KeyError:
        raise KeyError('Gene {} is not found in expression data var_names'.format(gene_name))

    peak_idx = distance_matrix[gene_idx, :].indices
    tss_distance = distance_matrix[gene_idx, :].data

    model_features = {}
    for region_name, mask in zip(['promoter','upstream','downstream'], self._get_masks(tss_distance)):
        model_features[region_name + '_idx'] = peak_idx[mask]
        model_features[region_name + '_distances'] = np.abs(tss_distance[mask])

    model_features.pop('promoter_distances')

    return self._get_features_for_model(
                        gene_expr = gene_expr,
                        read_depth = read_depth,
                        expr_softmax_denom = expr_softmax_denom,
                        NITE_features = NITE_features,
                        atac_softmax_denom = atac_softmax_denom,
                        include_factor_data = include_factor_data,
                        **model_features,
                        )


def wraps_rp_func(adata_adder = lambda self, expr_adata, atac_adata, output, **kwargs : None, 
    bar_desc = '', include_factor_data = False):

    def wrap_fn(func):

        def rp_signature(*, expr_adata, atac_adata, n_workers = 1, atac_topic_comps_key = 'X_topic_compositions'):
            pass

        def isd_signature(*, expr_adata, atac_adata, n_workers = 1, checkpoint = None, 
            atac_topic_comps_key = 'X_topic_compositions', factor_type = 'motifs'):
            pass

        func_signature = inspect.signature(func).parameters.copy()
        func_signature.pop('model')
        func_signature.pop('features')
        
        if include_factor_data:
            mock = inspect.signature(isd_signature).parameters.copy()
        else:
            mock = inspect.signature(rp_signature).parameters.copy()

        func_signature.update(mock)
        func.__signature__ = inspect.Signature(list(func_signature.values()))
        
        @wraps(func)
        def get_RP_model_features(self,*, expr_adata, atac_adata, atac_topic_comps_key = 'X_topic_compositions', 
            factor_type = 'motifs', checkpoint = None, n_workers = 1, **kwargs):

            assert(isinstance(n_workers, int) and (n_workers >= 1 or n_workers == -1))
            assert len(expr_adata) == len(atac_adata), 'Must pass adatas with same number of cells to this function'
            assert np.all(expr_adata.obs_names == atac_adata.obs_names), 'To use RP models, cells must have same barcodes/obs_names'

            unannotated_genes = np.setdiff1d(self.genes, atac_adata.uns['distance_to_TSS_genes'])
            if len(unannotated_genes) > 0:
                raise ValueError('The following genes for RP modeling were not found in the TSS annotation: ' + ', '.join(unannotated_genes))

            if not 'model_read_scale' in expr_adata.obs.columns:
                self.expr_model._get_read_depth(expr_adata)

            read_depth = expr_adata.obs_vector('model_read_scale')

            if not 'softmax_denom' in expr_adata.obs.columns:
                self.expr_model._get_softmax_denom(expr_adata)

            expr_softmax_denom = expr_adata.obs_vector('softmax_denom')

            if not 'softmax_denom' in atac_adata.obs.columns:
                self.accessibility_model._get_softmax_denom(atac_adata)

            atac_softmax_denom = atac_adata.obs_vector('softmax_denom')

            if not atac_topic_comps_key in atac_adata.obsm:
                self.accessibility_model.predict(atac_adata, add_key = atac_topic_comps_key, add_cols = False)

            NITE_features = atac_adata.obsm[atac_topic_comps_key]

            if not 'distance_to_TSS' in atac_adata.varm:
                raise Exception('Peaks have not been annotated with TSS locations. Run "get_distance_to_TSS" before proceeding.')

            distance_matrix = atac_adata.varm['distance_to_TSS'].T #genes, #regions

            hits_data = dict()
            if include_factor_data:
                hits_data = fetch_factor_hits(self.accessibility_model, atac_adata, factor_type = factor_type,
                    binarize = True)

            if include_factor_data and not checkpoint is None:
                logger.warn('Resuming pISD from checkpoint. If wanting to recalcuate, use a new checkpoint file, or set checkpoint to None.')
                kwargs['checkpoint'] = checkpoint

            get_model_features_function = partial(set_up_model, atac_adata = atac_adata,
                expr_adata = expr_adata, distance_matrix = distance_matrix, read_depth = read_depth, 
                expr_softmax_denom = expr_softmax_denom, NITE_features = NITE_features, 
                atac_softmax_denom = atac_softmax_denom, include_factor_data = include_factor_data,
                self = self)

            if n_workers == 1:
                
                results = [
                    func(self, model, get_model_features_function(model.gene), **hits_data, **kwargs)
                    for model in tqdm(self.models, desc =bar_desc)
                ]

            else:
                
                def feature_producer():
                    for model in self.models:
                        yield model, get_model_features_function(model.gene)

                results = Parallel(n_jobs=n_workers, verbose=0, pre_dispatch='2*n_jobs', max_nbytes = None)\
                    (delayed(func)(self, model, features, **hits_data, **kwargs) 
                    for model, features in tqdm(feature_producer(), desc = bar_desc, total = len(self.models)))

            return adata_adder(self, expr_adata, atac_adata, results, factor_type = factor_type)

        return get_RP_model_features

    return wrap_fn


def add_isd_results(self, expr_adata, atac_adata, output, factor_type = 'motifs', **kwargs):

    #ko_logp, f_Z, expression, logp_data, informative_samples = list(zip(*output))
    ko_logp, informative_samples = list(zip(*output))

    factor_mask = atac_adata.uns[factor_type]['in_expr_data']
    
    ko_logp = np.vstack(ko_logp).T
    informative_samples = np.vstack(informative_samples).T

    ko_logp = project_matrix(expr_adata.var_names, self.genes, ko_logp)

    projected_ko_logp = np.full((len(factor_mask), ko_logp.shape[-1]), np.nan)
    projected_ko_logp[np.array(factor_mask), :] = ko_logp
    
    expr_adata.varm[factor_type + '-prob_deletion'] = projected_ko_logp.T
    logger.info('Appending to expression adata:')
    logger.info("Added key to varm: '{}-prob_deletion')".format(factor_type))

    informative_samples = project_matrix(expr_adata.var_names, self.genes, informative_samples)
    informative_samples = np.where(~np.isnan(informative_samples), informative_samples, 0)
    expr_adata.layers[factor_type + '-informative_samples'] = sparse.csr_matrix(informative_samples)
    logger.info('Added key to layers: {}-informative_samples'.format(factor_type))

    expr_adata.uns[factor_type] = atac_adata.uns[factor_type].copy()
    logger.info('Added key to uns: {}'.format(factor_type))

    #return f_Z, expression, logp_data



def fetch_get_influential_local_peaks(self, adata):

    try:
        gene_idx = np.argwhere(np.array(adata.uns['distance_to_TSS_genes']) == self.gene)[0,0]
    except IndexError:
        raise IndexError('Gene {} does not appear in peak annotation'.format(self.gene))

    if not 'distance_to_TSS' in adata.varm:
        raise Exception('Peaks have not been annotated with TSS locations. Run "get_distance_to_TSS" before proceeding.')

    distance_matrix = adata.varm['distance_to_TSS'].T #genes, #regions

    return {
        'peak_idx' : distance_matrix[gene_idx, :].indices,
        'tss_distance' : distance_matrix[gene_idx, :].data
    }


def return_peaks_by_idx(adata, output):

    idx, dist = output

    proximal_peaks = adata.var.iloc[idx]
    proximal_peaks['distance_to_TSS'] = np.abs(dist)
    proximal_peaks['is_upstream'] = dist <= 0

    return proximal_peaks
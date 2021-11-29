import logging
import inspect
from functools import wraps
import numpy as np
from scipy.sparse import isspmatrix
from mira.adata_interface.core import fetch_layer,project_matrix
from mira.adata_interface.regulators import fetch_peaks, fetch_factor_hits
import tqdm
from scipy import sparse
logger = logging.getLogger(__name__)

def get_peak_and_tss_data(self, adata, tss_data = None, peak_chrom = 'chr', peak_start = 'start', peak_end = 'end', 
        gene_id = 'geneSymbol', gene_chrom = 'chrom', gene_start = 'txStart', gene_end = 'txEnd', gene_strand = 'strand'):

    if tss_data is None:
        raise Exception('User must provide dataframe of tss data to "tss_data" parameter.')

    return_dict = fetch_peaks(self, adata, chrom = peak_chrom, start = peak_start, end = peak_end)

    return_dict.update(
        {
            'gene_id' : tss_data[gene_id].values,
            'chrom' : tss_data[gene_chrom].values,
            'start' : tss_data[gene_start].values,
            'end' : tss_data[gene_end].values,
            'strand' : tss_data[gene_strand].values
        }
    )

    return return_dict


def add_peak_gene_distances(adata, output):

    distances, genes = output

    adata.varm['distance_to_TSS'] = distances.tocsc()
    adata.uns['distance_to_TSS_genes'] = list(genes)

    logger.info('Added key to var: distance_to_TSS')
    logger.info('Added key to uns: distance_to_TSS_genes')


def wraps_rp_func(adata_adder = lambda self, expr_adata, atac_adata, output, **kwargs : None, 
    bar_desc = '', include_factor_data = False):

    def wrap_fn(func):

        def rp_signature(*, expr_adata, atac_adata, atac_topic_comps_key = 'X_topic_compositions'):
            pass

        def isd_signature(*, expr_adata, atac_adata, atac_topic_comps_key = 'X_topic_compositions', factor_type = 'motifs'):
            pass

        func_signature = inspect.signature(func).parameters.copy()
        func_signature.pop('model')
        func_signature.pop('features')
        
        if include_factor_data:
            rp_signature = isd_signature

        mock = inspect.signature(rp_signature).parameters.copy()

        func_signature.update(mock)
        func.__signature__ = inspect.Signature(list(func_signature.values()))
        
        @wraps(func)
        def get_RP_model_features(self,*, expr_adata, atac_adata, atac_topic_comps_key = 'X_topic_compositions', 
            factor_type = 'motifs', **kwargs):

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

            results = []
            for model in tqdm.tqdm(self.models, desc = bar_desc):

                gene_name = model.gene
                try:
                    gene_idx = np.argwhere(np.array(atac_adata.uns['distance_to_TSS_genes']) == gene_name)[0,0]
                except IndexError:
                    raise IndexError('Gene {} does not appear in peak annotation'.format(gene_name))

                try:
                    gene_expr = expr_adata.obs_vector(gene_name, layer = self.counts_layer).astype(int)
                except KeyError:
                    raise KeyError('Gene {} is not found in expression data var_names'.format(gene_name))

                peak_idx = distance_matrix[gene_idx, :].indices
                tss_distance = distance_matrix[gene_idx, :].data

                model_features = {}
                for region_name, mask in zip(['promoter','upstream','downstream'], self._get_masks(tss_distance)):
                    model_features[region_name + '_idx'] = peak_idx[mask]
                    model_features[region_name + '_distances'] = np.abs(tss_distance[mask])

                model_features.pop('promoter_distances')
                
                results.append(func(
                        self, model, 
                        self._get_features_for_model(
                            gene_expr = gene_expr,
                            read_depth = read_depth,
                            expr_softmax_denom = expr_softmax_denom,
                            NITE_features = NITE_features,
                            atac_softmax_denom = atac_softmax_denom,
                            include_factor_data = include_factor_data,
                            **model_features,
                            ),
                        **hits_data,
                        **kwargs)
                )

            return adata_adder(self, expr_adata, atac_adata, results, factor_type = factor_type)

        return get_RP_model_features

    return wrap_fn


def add_isd_results(self, expr_adata, atac_adata, output, factor_type = 'motifs', **kwargs):

    ko_logp, f_Z, expression, logp_data, informative_samples = list(zip(*output))

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
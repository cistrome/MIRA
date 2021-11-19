import numpy as np
from mira.adata_interface.topic_model import fetch_topic_comps

def get_peaks(self, adata, chrom = 'chr', start = 'start', end = 'end'):

    try:
        return dict(peaks = adata.var[[chrom, start, end]].values.tolist())
    except IndexError:
        raise Exception('Some of columns {}, {}, {} are not in .var'.format(chrom, start, end))


def add_factor_hits_data(self, adata, output,*, factor_type):

    factor_id, factor_name, parsed_name, hits = output

    adata.varm[factor_type + '_hits'] = hits.T.tocsc()
    meta_dict = {
        'id' : list(factor_id),
        'name' : list(factor_name),
        'parsed_name' : list(parsed_name),
        'in_expr_data' : [True]*len(factor_id),
    }
    adata.uns[factor_type] = meta_dict

    logger.info('Added key to varm: ' + factor_type + '_hits')
    logger.info('Added key to uns: ' + factor_type)
    #logger.info('Added key to uns: ' + ', '.join([factor_type + '_' + suffix for suffix in ['id','name','parsed_name','in_expr_data']]))


def add_factor_mask(self, adata, mask,*,factor_type):

    if not factor_type in adata.uns:
        raise KeyError('No metadata for factor type {}. User must run "find_motifs" to add motif data.'.format(factor_type))

    assert(isinstance(mask, (list, np.ndarray)))

    mask = list(mask)
    assert(all([type(x) == bool for x in mask]))
    assert(len(mask) == len(adata.uns[factor_type]['in_expr_data']))

    adata.uns[factor_type]['in_expr_data'] = mask


def get_factor_meta(self, adata, factor_type = 'motifs', mask_factors = True):

    fields = ['id','name','parsed_name']

    try:
        meta_dict = adata.uns[factor_type]
    except KeyError:
        raise KeyError('No metadata for factor type {}. User must run "find_motifs" to add motif data.'.format(factor_type))

    mask = np.array(meta_dict['in_expr_data'])
    col_len = len(mask)

    if not mask_factors:
        mask = np.ones_like(mask).astype(bool)

    metadata = [
        list(np.array(meta_dict[field])[mask])
        for field in fields
    ]

    metadata = list(zip(*metadata))
    metadata = [dict(zip(fields, v)) for v in metadata]

    return metadata, mask


def get_factor_hits(self, adata, factor_type = 'motifs', mask_factors = True, binarize = True):

    try:

        metadata, mask = get_factor_meta(None, adata, factor_type = factor_type, mask_factors = mask_factors)

        hits_matrix = adata[:, self.features].varm[factor_type + '_hits'].T.tocsr()
        hits_matrix = hits_matrix[mask, :]

        if binarize:
            hits_matrix.data = np.ones_like(hits_matrix.data)
    
    except KeyError:
        raise KeyError('User must run "find_motifs" or "find_ChIP_hits" to add binding data before running this function')

    return dict(
        hits_matrix = hits_matrix,
        metadata = metadata
    )


def get_factor_hits_and_latent_comps(self, adata, factor_type = 'motifs', mask_factors = True, key = 'X_topic_compositions'):

    return dict(
        **get_factor_hits(self, adata, factor_type = factor_type, mask_factors = mask_factors),
        **fetch_topic_comps(self, adata, key = key)
    )


def get_motif_score_adata(self, adata, output):

    metadata, scores, norm_scores = output

    X = anndata.AnnData(
        var = metadata,
        obs = adata.obs.copy(),
        X = norm_scores,
    )
    X.layers['raw_logp_binding'] = scores

    return X


def save_factor_enrichment(self, adata, output,*, module_num, factor_type):

    adata.uns[('enrichment', module_num, factor_type)] = output
    return output



def fetch_logp_data(self, adata, counts_layer = None):

    try:
        cis_logp = adata.layers['cis_logp']
    except KeyError:
        raise KeyError('User must run "get_logp" using a trained cis_model object before running this function')

    try:
        trans_logp = adata.layers['trans_logp']
    except KeyError:
        raise KeyError('User must run "get_logp" using a trained global_model (set use_trans_features = True on cis_model object) before running this function')

    overlapped_genes = np.logical_and(np.isfinite(cis_logp).all(0), np.isfinite(trans_logp).all(0))
    expression = fetch_layer(adata, counts_layer)

    return dict(
        cis_logp = cis_logp[:, overlapped_genes],
        trans_logp = trans_logp[:, overlapped_genes],
        gene_expr = expression[:, overlapped_genes],
        genes = adata.var_names[overlapped_genes].values,
    )


def project_row(adata_index, project_features, vals, width):

    orig_feature_idx = dict(zip(adata_index, np.arange(width)))

    original_to_imputed_map = np.array(
        [orig_feature_idx[feature] for feature in project_features]
    )

    new_row = np.full(width, np.nan)
    new_row[original_to_imputed_map] = vals
    return new_row
    

def add_global_test_statistic(self, adata, output):

    genes, test_stat, pval, nonzero_counts = output
    
    adata.var['global_regulation_test_statistic'] = \
        project_row(adata.var_names.values, genes, test_stat, adata.shape[-1])

    adata.var['global_regulation_pval'] = \
        project_row(adata.var_names.values, genes, pval, adata.shape[-1])

    adata.var['nonzero_counts'] = \
        project_row(adata.var_names.values, genes, nonzero_counts, adata.shape[-1])

    logger.info('Added keys to var: global_regulation_test_statistic, global_regulation_pval, nonzero_counts')


def fetch_global_test_statistic(self, adata):

    try:
        test_stat = adata.var['global_regulation_test_statistic']
    except KeyError:
        raise KeyError(
            'User must run "global_local_test" function to calculate test_statistic before running this function'
        )

    genes = adata.var_names
    mask = np.isfinite(test_stat)

    return dict(
        genes = genes[mask],
        test_statistic = test_stat[mask],
    )

def fetch_cis_trans_prediction(self, adata):

    try:
        cis = adata.layers['cis_prediction']
    except KeyError:
        raise KeyError('User must run "predict" with a cis_model object before running this function.')

    try:
        trans = adata.layers['trans_prediction']
    except KeyError:
        raise KeyError('User must run "predict" with a cis_model object with "use_trans_features" set to true before running this function.')

    return dict(
        cis_prediction = cis,
        trans_prediction = trans,
    )

def add_chromatin_differential(self, adata, output):
    adata.layers['chromatin_differential'] = output
    logger.info('Added key to layers: chromatin_differential')
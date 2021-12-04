import mira.adata_interface.core as adi
import mira.adata_interface.lite_nite as lni
from scipy.sparse import isspmatrix
import logging
import numpy as np

logger = logging.getLogger(__name__)

def fetch_differential_plot(self, adata, genes=None, counts_layer = None):

    assert(not genes is None)

    if isinstance(genes, str):
        genes = [genes]

    adata = adata[:, genes]

    r = lni.fetch_lite_nite_prediction(None, adata)
    found_genes = r.pop('genes')
    if not tuple(genes) == tuple(found_genes):
        raise ValueError(
            'Genes {} do not have LITE and NITE model predictions.'.format(', '.join(np.setdiff1d(genes, found_genes)))
        )
    
    try:
        r['chromatin_differential'] = adata.layers['chromatin_differential'].toarray()
    except KeyError:
        raise KeyError('User must run function "get_cis_differential" before running this function.')

    try:
        r['umap'] = adata.obsm['X_umap']
    except KeyError:
        raise KeyError('X_umap: adata must have a UMAP representation to make this plot.')

    expr = adi.fetch_layer(None, adata, counts_layer)
    if isspmatrix(expr):
        expr = expr.toarray()

    r['expression'] = expr
    r['gene_names'] = genes

    return r


def fetch_streamplot_data(self, adata, 
        data = None,
        layers = None, 
        pseudotime_key = 'mira_pseudotime',
        group_key = 'tree_states',
        tree_graph_key = 'connectivities_tree', 
        group_names_key = 'tree_state_names',
    ):

    if data is None:
        raise Exception('"data" must be names of columns to plot, not None')
        
    if isinstance(data, str):
        data = [data]
    else:
        assert(isinstance(data, list))

    if isinstance(layers, list):
        assert(len(layers) == len(data))
    else:
        layers = [layers] * len(data)

    if len(set(layers)) > 1 and len(data) > len(set(data)): #multiple layer types, nonunique gene labels
        feature_labels = [
            '{}: {}'.format(col, layer) if not layer is None else col
            for col, layer in zip(data, layers)
        ]
    else:
        feature_labels = data

    tree_graph = None
    try:
        tree_graph = adata.uns[tree_graph_key]
    except KeyError:
        logger.warn(
            'User must run "get_tree_structure" or provide a connectivities matrix of size (groups x groups) to specify tree layout. Plotting without tree structure'
        )

    group_names = None
    try:
        group_names = adata.uns[group_names_key]
    except KeyError:
        logger.warn(
            'No group names provided. Assuming groups are named 0,1,...N'
        )

    pseudotime = adata.obs_vector(pseudotime_key)
    group = adata.obs_vector(group_key)

    columns = [
        adata.obs_vector(col, layer = layer)[:, np.newaxis]
        for col, layer in zip(data, layers)
    ]

    numeric_col = [np.issubdtype(col.dtype, np.number) for col in columns]

    assert(
        all(numeric_col) or not any(numeric_col)
    ), 'All plotting features must be either numeric or nonnumeric. Cannot mix.'

    features = np.hstack(columns)

    return dict(
        pseudotime = pseudotime,
        tree_graph = tree_graph,
        group_names = group_names,
        group = group,
        features = features,
        feature_labels = feature_labels,
    )
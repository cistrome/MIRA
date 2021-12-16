from itertools import combinations_with_replacement
import logging
import mira.adata_interface.core as adi

logger = logging.getLogger(__name__)


def fetch_diffmap_eigvals(self, adata, diffmap_key = 'X_diffmap'):
    return dict(
        diffmap = adata.obsm[diffmap_key],
        eig_vals = adata.uns['diffmap_evals']
    )


def add_diffmap(adata, output, diffmap_key = 'X_diffmap'):
    diffmap, eigen_gap = output
    logging.info('Added key to obsm: {}, normalized diffmap with {} components.'.format(
        diffmap_key,
        str(diffmap.shape[-1])
    ))
    logging.info('Added key to uns: eigen_gap')

    logging.warn('Be sure to inspect the diffusion components to make sure they captured the heterogeniety of your dataset. Occasionally, the eigen_gap heuristic truncates the number of diffusion map components too early, leading to loss of variance.')
    adata.obsm[diffmap_key] = diffmap
    adata.uns['eigen_gap'] = eigen_gap


def fetch_connectivities(self, adata, connectivities_key = 'X_diffmap_connectivities'):

    connectivities = adata.obsp[connectivities_key]   
    return dict(connectivities = connectivities)


def fetch_diffmap_distances(self, adata, diffmap_distances_key = 'X_diffmap_distnaces', 
        diffmap_coordinates_key = 'X_diffmap'):

    try:

        distance_matrix = adata.obsp[diffmap_distances_key]
        diffmap = adata.obsm[diffmap_coordinates_key]
    except KeyError:
        raise KeyError(
            '''
You must calculate a diffusion map for the data, and get diffusion-based distances before running this function. Using scanpy:
    
    sc.tl.diffmap(adata)
    sc.pp.neighbors(adata, n_neighbors = 30, use_rep = "X_diffmap", key_added = "X_diffmap")

Or you can set **diffmap_distances_key** to "distances" to use directly use the joint KNN graph.
            '''
        )
    return dict(distance_matrix = distance_matrix, diffmap = diffmap)


def fetch_diffmap_distances_and_components(self, adata, diffmap_distances_key = 'X_diffmap_distances',
        diffmap_coordinates_key = 'X_diffmap'):
    try:
        components = adata.obs_vector('mira_connected_components')
    except KeyError:
        raise KeyError('User must run "get_connected_components" before running this function.')
        
    return dict(
        **fetch_diffmap_distances(self, adata, diffmap_distances_key, diffmap_coordinates_key = diffmap_coordinates_key),
        components = components
    )


def add_transport_map(adata, output):

    pseudotime, transport_map, start_cell = output 

    adata.obs['mira_pseudotime'] = pseudotime
    adata.obsp['transport_map'] = transport_map
    adata.uns['iroot'] = start_cell

    logger.info('Added key to obs: mira_pseudotime')
    logger.info('Added key to obsp: transport_map')
    logger.info('Added key to uns: iroot')


def add_branch_probs(adata, output):
    branch_probs, lineage_names, entropy = output

    adata.obsm['branch_probs'] = branch_probs
    adata.uns['lineage_names'] = lineage_names

    logger.info('Added key to obsm: branch_probs')
    logger.info('Added key to uns: lineage_names')

    for lineage, probs in zip(lineage_names, branch_probs.T):
        adi.add_obs_col(adata, probs, colname=str(lineage) + '_prob')

    adi.add_obs_col(adata, entropy, colname = 'differentiation_entropy')


def fetch_transport_map(self, adata):
    return dict(transport_map = adata.obsp['transport_map'])


def fetch_tree_state_args(self, adata):

    try:
        return dict(
            lineage_names = adata.uns['lineage_names'],
            branch_probs = adata.obsm['branch_probs'],
            pseudotime = adata.obs['mira_pseudotime'].values,
            start_cell = adata.uns['iroot'],
        )
    except KeyError:
        raise KeyError('One of the required pieces to run this function is not present. Make sure you\'ve first run "get_transport_map" and "get_branch_probabilities".')


def add_tree_state_args(adata, output):

    adata.obs['tree_states'] = output['tree_states']
    adata.uns['tree_state_names'] = output['state_names']
    adata.uns['connectivities_tree'] = output['tree']

    logger.info('Added key to obs: tree_states')
    logger.info('Added key to uns: tree_state_names')
    logger.info('Added key to uns: connectivities_tree')
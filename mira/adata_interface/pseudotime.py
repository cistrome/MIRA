from itertools import combinations_with_replacement
import logging
import mira.adata_interface.core as adi
import numpy as np

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


def fetch_connectivities(self, adata, key = 'X_diffmap_connectivities'):

    connectivities = adata.obsp[key]   
    return dict(connectivities = connectivities)


def fetch_diffmap_distances(self, adata, diffmap_distances_key = 'X_diffmap_distances', 
        diffmap_coordinates_key = 'X_diffmap'):

    try:

        distance_matrix = adata.obsp[diffmap_distances_key]
        diffmap = adata.obsm[diffmap_coordinates_key]

        if diffmap_distances_key == 'X_diffmap_distances':
            try:
                adata.uns['X_diffmap']['params']['n_pcs']
            except KeyError:
                logger.warning('''
    Number of diffusion components (n_pcs) to use while calculating diffusion KNN graph was not set.
    This is new as of version 0.1.1. When calculating a diffusion KNN graph following 
        
        sc.tl.diffmap(adata)
        mira.time.normalize_diffmap(adata)

    You can check for the optimal number of diffusion components to use by:

        mira.pl.plot_eigengap(adata)

    The optimal number of components is the where the eigengap hueristic (y-axis)
    peaks. Or the minimum number of components that describes your cells' populations
    as per the UMAPS.

    You must then specify that number of components in the next step:

        sc.pp.neighbors(adata, n_pcs = HERE!, use_rep = "X_diffmap", key_added = "X_diffmap")
    ''')

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


def fetch_diffmap_distances_and_components(self, adata, start_cell = None,
        diffmap_distances_key = 'X_diffmap_distances',
        diffmap_coordinates_key = 'X_diffmap'):

    try:
        components = adata.obs_vector('mira_connected_components')
    except KeyError:
        raise KeyError('User must run "get_connected_components" before running this function.')

    assert(not start_cell is None), 'Must provide a start cell.'
    assert(isinstance(start_cell, (int, str)))

    if isinstance(start_cell, str):
        try:
            start_cell = np.argwhere(adata.obs_names == start_cell)[0,0]
        except IndexError:
            raise ValueError('Cell {} not in adata.obs_names'.format(str(start_cell)))
    elif start_cell >= len(adata):
        raise ValueError('Invalid cell#: {}, only {} cells in dataset.'.format(str(start_cell), str(len(adata))))
        
    return dict(
        **fetch_diffmap_distances(self, adata, diffmap_distances_key, diffmap_coordinates_key = diffmap_coordinates_key),
        components = components,
        start_cell = start_cell
    )


def add_transport_map(adata, output):

    pseudotime, transport_map, start_cell = output 

    adata.obs['mira_pseudotime'] = pseudotime
    adata.obsp['transport_map'] = transport_map
    adata.uns['start_cell'] = adata.obs_names[start_cell]

    logger.info('Added key to obs: mira_pseudotime')
    logger.info('Added key to obsp: transport_map')
    logger.info('Added key to uns: start_cell')


def get_cell_ids(adata, output):
    return adata.obs_names[output].values


def add_branch_probs(adata, output):
    branch_probs, lineage_names, terminal_cells, entropy = output

    adata.obsm['branch_probs'] = branch_probs
    adata.uns['lineage_names'] = lineage_names
    adata.uns['terminal_cells'] = adata.obs_names[terminal_cells].values

    logger.info('Added key to obsm: branch_probs')
    logger.info('Added key to uns: lineage_names')

    for lineage, probs in zip(lineage_names, branch_probs.T):
        adi.add_obs_col(adata, probs, colname=str(lineage) + '_prob')

    adi.add_obs_col(adata, entropy, colname = 'differentiation_entropy')


def fetch_transport_map(self, adata):
    return dict(transport_map = adata.obsp['transport_map'])


def fetch_transport_map_and_terminal_cells(self, adata, terminal_cells = None):

    assert(not terminal_cells is None)
    assert(isinstance(terminal_cells, dict) and len(terminal_cells) > 0)

    termini_dict = {}
    for lineage, cell in terminal_cells.items():
        assert(isinstance(lineage, str)), 'Lineage name {} is not of type str'.format(str(lineage))
        assert(isinstance(cell, (int, str))), 'Cell may be cell# or cell name of type int or str only.'
        if isinstance(cell, str):
            try:
                cell = np.argwhere(adata.obs_names == cell)[0,0]
            except IndexError:
                raise ValueError('Cell {} not found in adata.obs_names'.format(str(cell)))
        elif cell >= len(adata):
            raise ValueError('Invalid cell#: {}, only {} cells in dataset.'.format(str(cell), str(len(adata))))

        termini_dict[lineage] = cell

    assert(
        len(np.unique(termini_dict.keys()) == len(termini_dict))
    ), 'All lineage names must be unique'

    return dict(transport_map = adata.obsp['transport_map'], terminal_cells = termini_dict)


def fetch_tree_state_args(self, adata):

    try:
        return dict(
            lineage_names = adata.uns['lineage_names'],
            branch_probs = adata.obsm['branch_probs'],
            pseudotime = adata.obs['mira_pseudotime'].values,
            start_cell = np.argwhere(adata.obs_names == adata.uns['start_cell'])[0,0],
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


def fetch_eigengap(self, adata, basis = 'X_umap'):

    try:
        umap = adata.obsm[basis]
    except KeyError:
        raise KeyError('Basis {} has not been calculated'.format(str(basis)))

    try:
        eigvals = adata.uns['diffmap_evals']
        diffmap = adata.obsm['X_diffmap']
    except KeyError:
        raise KeyError('User must run "sc.tl.diffmap" before running this function.')
    
    try:
        eigen_gap = adata.uns['eigen_gap']
    except KeyError:
        raise KeyError('User must run "mira.time.normalized_diffmap" before running this function.')

    return dict(
        umap = umap,
        diffmap = diffmap, 
        eigen_gap = eigen_gap,
        eigvals = eigvals
    )
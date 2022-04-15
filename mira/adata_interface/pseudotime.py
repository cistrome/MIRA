from itertools import combinations_with_replacement
import logging
from tracemalloc import start
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

def validate_cell(adata, start_cell):

    assert(not start_cell is None), 'Must provide a start cell.'
    assert(isinstance(start_cell, (int, str)))

    if isinstance(start_cell, str):
        try:
            start_cell = np.argwhere(adata.obs_names == start_cell)[0,0]
        except IndexError:
            raise ValueError('Cell {} not in adata.obs_names'.format(str(start_cell)))
    elif start_cell >= len(adata):
        raise ValueError('Invalid cell#: {}, only {} cells in dataset.'.format(str(start_cell), str(len(adata))))

    return start_cell


def fetch_diffmap_distances_and_components(self, adata, start_cell = None,
        diffmap_distances_key = 'X_diffmap_distances',
        diffmap_coordinates_key = 'X_diffmap'):

    try:
        components = adata.obs_vector('mira_connected_components')
    except KeyError:
        raise KeyError('User must run "get_connected_components" before running this function.')

    start_cell = validate_cell(adata, start_cell)
        
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
        
        cell = validate_cell(adata, cell)

        termini_dict[lineage] = cell

    assert(
        len(np.unique(termini_dict.keys()) == len(termini_dict))
    ), 'All lineage names must be unique'

    return dict(transport_map = adata.obsp['transport_map'], terminal_cells = termini_dict)


def fetch_tree_state_args(self, adata, cellrank = False, 
    pseudotime_key = 'mira_pseudotime', branch_probs_key = 'branch_probs',
    lineage_names_key = 'lineage_names', start_cell = None):

    if start_cell is None:
        try:
            start_cell = adata.uns['start_cell']
        except KeyError:
            raise KeyError('No start cell provided, and start cell not found in .uns["start_cell"]. To run this function, provide a start cell.')

    if not cellrank:
        try:
            return dict(
                lineage_names = adata.uns[lineage_names_key],
                branch_probs = adata.obsm[branch_probs_key],
                pseudotime = adata.obs[pseudotime_key].values,
                start_cell = validate_cell(adata, start_cell),
            )
        except KeyError:
            raise KeyError('One of the required pieces to run this function is not present. Make sure you\'ve first run "get_transport_map" and "get_branch_probabilities".')
    else:
        return dict(
                lineage_names = adata.obsm['to_terminal_states'].names,
                branch_probs = np.array(adata.obsm['to_terminal_states']),
                pseudotime = adata.obs[pseudotime_key].values,
                start_cell = validate_cell(adata, start_cell),
            )


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

def fetch_trace_args(self, adata, 
    basis = 'X_umap', 
    pseudotime_key = 'mira_pseudotime',
    diffmap_distances_key = 'X_diffmap_distances', 
    diffmap_coordinates_key = 'X_diffmap',
    transport_map_key = None,
    start_cells = None, start_lineage = None,
    num_start_cells = 50):

    out = {}
    try:
        out['basis'] = adata.obsm[basis]
    except KeyError:
        raise KeyError('Basis {} has not been calculated'.format(str(basis)))

    if transport_map_key is None:
        try:
            out.update(
                fetch_diffmap_distances(self, adata, diffmap_coordinates_key=diffmap_coordinates_key,
                    diffmap_distances_key=diffmap_distances_key)
            )
            out['pseudotime'] = adata.obs_vector(pseudotime_key)
            del out['diffmap']
        except KeyError:
            raise KeyError('If no transport map key provided, uses distance matrix and pseudotime to produce a new transport map.')

    if start_cells is None and start_lineage is None:
        raise ValueError('One of either "start_cells" or "start_lineage" must be given.')

    if not start_cells is None:

        assert(isinstance(start_cells, (list, np.ndarray))), 'If provided, "start_cells" must be a list or np.ndarray of barcodes, cell idx, or a boolean mask of cells.'
        if isinstance(start_cells, list):
            start_cells = np.array(start_cells)

        if len(start_cells) == len(adata):

            assert(start_cells.dtype in [bool, int]), 'If providing a mask over all cells as start cells, mask must be of type bool or int.'
            start_cells = start_cells.astype(bool)

        else:
            if start_cells.dtype.kind == 'U':
                
                excluded_barcodes = np.setdiff1d(start_cells, adata.obs_names.values)
                assert(len(excluded_barcodes) == 0), 'Barcodes {} are not in adata.obs_names.'.format(
                    ', '.join(excluded_barcodes)
                )

                start_cells = np.argwhere(np.isin(adata.obs_names.values, start_cells))[:,0]

                arr = np.zeros(len(adata))
                arr[start_cells] = 1
                start_cells = arr.astype(bool)

            elif start_cells.dtype.kind == 'i':

                arr = np.zeros(len(adata))
                arr[start_cells] = 1
                start_cells = arr.astype(bool)

            else:
                raise ValueError('Providing an array/list of type {} for "start_cells" is not supported.'.format(start_cells.dtype))

    else: 

        lineage_names, branch_probs = adata.uns['lineage_names'], adata.obsm['branch_probs']

        branch_probs = dict(zip(lineage_names, branch_probs.T))
        assert(start_lineage in lineage_names)
        assert(isinstance(num_start_cells, int) and num_start_cells < len(adata))

        start_cells = (-branch_probs[start_lineage]).argsort().argsort() < num_start_cells

    out['start_cells'] = start_cells

    if not transport_map_key is None:
        out['transport_map'] = adata.obsp[transport_map_key]   
    
    return out
    
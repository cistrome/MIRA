from networkx.algorithms import components
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse.base import isspmatrix
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import inv, pinv
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from scipy.sparse import csgraph
from scipy.stats import entropy, pearsonr, norm
from copy import deepcopy
from scipy.sparse.linalg import eigs
import logging
import tqdm
from tqdm.std import trange
import mira.adata_interface.core as adi
import mira.adata_interface.pseudotime as pti
from functools import partial

logger = logging.getLogger(__name__)


class TreeException(Exception):
    pass

def get_root_state(G):
    G = G.astype(bool)

    incoming_connections = G.sum(0) #(incoming, outgoing)

    if (incoming_connections == 0).sum() > 1:
        raise TreeException('Graph has multiple possible sources, and is not a tree')

    return np.argwhere(incoming_connections == 0)[0,0]

def num_children(G, node_idx):
    G = G.astype(bool)
    return G[node_idx, :].sum()

def get_children(G, node_idx):
    G = G.astype(bool)

    return np.argwhere(G[node_idx, :])[:,0]

def is_leaf(G, node_idx):
    
    G = G.astype(bool)
    return num_children(G, node_idx) == 0

def get_dendogram_levels(G):
    G = G.astype(bool)
    nx_graph = nx.convert_matrix.from_numpy_array(G)
    
    start_node = get_root_state(G)
    dfs_tree = list(nx.dfs_predecessors(nx_graph, start_node))[::-1] + [start_node]

    centerlines = {}
    num_termini = [0]

    def get_or_set_node_position(node):

        if not node in centerlines:
            if is_leaf(G, node):
                centerlines[node] = num_termini[0]
                num_termini[0]+=1
            else:
                centerlines[node] = np.mean([get_or_set_node_position(child) for child in get_children(G, node)])

        return centerlines[node]

    for node in dfs_tree:
        get_or_set_node_position(node)

    return centerlines

### ___ PALANTIR FUNCTIONS ____ ##

@adi.wraps_functional(pti.fetch_diffmap_eigvals, pti.add_diffmap, ['diffmap','eig_vals'])
def normalize_diffmap(*,diffmap, eig_vals):

    num_comps = np.maximum(np.argmax(eig_vals[:-1] - eig_vals[1:]), 4)
    diffmap = diffmap[:, 1:num_comps]
    
    diffmap/=np.linalg.norm(diffmap, axis = 0, keepdims = True)
    eig_vals = eig_vals[1:num_comps]
    diffmap *= (eig_vals / (1 - eig_vals))[np.newaxis, :]

    return diffmap


def sample_waypoints(num_waypoints = 3000,*, diffmap):

    np.random.seed(2556)

    waypoint_set = list()
    no_iterations = int((num_waypoints) / diffmap.shape[1])

    # Sample along each component
    N = len(diffmap)
    for feature_col in diffmap.T:

        # Random initialzlation
        iter_set = [np.random.choice(N)]

        # Distances along the component
        dists = np.zeros([N, no_iterations])
        dists[:, 0] = np.abs(feature_col - feature_col[iter_set])
        for k in range(1, no_iterations):
            # Minimum distances across the current set
            min_dists = dists[:, 0:k].min(axis=1)

            # Point with the maximum of the minimum distances is the new waypoint
            new_wp = np.where(min_dists == min_dists.max())[0][0]
            iter_set.append(new_wp)

            # Update distances
            dists[:, k] = np.abs(feature_col - feature_col[new_wp])

        # Update global set
        waypoint_set = waypoint_set + iter_set

    # Unique waypoints
    waypoints = np.unique(waypoint_set)

    return waypoints


@adi.wraps_functional(pti.fetch_diffmap_distances, 
    partial(adi.add_obs_col, colname = 'mira_connected_components'),
    ['distance_matrix','diffmap'])
def get_connected_components(*,distance_matrix, diffmap):

    assert(isspmatrix(distance_matrix))
    N = distance_matrix.shape[0]
    G = nx.convert_matrix.from_scipy_sparse_matrix(distance_matrix)

    components = nx.algorithms.components.connected_components(G)
    
    num_components = 0
    component_num = np.full(N, 0, dtype = int)
    for i, nodes in enumerate(components):
        num_components+=1
        for node in nodes:
            component_num[node] = i
    logger.info('Found {} components of KNN graph.'.format(str(num_components)))

    return component_num.astype(str)


def get_pseudotime(max_iterations = 25, n_waypoints = 3000, n_jobs = -1,*, start_cell, distance_matrix, diffmap):

    assert(isinstance(max_iterations, int) and max_iterations > 0)
    N = distance_matrix.shape[0]
    cells = np.union1d(sample_waypoints(num_waypoints=n_waypoints, diffmap = diffmap), np.array([start_cell]))
    
    logger.info('Calculting inter-cell distances ...')
    # Distances
    dists = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(csgraph.dijkstra)(distance_matrix, False, cell)
        for cell in cells
    )

    # Convert to distance matrix
    D = np.vstack([d[np.newaxis, :] for d in dists])

    start_waypoint_idx = np.argwhere(cells == start_cell)[0]

    # Determine the perspective matrix
    # Waypoint weights
    sdv = np.std(np.ravel(D)) * 1.06 * len(np.ravel(D)) ** (-1 / 5)
    W = np.exp(-0.5 * np.power((D / sdv), 2))
    # Stochastize the matrix
    W = W / W.sum(0)

    # Initalize pseudotime to start cell distances
    pseudotime = D[start_waypoint_idx, :].reshape(-1)
    converged = False
    
    max_computations = max_iterations * (len(cells) - 1)

    t = trange(max_computations, desc = 'Calculating pseudotime')
    _t = iter(t)
    # Iteratively update perspective and determine pseudotime
    for iteration in range(max_iterations):
        # Perspective matrix by alinging to start distances
        P = deepcopy(D)
        for i, waypoint_idx in enumerate(cells):
            # Position of waypoints relative to start
            if waypoint_idx != start_cell:
                idx_val = pseudotime[waypoint_idx]

                # Convert all cells before starting point to the negative
                before_indices = pseudotime < idx_val
                P[i, before_indices] = -D[i, before_indices]

                # Align to start
                P[i, :] = P[i, :] + idx_val

            next(_t)

        # Weighted pseudotime
        new_traj = np.multiply(P,W).sum(0)
        # Check for convergence
        corr = pearsonr(pseudotime, new_traj)[0]
        if corr > 0.9999:
            converged = True

        # If not converged, continue iteration
        pseudotime = new_traj

        if converged:
            break

    pseudotime = pseudotime - pseudotime.min() #make 0 minimum
    waypoint_weights = W
    
    return pseudotime


def make_markov_matrix(affinity_matrix):
    inverse_rowsums = sparse.diags(1/np.array(affinity_matrix.sum(axis = 1)).reshape(-1)).tocsr()
    markov_matrix = inverse_rowsums.dot(affinity_matrix)
    return markov_matrix


def get_kernel_width(ka, distance_matrix):

    N = distance_matrix.shape[0]
    distance_matrix = distance_matrix.tocsr()
    indptr = distance_matrix.indptr
    k = indptr[1] - indptr[0]
    assert(np.all(indptr[1:] - indptr[0:-1] == k)), 'distance matrix is not a valid Knn matrix. Different numbers of neighbors for each cell.'

    j, i, d = sparse.find(distance_matrix.T)
    distances = d.reshape(N, k)
    
    sorted_dist = np.sort(distances, axis = 1)
    #ka_index = np.minimum(np.argmin(~np.isinf(sorted_dist), axis = 1) - 1, ka)
    kernel_width = sorted_dist[:, ka]

    return kernel_width


def prune_edges(*,kernel_width, distance_matrix, pseudotime):
    
    N = distance_matrix.shape[0]
    j, i, d = sparse.find(distance_matrix.T)

    delta_pseudotime = pseudotime[i] - pseudotime[j]
    rem_edges = delta_pseudotime > kernel_width[i]
    
    d[rem_edges] = np.inf

    distance_matrix = sparse.csr_matrix((d, (i, j)), [N,N])
    
    return distance_matrix


def get_adaptive_affinity_matrix(*,kernel_width, distance_matrix, pseudotime = None):

    N = distance_matrix.shape[0]
    j, i, d = sparse.find(distance_matrix.T)

    affinity = np.exp(
        -(d ** 2) / kernel_width[i]**2 * 0.5
        -(d ** 2) / kernel_width[j]**2 * 0.5
    )

    affinity_matrix = sparse.csr_matrix((affinity, (i, j)), [N,N])
    affinity_matrix.eliminate_zeros()

    return affinity_matrix


@adi.wraps_functional(pti.fetch_diffmap_distances_and_components, pti.add_transport_map, 
    ['distance_matrix','diffmap','components'])
def get_transport_map(ka = 5, n_jobs = -1,*, start_cell, distance_matrix, diffmap,
                components):
    
    assert(len(np.unique(components)) == 1), 'Graphs with multiple connected components may not be used. Subset cells to include only one connected component.'

    #logger.info('Calculating diffusion pseudotime ...')
    pseudotime = get_pseudotime(n_jobs = n_jobs, start_cell= start_cell, 
        distance_matrix = distance_matrix, diffmap = diffmap)

    kernel_width = get_kernel_width(ka, distance_matrix)

    distance_matrix = prune_edges(kernel_width = kernel_width, 
        pseudotime = pseudotime, distance_matrix = distance_matrix)

    logger.info('Calculating transport map ...')
    affinity_matrix = get_adaptive_affinity_matrix(kernel_width = kernel_width,
        distance_matrix = distance_matrix)

    transport_map = make_markov_matrix(affinity_matrix)

    return pseudotime, transport_map, start_cell

@adi.wraps_functional(pti.fetch_transport_map, adi.return_output, ['transport_map'])
def find_terminal_cells(iterations = 1, max_termini = 10, threshold = 1e-3, *, transport_map):

    assert(transport_map.shape[0] == transport_map.shape[1])
    assert(len(transport_map.shape) == 2)

    def _get_stationary_points():

        vals, vectors = eigs(transport_map.T, k = max_termini)

        stationary_vecs = np.isclose(np.real(vals), 1., threshold)

        return list(np.real(vectors)[:, stationary_vecs].argmax(0))

    terminal_points = set()
    for i in range(iterations):
        terminal_points = terminal_points.union(_get_stationary_points())

    logger.info('Found {} terminal states from stationary distribution.'.format(str(len(terminal_points))))

    return np.array(list(terminal_points))

@adi.wraps_functional(pti.fetch_transport_map, pti.add_branch_probs, ['transport_map'])
def get_branch_probabilities(*, transport_map, terminal_cells):

    logger.info('Simulating random walks ...')

    lineage_names, absorbing_cells = list(zip(*terminal_cells.items()))

    absorbing_states_idx = np.array(absorbing_cells)
    num_absorbing_cells = len(absorbing_cells)
    
    absorbing_states = np.zeros(transport_map.shape[0]).astype(bool)
    absorbing_states[absorbing_states_idx] = True

    transport_map = transport_map.toarray()
    # Reset absorption state affinities by Removing neigbors
    transport_map[absorbing_states, :] = 0
    # Diagnoals as 1s
    transport_map[absorbing_states, absorbing_states] = 1

    # Fundamental matrix and absorption probabilities
    # Transition states
    trans_states = ~absorbing_states

    # Q matrix
    Q = transport_map[trans_states, :][:, trans_states]

    # Fundamental matrix
    mat = np.eye(Q.shape[0]) - Q

    N = inv(mat)

    # Absorption probabilities
    branch_probs = np.dot(N, transport_map[trans_states, :][:, absorbing_states_idx])
    branch_probs[branch_probs < 0] = 0.

    # Add back terminal states
    branch_probs_including_terminal = np.full((transport_map.shape[0], num_absorbing_cells), 0.)
    branch_probs_including_terminal[trans_states] = branch_probs
    branch_probs_including_terminal[absorbing_states_idx, np.arange(num_absorbing_cells)] = 1.

    return branch_probs_including_terminal, \
        list(lineage_names), entropy(branch_probs_including_terminal, axis = -1)


## __ TREE INFERENCE FUNCTIONS __ ##

def get_lineages(*,branch_probs, start_cell):

    return get_lineage_prob_fc(branch_probs = branch_probs, start_cell = start_cell) >= 0


def get_lineage_prob_fc(*, branch_probs, start_cell):

    ep = 0.01
    lineage_prob_fc = np.hstack([
        (np.log2(lineage_prob + ep) - np.log2(lineage_prob[start_cell] + ep) )[:, np.newaxis]
        for lineage_prob in branch_probs.T
    ])

    return lineage_prob_fc

def get_lineage_branch_time(lineage1, lineage2, pseudotime, prob_fc, threshold = 0.5):

    lin_mask = np.logical_or(prob_fc[:, lineage1] > 0, prob_fc[:, lineage2] > 0)
    divergence = prob_fc[lin_mask, lineage1] - prob_fc[lin_mask, lineage2]

    state_1 = pseudotime[lin_mask][divergence > threshold]
    state_2 = pseudotime[lin_mask][divergence < -threshold]
    
    
    if len(state_1) == 0:
        return state_2.min()
    elif len(state_2) == 0:
        return state_1.min()
    else:
        return max(state_1.min(), state_2.min())


@adi.wraps_functional(pti.fetch_tree_state_args, pti.add_tree_state_args, 
    ['lineage_names', 'branch_probs', 'pseudotime', 'start_cell'])
def get_tree_structure(threshold = 0.1,*, lineage_names, branch_probs, pseudotime, start_cell):

    def get_all_leaves_from_node(edge):

        if isinstance(edge, tuple):
            return [*get_all_leaves_from_node(edge[0]), *get_all_leaves_from_node(edge[1])]
        else:
            return [edge]

    def get_node_name(self, node):
        return ', '.join(map(str, self.lineage_names[self.get_all_leaves_from_node(node)]))

    def merge_rows(x, col1, col2):
        return np.hstack([
            (x[:,col1] + x[:, col2])[:, np.newaxis], #merge two lineages into superlineage
            x[:, ~np.isin(np.arange(x.shape[-1]), [col1, col2])]
        ])


    num_cells, num_lineages = branch_probs.shape
    all_merged = False

    tree_states = np.zeros(num_cells)

    lineages = get_lineages(branch_probs = branch_probs, start_cell = start_cell)
    branch_probs = branch_probs.copy()
    lineage_names = list(lineage_names)

    lineage_tree = nx.DiGraph()
    states_assigned = 1

    while not all_merged:

        prob_fc = get_lineage_prob_fc(branch_probs = branch_probs, start_cell = start_cell)
        
        split_time_matrix = np.full((num_lineages, num_lineages), -1.0)
        for i in range(0,num_lineages-1):
            for j in range(i+1, num_lineages):

                branch_time = get_lineage_branch_time(i, j, pseudotime, prob_fc, threshold)
                split_time_matrix[i,j] = branch_time
        
        branch_time = split_time_matrix.max()
        latest_split_event = np.where(split_time_matrix == branch_time)
        merge1, merge2 = latest_split_event[0][0], latest_split_event[1][0]

        new_branch_name = (lineage_names[merge1], lineage_names[merge2])

        assign_cells_mask = np.logical_and(pseudotime >= branch_time, np.logical_or(lineages[:,merge1], lineages[:, merge2]))
        assign_cells_mask = np.logical_and(assign_cells_mask, ~tree_states.astype(bool))
        
        divergence = prob_fc[assign_cells_mask, merge1] - prob_fc[assign_cells_mask, merge2]
        get_assign_indices = lambda y : np.argwhere(assign_cells_mask)[:,0][y * divergence > 0]

        tree_states[get_assign_indices(1)] = states_assigned
        lineage_tree.add_edge(new_branch_name, lineage_names[merge1], branch_time = branch_time, state = states_assigned)
        states_assigned+=1

        tree_states[get_assign_indices(-1)] = states_assigned
        lineage_tree.add_edge(new_branch_name, lineage_names[merge2], branch_time = branch_time, state = states_assigned)
        states_assigned+=1

        lineages = merge_rows(lineages, merge1, merge2).astype(bool)
        branch_probs = merge_rows(branch_probs, merge1, merge2)
        lineage_names = [new_branch_name] + [lin for i, lin in enumerate(lineage_names) if not i in [merge1, merge2]]

        num_lineages = lineages.shape[-1]
        
        if num_lineages == 1:
            all_merged = True

    lineage_tree.add_edge('Root', lineage_names[0], branch_time = -1, state = 0)

    def get_node_name(node):
        if node == 'Root':
            return node

        return ', '.join(set(get_all_leaves_from_node(node)))
            
    state_names = {
        edge[2]['state'] : get_node_name(edge[1])
        for edge in lineage_tree.edges(data = True)
    }
    
    return {
        'tree_states' : [state_names[s] for s in tree_states.astype(int)],
        'tree' : nx.to_numpy_array(lineage_tree, weight='branch_time'),
        'state_names' : [get_node_name(node) for node in lineage_tree.nodes],
    }
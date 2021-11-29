
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as pltcolors
from mira.plots.base import map_colors
from matplotlib.patches import Patch
import warnings
from scipy.signal import savgol_filter
from sklearn.preprocessing import minmax_scale
import networkx as nx
from functools import partial
from mira.pseudotime.pseudotime import get_dendogram_levels, get_root_state, is_leaf
from mira.plots.base import map_plot
import mira.adata_interface.core as adi
import mira.adata_interface.plots as pli
from mira.plots.swarmplot import _plot_swarm_segment, _get_swarm_colors
import logging
logger = logging.getLogger(__name__)


def _plot_fill(is_top = False,*, ax, time, fill_top, fill_bottom, color, linecolor, linewidth, alpha = 1.):

    ax.fill_between(time, fill_top, fill_bottom, color = color, alpha = alpha)

    if not linecolor is None:
        ax.plot(time, fill_top, color = linecolor, linewidth = linewidth)
        if is_top:
            ax.plot(time, fill_bottom, color = linecolor, linewidth = linewidth)


def _plot_stream_segment(is_leaf = True, centerline = 0, window_size = 101,center_baseline = True, is_root = True,
        palette = 'Set3', linecolor = 'black', linewidth = 0.1, feature_labels = None, hide_feature_threshold = 0,
        hue_order = None, show_legend = True, max_bar_height = 0.6, legend_cols = 5,
        color = 'black', segment_connection = None,*, ax, features, pseudotime, **kwargs,):

    if (features.shape) == 1:
        features = features[:, np.newaxis]

    num_samples, num_features = features.shape
    max_windowsize = num_samples - 1
    if max_windowsize % 2 == 0:
        max_windowsize-=1

    if feature_labels is None:
        feature_labels = np.arange(num_features).astype(str)

    features = np.where(features < hide_feature_threshold, 0., features)

    features = features[np.argsort(pseudotime)] #sort by time
    features = savgol_filter(features, min(window_size, max_windowsize), 1, axis = 0) #smooth
    ascending_time = pseudotime[np.argsort(pseudotime)] #sort time

    features = np.cumsum(features, axis=-1)
    
    if center_baseline:
        baseline_adjustment = (features[:,-1]/2)[:, np.newaxis]
    else:
        baseline_adjustment = np.ones((features.shape[0], 1)) * -max_bar_height/2

    feature_fill_positions = features - baseline_adjustment + centerline
    feature_bottom = centerline - baseline_adjustment.reshape(-1)
    linecolor = linecolor if num_features > 1 else color
    
    plot_kwargs = dict(
        ax = ax, time = ascending_time,
        linecolor = linecolor, linewidth = linewidth
    )

    legend_params = dict(loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon = False, ncol = legend_cols, 
                title_fontsize='x-large', fontsize='large', markerscale = 1)
    
    if num_features == 1:
        _plot_fill(is_top=True, fill_top= feature_fill_positions[:,0], fill_bottom = feature_bottom, color = color, **plot_kwargs)
    else:

        not_all_zeros = ~np.all(features <= hide_feature_threshold, axis = 0)

        for i, color in enumerate(
                map_colors(ax, feature_labels[::-1], add_legend = is_root and show_legend, 
                    hue_order = hue_order[::-1] if not hue_order is None else None, 
                    legend_kwargs = legend_params,
                          palette = palette)
            ):
            col_num = num_features - 1 - i
            if not_all_zeros[col_num]:
                _plot_fill(is_top = i == 0, fill_top= feature_fill_positions[:, col_num], fill_bottom = feature_bottom, 
                    color = color, **plot_kwargs)

    #box in borderlines
    ax.vlines(ascending_time[0], ymin = feature_bottom[0], ymax = feature_fill_positions[0, -1], 
        color = linecolor, linewidth = linewidth)

    ax.vlines(ascending_time[-1], ymin = feature_bottom[-1], ymax = feature_fill_positions[-1, -1], 
        color = linecolor, linewidth = linewidth)
    ax.axis('off')


def _plot_scatter_segment(is_leaf = True, centerline = 0, window_size = 101, is_root = True, size = 3, show_points = True,
        palette = 'Set3', linecolor = 'black', linewidth = 0.5, feature_labels = None, hue_order = None, show_legend = True, legend_cols = 5,
        color = 'black', max_bar_height = 0.6, alpha = 1.,*, ax, features, pseudotime, **kwargs,):
    
    if (features.shape) == 1:
        features = features[:, np.newaxis]

    num_samples, num_features = features.shape
    max_windowsize = num_samples - 1
    if max_windowsize % 2 == 0:
        max_windowsize-=1

    if feature_labels is None:
        feature_labels = np.arange(num_features).astype(str)

    features = features[np.argsort(pseudotime)] #sort by time
    smoothed_features = savgol_filter(features, min(window_size, max_windowsize), 1, axis = 0) #smooth
    ascending_time = pseudotime[np.argsort(pseudotime)] #sort time

    #legend_params = dict(loc="center left", markerscale = 1, frameon = False, title_fontsize='x-large', fontsize='large',
    #                        bbox_to_anchor=(1.05, 0.5))

    legend_params = dict(loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon = False, ncol = legend_cols, 
                title_fontsize='x-large', fontsize='large', markerscale = 1)

    min_time, max_time = pseudotime.min(), pseudotime.max()
    centerline -= max_bar_height/2
    
    def scatter(features, smoothed_features, _color):

        if show_points:
            ax.scatter(
                ascending_time,
                centerline + features,
                color = _color,
                s = size
            )

        ax.plot(
            ascending_time,
            smoothed_features + centerline,
            color = _color, 
            linewidth = np.sqrt(size),
            alpha = alpha,
        )
        
        ax.vlines(min_time, ymin = centerline, ymax = centerline + max_bar_height, color = linecolor, linewidth = linewidth)
        ax.vlines(max_time, ymin = centerline, ymax = centerline + max_bar_height, color = linecolor, linewidth = linewidth)
        #ax.hlines(centerline, xmin = min_time, xmax = max_time, color = linecolor, linewidth = linewidth)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        #ax.spines['left'].set_visible(False)
        #ax.set(xticks = [], yticks = [])
        #ax.hlines(centerline, xmin = min_time, xmax = max_time, color = linecolor, linewidth = linewidth)
    
    if num_features == 1:
        scatter(features[:,0], smoothed_features[:,0], color)
        
    else:
        for i, color in enumerate(
                map_colors(ax, feature_labels[::-1], add_legend = is_root and show_legend, 
                    hue_order = hue_order[::-1] if not hue_order is None else None, 
                    legend_kwargs = legend_params,
                    palette = palette)
            ):
            scatter(features[:, num_features - 1 - i], smoothed_features[:, num_features - 1 - i], color)

    ax.axis('off')
            
def _plot_heatmap_segment(is_leaf = True, centerline = 0, window_size = 101, is_root = True,
        palette = 'inferno', linecolor = 'black', linewidth = 0.5, feature_labels = None, hue_order = None, show_legend = True, 
        color = 'black', max_bar_height = 0.6,*, ax, features, pseudotime, **kwargs,):
    
    if (features.shape) == 1:
        features = features[:, np.newaxis]

    num_samples, num_features = features.shape
    max_windowsize = num_samples - 1
    if max_windowsize % 2 == 0:
        max_windowsize-=1

    if feature_labels is None:
        feature_labels = np.arange(num_features).astype(str)

    features = features[np.argsort(pseudotime)] #sort by time
    #smoothed_features = savgol_filter(features, min(window_size, max_windowsize), 1, axis = 0) #smooth
    ascending_time = pseudotime[np.argsort(pseudotime)] #sort time

    optimal_num_bins = num_samples//window_size
    samples_per_bin = int(num_samples/optimal_num_bins + 1)

    sample_bin_num = np.arange(num_samples)//samples_per_bin
    num_bins = sample_bin_num[-1] + 1

    Y = np.linspace(centerline - max_bar_height/2, centerline + max_bar_height/2, num_features + 1)
    X = np.full(num_bins + 1, 0.)
    C = np.full((num_features, num_bins), 0.)

    for _bin in range(num_bins):
        bin_mask = sample_bin_num == _bin
        C[:, _bin] = np.nanmean(features[bin_mask, :], axis = 0)
        X[_bin] = ascending_time[bin_mask].min()
        X[_bin + 1] = ascending_time[bin_mask].max()

    C = minmax_scale(C, axis = -1)

    ax.pcolormesh(
        X, Y, np.flip(C, 0), cmap = palette, edgecolors = linecolor
    )

    if is_root:
        ax.set_yticks(Y[:-1] + (Y[1:] - Y[:-1])/2)
        ax.set_yticklabels(feature_labels[::-1])
        ax.set_xticks([])
    else:
        ax.set_yticks([])

    for spine in ['left','right','bottom','top']:
        ax.spines[spine].set_visible(False)
    
    ax.set(xlim = (pseudotime.min(), pseudotime.max() + 0.05))


def _plot_pseudotime_scale(*, ax, pseudotime, plot_bottom = 0):

    start, end = pseudotime.min(), pseudotime.max()
    base = plot_bottom
    ax.fill_between(
        [start, end], 
        [base, base + 0.15], 
        [base, base],
        color = 'lightgrey'
    )
    ax.text(1.005*end, base, 'Time', fontsize = 'large', ha = 'left')


def _plot_scaffold(is_leaf = True, centerline = 0, linecolor = 'lightgrey', linewidth = 1, lineage_name = '',*,
    segment_connection, ax, features, pseudotime, **kwargs):

    (prev_center, prev_maxtime), (curr_center, curr_maxtime) = segment_connection

    ax.vlines(prev_maxtime, ymin = prev_center, ymax = curr_center, color = linecolor, linewidth = linewidth)
    ax.hlines(curr_center, xmin = prev_maxtime, xmax = curr_maxtime, color = linecolor, linewidth = linewidth)

    if is_leaf:
        ax.text(pseudotime.max()*1.01, centerline, lineage_name, fontsize='x-large', ha = 'left')
        
    
def _build_tree(cell_colors = None, size = None, shape = None,*, max_bar_height = 0.75, ax, features, pseudotime, cluster_id, tree_graph, lineage_names, min_pseudotime, plot_fn):

    centerlines = get_dendogram_levels(tree_graph)
    source = get_root_state(tree_graph)
    nx_graph = nx.convert_matrix.from_numpy_array(tree_graph)

    max_times, min_times, min_centerline, max_centerline = {}, {}, np.inf, -np.inf
    max_times[source] = pseudotime.min()
    min_times[source] = pseudotime.min()

    has_plotted = False
    for i, (start_clus, end_clus) in enumerate(
            [(source, source), *nx.algorithms.traversal.bfs_edges(nx_graph, source)]
        ):

        centerline = centerlines[end_clus]

        segment_mask = cluster_id == lineage_names[end_clus]

        segment_features = features[segment_mask]
        segment_pseudotime = pseudotime[segment_mask]
        segment_cell_colors = None if cell_colors is None else cell_colors[segment_mask]

        if len(segment_features) > 0:
            
            max_times[end_clus] = segment_pseudotime.max()
            min_times[end_clus] = segment_pseudotime.min()
            
            connection = ((centerlines[start_clus], max_times[start_clus]), 
                            (centerlines[end_clus], min_times[end_clus]))

            segment_is_leaf = is_leaf(tree_graph, end_clus)

            plot_fn(features = segment_features, pseudotime = segment_pseudotime, is_leaf = segment_is_leaf, is_root = not has_plotted, ax = ax,
                centerline = centerline, lineage_name = lineage_names[end_clus], segment_connection = connection, cell_colors = segment_cell_colors)
            has_plotted = True

            min_centerline = min(min_centerline, centerline)
            max_centerline = max(max_centerline, centerline)

        else:
            max_times[end_clus] = pseudotime.min()
            min_times[end_clus] = pseudotime.min()

    plot_bottom = min_centerline - max_bar_height/2 - 0.3
    ax.set(ylim = (plot_bottom, max_centerline + max_bar_height/2 + 0.15))

    return plot_bottom


def _normalize_numerical_features(features,*, clip, scale_features, max_bar_height, style):

    if not clip is None:
        means, stds = features.mean(0, keepdims = True), features.std(0, keepdims = True)
        clip_min, clip_max = means - clip*stds, means + clip*stds
        features = np.clip(features, clip_min, clip_max)

    features_min, features_max = features.min(0, keepdims = True), features.max(0, keepdims = True)
    
    if scale_features:
        features = (features - features_min)/(features_max - features_min) #scale relative heights of features
    else:
        features = features-features_min 

    features = np.maximum(features, 0) #just make sure no vals are negative

    if style == 'stream':
        features = features/(features.sum(-1).max()) * max_bar_height
    elif style == 'scatter':
        features = features/(features.max(0)) * max_bar_height

    return features
    

@adi.wraps_functional(pli.fetch_streamplot_data, adi.return_output,
    ['group_names','features','pseudotime','group','tree_graph', 'feature_labels']
)
def plot_stream(style = 'stream', split = False, log_pseudotime = True, scale_features = False, order = 'ascending',
    title = None, show_legend = True, legend_cols = 5, max_bar_height = 0.6, size = None, max_swarm_density = 2000, hide_feature_threshold = 0,
    palette = None, color = 'black', linecolor = 'black', linewidth = None, hue_order = None, pseudotime_triangle = True,
    scaffold_linecolor = 'lightgrey', scaffold_linewidth = 1, min_pseudotime = 0.05,
    figsize = (10,5), ax = None, plots_per_row = 4, height = 4, aspect = 1.3, tree_structure = True,
    center_baseline = True, window_size = 101, clip = 10, alpha = 1., vertical = False,
    feature_labels = None, group_names = None, tree_graph = None,*, features, pseudotime, group):

    assert(isinstance(max_bar_height, float) and max_bar_height > 0 and max_bar_height <= 1)
    assert(isinstance(features, np.ndarray))
    assert(style in ['line','scatter','stream','swarm', 'heatmap'])

    show_points = True
    if style == 'line':
        style = 'scatter'
        show_points = False

    if not style == 'swarm':
        assert(np.issubdtype(features.dtype, np.number)), 'Only "swarm" style can plot categorical/non-numeric data.'
    
    if len(features.shape) == 1:
        features = features[:,np.newaxis]
    assert(len(features.shape) == 2)
    num_features = features.shape[-1]
    
    if np.issubdtype(features.dtype, np.number):
        features = _normalize_numerical_features(features, clip = clip, 
            scale_features = scale_features, max_bar_height = max_bar_height, style = style)
    
    if group_names is None:
        if not tree_graph is None:
            group_names = list(np.arange(tree_graph.shape[-1]).astype(str))

    if style == 'heatmap' and tree_structure:
        logger.warn('"tree_structure" is not available in heatmap mode. Set "tree_structure" to False.')

    if tree_graph is None or style == 'heatmap':
        tree_structure = False

    if not tree_structure:
        max_bar_height = 1.0

    if feature_labels is None:
        feature_labels = np.arange(features.shape[-1]).astype(str)
    feature_labels = np.array(feature_labels)

    if log_pseudotime:
        pseudotime = np.log(pseudotime+ 1)
        min_pseudotime = np.log(min_pseudotime + 1)

    if not order is None:
        feature_order = np.argsort(
            pseudotime[features.argmax(0)]
        )

        if order == 'descending':
            feature_order = feature_order[::-1]

        feature_labels = feature_labels[feature_order]
        features = features[:, feature_order]

    segment_kwargs = dict(
        window_size = window_size, center_baseline = center_baseline, hide_feature_threshold = hide_feature_threshold, legend_cols = legend_cols,
        palette = palette, linecolor = linecolor, linewidth = linewidth, hue_order = hue_order, show_legend = show_legend, alpha = alpha,
        color = color, size = size, max_bar_height = max_bar_height, max_swarm_density = max_swarm_density, show_points = show_points,
    )
    segment_kwargs = {k : v for k, v in segment_kwargs.items() if not v is None} #eliminate None values to allow segment functions to fill default values
    
    scaffold_kwargs = dict(linecolor = scaffold_linecolor, linewidth = scaffold_linewidth)

    build_tree_kwargs = dict(pseudotime = pseudotime, cluster_id = group, tree_graph = tree_graph,
            lineage_names = group_names, min_pseudotime = min_pseudotime, max_bar_height = max_bar_height)
    
    if style == 'stream':
        plot_fn = _plot_stream_segment
    elif style == 'scatter':
        plot_fn = _plot_scatter_segment
    elif style == 'swarm':
        plot_fn = _plot_swarm_segment
    elif style == 'heatmap':
        plot_fn = _plot_heatmap_segment

    def make_plot(ax, features):
        
        cell_colors = None
        if style == 'swarm':
            cell_colors = _get_swarm_colors(ax = ax, palette = palette, features = features[:,0],
                show_legend = show_legend, hue_order = hue_order)
            build_tree_kwargs['cell_colors'] = cell_colors

        scaffold_fn = partial(_plot_scaffold, **scaffold_kwargs)
        segment_fn = partial(plot_fn, feature_labels = feature_labels, **segment_kwargs)
        
        plot_bottom = -max_bar_height/2 - 0.3
        if tree_structure:
            _build_tree(**build_tree_kwargs, features = features, ax = ax, plot_fn=scaffold_fn)
            plot_bottom = _build_tree(**build_tree_kwargs, features = features, ax = ax, plot_fn=segment_fn)
            
        else:
            segment_fn(features = features, pseudotime = pseudotime, is_leaf = False, ax = ax, max_bar_height = max_bar_height,
                centerline = 0, lineage_name = '', segment_connection = None, is_root = True, cell_colors = cell_colors)

            ax.set(ylim = (plot_bottom, max_bar_height/2 + 0.15))
        
        if pseudotime_triangle:
            _plot_pseudotime_scale(ax = ax, pseudotime = pseudotime, plot_bottom = plot_bottom)

    fig = None

    if num_features == 1 or (not split and style in ['stream','scatter','heatmap']):
        
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=figsize)

        make_plot(ax, features)

        if not title is None:
            ax.set_title(str(title), fontdict= dict(fontsize = 'x-large'))

    else:
        
        def map_stream(ax, features, label):
            make_plot(ax, features[:, np.newaxis])
            ax.set_title(str(label), fontdict= dict(fontsize = 'x-large'))
            
        fig,ax = map_plot(map_stream, list(zip(features.T, feature_labels)), plots_per_row= plots_per_row, height= height, aspect= aspect,
                        vertical = vertical)
        if not title is None:
            fig.suptitle(title, fontsize = 16)
        
    plt.tight_layout()

    if not fig is None:
        return fig, ax
    else:
        return ax
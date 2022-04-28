
import numpy as np
import matplotlib.pyplot as plt
from mira.plots.base import map_colors
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


def _plot_fill(is_top = False,*, ax, time, fill_top, fill_bottom, color, linecolor, linewidth, alpha = 1., orientation = 'h'):

    if orientation == 'v':
        ax.fill_betweenx(time, fill_top, fill_bottom, color = color, alpha = alpha)

        if not linecolor is None:
            ax.plot(fill_top, time, color = linecolor, linewidth = linewidth)
            if is_top:
                ax.plot(fill_bottom, time, color = linecolor, linewidth = linewidth)

    else:
        ax.fill_between(time, fill_top, fill_bottom, color = color, alpha = alpha)

        if not linecolor is None:
            ax.plot(time, fill_top, color = linecolor, linewidth = linewidth)
            if is_top:
                ax.plot(time, fill_bottom, color = linecolor, linewidth = linewidth)
        



def _plot_stream_segment(is_leaf = True, centerline = 0, window_size = 101,center_baseline = True, is_root = True,
        palette = 'Set3', linecolor = 'black', linewidth = 0.1, feature_labels = None, hide_feature_threshold = 0,
        hue_order = None, show_legend = True, max_bar_height = 0.6, legend_cols = 5, orientation = 'h',
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
        ax = ax, time = ascending_time, orientation = orientation,
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
        palette = 'Set3', linecolor = 'black', linewidth = 0.5, feature_labels = None, 
        hue_order = None, show_legend = True, legend_cols = 5, orientation = 'h',
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

        if orientation == 'v':
            if show_points:
                ax.scatter(
                    centerline + features,
                    ascending_time,
                    color = _color, s = size
                )

            ax.plot(
                smoothed_features + centerline,
                ascending_time,
                color = _color, linewidth = np.sqrt(size), alpha = alpha,
            )
            
            ax.hlines(min_time, xmin = centerline, xmax = centerline + max_bar_height, color = linecolor, linewidth = linewidth)
            ax.hlines(max_time, xmin = centerline, xmax = centerline + max_bar_height, color = linecolor, linewidth = linewidth)
            #ax.hlines(centerline, xmin = min_time, xmax = max_time, color = linecolor, linewidth = linewidth)

            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        else:
            if show_points:
                ax.scatter(
                    ascending_time,
                    centerline + features,
                    color = _color, s = size
                )

            ax.plot(
                ascending_time,
                smoothed_features + centerline,
                color = _color, linewidth = np.sqrt(size), alpha = alpha,
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
        [base, base + 0.1], 
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
        
    
def _build_tree(cell_colors = None, size = None, shape = None, max_bar_height = 0.75,*,
    ax, features, pseudotime, cluster_id, tree_graph, lineage_names, min_pseudotime, plot_fn):

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
            pseudotime_range = segment_pseudotime.max() - segment_pseudotime.min()
            
            connection = ((centerlines[start_clus], max_times[start_clus]), 
                            (centerlines[end_clus], min_times[end_clus]))

            segment_is_leaf = is_leaf(tree_graph, end_clus)

            if segment_is_leaf and pseudotime_range < min_pseudotime:
                segment_pseudotime = (segment_pseudotime - segment_pseudotime.min()) * (min_pseudotime/pseudotime_range) + segment_pseudotime.min()

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


def _normalize_numerical_features(features, enforce_max = None, *, clip, scale_features, max_bar_height, style, split):

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

    if style == 'stream' and not split:
        height_normalizer = (features.sum(-1).max())
    else:
        height_normalizer = features.max(0)

    features = features/height_normalizer * max_bar_height

    if not enforce_max is None:
        features*=(features_max - features_min)/enforce_max

    return features
    

@adi.wraps_functional(pli.fetch_streamplot_data, adi.return_output,
    ['group_names','features','pseudotime','group','tree_graph', 'feature_labels']
)
def plot_stream(style = 'stream', split = False, log_pseudotime = True, scale_features = False, order = None,
    title = None, show_legend = True, legend_cols = 5, max_bar_height = 0.6, size = None, max_swarm_density = 1e5, hide_feature_threshold = 0,
    palette = None, color = 'black', linecolor = 'black', linewidth = None, hue_order = None, pseudotime_triangle = True,
    scaffold_linecolor = 'lightgrey', scaffold_linewidth = 1, min_pseudotime = -1, orientation = 'h',
    figsize = (10,5), ax = None, plots_per_row = 4, height = 4, aspect = 1.3, tree_structure = True,
    center_baseline = True, window_size = 101, clip = 10, alpha = 1., vertical = False, enforce_max = None,
    feature_labels = None, group_names = None, tree_graph = None,*, features, pseudotime, group):
    '''

    Plot a streamgraph representation of a differentiation or continuous process.
    Modifying the parameters produces variations of the streamgraph for making different
    sorts of comparisons. The available modes are:
        
    * stream - 3 to 20 continous features
    * swarm - one discrete feature
    * line and scatter - comparing modalities for one feature
    * heatmap - 20 or more continous features, no lineage tree

    .. note::

        To plot a stream graph, you must first perform lineage inference on the
        data using the :ref:`mira.time </api.rst#Pseudotime>` API.

    
    Parameters
    ----------

    data : list[str] or str
        Which data features of dataframe to plot. If str, plots one feature.
        If list, plots each feature. The feature may be the name of a gene or
        a cell-level attribute in the `.obs` dataFrame. 
    layers : list[str] or str
        Which layer of dataframe to plot for a given attribute. If str, all features
        provided in `data` will be found in the same layer. If list, must provide a
        list where each element is a layer that is the same length as `data`. For 
        features that are in `.obs`, any layer name may be provided. To plot two 
        attributes for the same gene, for example, expression and accessibility,
        list that gene twice in `data`, then specify the two layers to use.
    style : {"stream", "swarm", "heatmap", "line", "scatter"}, default = "stream"
        Style to plot data. The attributes and advantages of each style are outlined 
        in the *Notes* section.
    scale_features : boolean, default = False
        Independently scale each feature to the range [0,1]. Enables comparisons of
        feature trends with different magnitudes.
    split : boolean, defaut = False
        Whether to split each feature into its own plot. By default, stream, scatter,
        and line mode will plot multiple features on the same plot. Setting `split` to
        True will create a separate plot for each feature. This feature is not available
        for heatmaps, and is enforced behavior for swarms.
    order : {"ascending", "descending", None}, default = None
        Ascending order plots features in the order at which they peak in terms of 
        pseudotime, so feature that peak earlier will appear first on the plot. Vice-
        versa for descending order. Setting `order` to None will plot features in the
        order they are provided to `data`.
    window_size : { i | i > 0, i is odd }, default = 101
        Odd integer number. Used for smoothing of data for streams, lines, and 
        scatter plots. Used as the number of cells to aggregate per column
        in heatmap mode. Increasing this parameter will produce smoother plots.
    clip : float > 0, default = 10
        Values of feature *x* are clipped to be within the bounds of mean(x) +/- clip * std(x).
        This trims in outliers and reduces their effect on smoothing. This is useful for 
        noisy data.
    tree_structure : boolean, default = True
        Whether to plot the lineage tree structure of the data. This is disabled
        for heatmap mode. If set to False, this will not required that you have
        conducted lineage inference on the data, only that you have some
        sort of time assigned to each cell.

    Plot Aesthetics

    max_bar_height : float (0, 1), default = 0.6
        The amount of space occupied by the stream/scatter/line/swarm at its maximum
        magnitude. A `max_bar_height` of 1 will fill all available space with no 
        room between lineages.
    size : float > 0 or None, default = None
        Size of dots for swarm or scatter plots. Default of None will use defaults
        from swarmplot and scatterplot sub functions.
    max_swarm_density : float > 0, default = 1e5
        Maximum number of points per pseudotime on swarmplot. Reducing this parameter
        reduces the number of points to draw and speeds up plotting. This parameter may
        also be adjusted to prevent points from overflowing into the gutters of swarm
        segments.
    hide_feature_threshold : float >= 0, < 1, default = 0.
        If a feature comprises less than this fraction of the magnitude of the plot at
        some timepoint, hide that feature. This is useful when plotting streams with 
        many features, many of which are close to zero at any given time. Increasing this 
        parameter above 0. will hide those features and declutter the plot. 
    linewidth : float > 0 or None, default = None
        Width of elements colored by `linecolor`. Default of None differs to 
        style-specific default values.
    scaffold_linewidth : float > 0, default = 1
        Linewidth of scaffold
    pseudotime_triange : boolean, default = True
        Whether to plot the triange marking the pseudotime axis at bottom of plot.

    Pseudotime Options

    pseudotime_key : str, default = 'mira_pseudotime',
        Which key in `.obs` to use for the pseudotime for each cell (x-axis of plot). 
        Sometimes, the pseudotime calculated by the `mira.time` API may be inconvenient for
        plotting because segments of the lineage tree may have unweildy lengths. You
        can use your own pseudotime metric or transformation by specifying which column
        in `.obs` to find it.
    log_pseudotime : boolean, default = True
        Diffusion pseudotime increases exponentially with distance from the root. Log
        pseudotime compresses the upper ranges of pseudotime and typically yields more
        balanced plots.
    min_pseudotime : float > 0, default = 0.05
        This parameter ensures no segment on the lineage tree is shorter in 
        pseudotime than the value provided. If a certain segment of the lineage 
        tree is too short to be visualized, it may be increased.

    Coloring

    palette : str, list[str], or None; default = None
        Palette of plot. Default of None will set `palette` to the style-specific default.
    color : str, default = "black"
        When only plotting one feature, streams, lines, or scatters, are colored by this
        parameter rather than `palette`. This behavior is similar to matplotlib.
    linecolor : str, default = "black"
        Color of edges of plots, including outline of streams, scatters, and swarms.
    scaffold_linecolor : str, default = "lightgrey"
        Color of lineage tree scaffold
    hue_order : list[str] or None, default = None
        Order to assign hues to features provided by `data`. Works similarly to
        hue_order in seaborn. User must provide list of features corresponding to 
        the order of hue assignment. 

    Plot Specifications
    
    title : str or None, default = None
        Title of figure
    show_legend : boolean, default = True
        Show figure legend
    legend_cols : int, default = 5
        Number of columns for horizontal legend.
    figsize : tuple(float, float), default = (7,4)
        Size of figure
    ax : matplotlib.pyplot.axes, deafult = None
        Provide axes object to function to add streamplot to a subplot composition,
        et cetera. If no axes are provided, they are created internally.
    plots_per_row : int > 0, default = 4
        Number of plots per row when in swarm mode or when `split` is True.
    height : flaot > 0, default = 4
        Height of plot when split. Otherwise, function uses `figsize`.
    aspect : float > 0, default 1.3
        Apsect ratio of split plots
    
    Other Parameters
    ----------------
    alpha : float in [0,1], defaut = 1
        Transparency of plot elements.
    vertical : boolean, default = False
        Does not currently do anything.
    tree_graph_key : str, deafult = 'connectivities_tree', 
        Which key in `.uns` to find the connectivities tree between lineage tree segments.
        Contains a np.ndarray of shape (2*n_tree_states - 1, 2*n_tree_states - 1) with 
        elements equal to one at index i,j meaning tree_state j is a descendent of i.
        This is found by `mira.time.get_tree_structure`, but may be manually encoded.
    group_names_key = 'tree_state_names',
        Which key in `.uns` to find the names of the tree states corresponding to columns
        and rows of `tree_graph_key`.
    group_key = 'tree_states',
        Which column in `.obs` to find the cell membership to particular tree states.

    Examples
    --------

    .. note::

        Below, we provide a smattering of examples. For a more in-depth tutorial, 
        see `the streamgraph tutorial <notebooks/tutorial_streamgraphs.ipynb>`_.

    **Plotting topics.** Plot the composition of topics along a differentiation.
    Here `hide_feature_threshold` hides topics which aren't contributing to the
    cell composition. This significantly cleans up the plot.

    .. code-block :: python

        >>> topics = [6,9,10,5,4,22]
        >>> mira.pl.plot_stream(data, data = ["topic_" + str(i) for i in topics], style = "stream",
        ...     hide_feature_threshold = 0.03, window_size = 301, max_bar_height = 0.8, order = 'ascending',
        ...     palette = "Set3", legend_cols = 3, log_pseudotime = False, linewidth=0.3)
        >>> plt.show()

    .. image :: /_static/stream/topic_stream.svg
        :width: 1000
        :align: center

    **Comparing expression and accessibility.** We provide the gene "LEF1" to
    `data` twice, then indicate to MIRA to plot the expression, then accessibility
    of "LEF1". We set `order` to `None` so that the provided palette always matches 
    with the correct mode. We also set `scale_features` to `True` so that we can
    compare trends instead of absolute magnitudes.

    .. code-block :: python

        >>> mira.pl.plot_stream(data, data = ["LEF1","LEF1"], style = "line",
        ...    layers = ["expression","accessibility"], window_size = 301, max_bar_height = 0.8,
        ...    palette = ["red","black"], order = None, scale_features = True, figsize=(5,3),
        ...    clip = 3, log_pseudotime = False)
        >>> plt.show()

    .. image :: /_static/stream/mode_comparison.svg
        :width: 500
        :align: center

    **Plotting cluster membership using swarm mode.** Swarm mode is useful
    for plotting discrete features.

    .. code-block :: python

        >>> mira.pl.plot_stream(data, data = "true_cell", style = "swarm", palette = "Set3",
        ...     max_swarm_density = 100, max_bar_height = 0.8, size = 5, log_pseudotime = False)
        >>> plt.show()

    .. image :: /_static/stream/swarm.png
        :width: 800
        :align: center

    **Visualizing marker genes.** Each gene is plotted on its own stream.

    .. code-block :: python

        >>> mira.pl.plot_stream(data, data = ["LEF1","WNT3","CTSC","LGR5"], style = "stream", 
        ...     color = "black", split = True, clip = 2, scale_features=True,
        ...     log_pseudotime = False, window_size = 301)
        >>> plt.show()

    .. image :: /_static/stream/marker_genes.svg
        :width: 1200
        :align: center
    
    **Using heatmap mode.** Note that heatmap mode does not contain lineage tree information,
    so it is best to subset the tree down to one lineage. You can do this by subsetting
    the input data to only contain cells along the path you want to see.

    Below, the boolean mask `adata.obs.tree_states.str.contains("Cortex")` selects for
    cells whose `tree_state` attribute indicates that cell is upstream of the cortex 
    lineage. 

    .. code-block :: python

        >>> mira.pl.plot_stream(data[data.obs.tree_states.str.contains("Cortex")], 
        ...     data = ["LGR5","EDAR","LEF1","WNT3"], style = "heatmap", order = None,
        ...     window_size = 101, scale_features=True, tree_structure = False, figsize=(7,3),
        ...     log_pseudotime = False)
        >>> plt.show()

    .. image :: /_static/stream/heatmap.svg
        :width: 600
        :align: center

    You can subset cells using more complicated filters. For example, to include only
    cells which may differentiate into **Cortex** or **Medulla** cells:

    .. code-block :: python

        >>> mira.pl.plot_stream(data[data.obs.tree_states.str.contains("Cortex|Medulla")],
        ...     data = ["DSG4","SOAT1","LEF1"], style = "stream", window_size = 301,
        ...     scale_features = True, palette='Set2', linewidth=0.5, clip = 1,
        ...     max_bar_height = 0.99)
        >>> plt.show()

    .. image :: /_static/stream/cell_selection.svg
        :width: 800
        :align: center

    Finally, you can create plots without lineage structure by setting *tree_structure*
    to *False*. This creates a more traditional 2-dimensional plot, showing, for example,
    the levels of *Lgr5* and *Lef1* along the path from **ORS** to **Cortex** cells:

    .. code-block :: python

        >>> mira.pl.plot_stream(data[data.obs.tree_states.str.contains('Cortex')], 
        ...     data = ['LEF1','LGR5'], log_pseudotime=False, title = 'Gene Counts',
        ...     style = 'scatter', window_size = 301, tree_structure=False,
        ...     palette=['black','red'], max_bar_height=0.99, size = 3)
        >>> plt.show()

    .. image :: /_static/stream/scatterplot.svg
        :width: 800
        :align: center

    '''

    assert(isinstance(max_bar_height, float) and max_bar_height > 0 and max_bar_height <= 1)
    assert(isinstance(features, np.ndarray))
    assert(style in ['line','scatter','stream','swarm', 'heatmap'])
    if orientation == 'v':
        pseudotime_triangle = False
        assert(not tree_structure)

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
        features = _normalize_numerical_features(features, clip = clip, split = split, enforce_max=enforce_max,
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

    if not order is None:
        
        pseudotime_order = np.argsort(pseudotime)

        first_over_50_idx = (features[pseudotime_order].cumsum(0)/features.sum(0) > 0.5).argmax(0)
        feature_order = np.argsort(pseudotime[pseudotime_order][first_over_50_idx])

        if order == 'descending':
            feature_order = feature_order[::-1]

        feature_labels = feature_labels[feature_order]
        features = features[:, feature_order]

    segment_kwargs = dict(
        window_size = window_size, center_baseline = center_baseline, hide_feature_threshold = hide_feature_threshold, legend_cols = legend_cols,
        palette = palette, linecolor = linecolor, linewidth = linewidth, hue_order = hue_order, show_legend = show_legend, alpha = alpha,
        color = color, size = size, max_bar_height = max_bar_height, max_swarm_density = max_swarm_density, show_points = show_points, 
        orientation = orientation,
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

            if orientation == 'h':
                ax.set(ylim = (plot_bottom, max_bar_height/2 + 0.15))
            else:
                ax.set(xlim = (-max_bar_height/2, max_bar_height/2))
                #ax.set(ylim = (pseudotime.min() - 0.05, pseudotime.max() + 0.05))
                ax.invert_yaxis()
        
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

if __name__ == "__main__":
    import doctest
    doctest.testmod()
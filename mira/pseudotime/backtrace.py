from mira.pseudotime.pseudotime import prune_edges, make_markov_matrix, \
    get_adaptive_affinity_matrix, get_kernel_width

from mira.plots.base import plot_umap, map_colors, map_plot
import numpy as np
import mira.adata_interface.pseudotime as pti
import mira.adata_interface.core as adi
from functools import partial
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def _make_map(ka = 5,*, distance_matrix, pseudotime,):

    kernel_width = get_kernel_width(ka, distance_matrix)

    distance_matrix = prune_edges(kernel_width = kernel_width, 
        pseudotime = pseudotime, distance_matrix = distance_matrix)

    affinity_matrix = get_adaptive_affinity_matrix(kernel_width = kernel_width,
        distance_matrix = distance_matrix)

    transport_map = make_markov_matrix(affinity_matrix)

    return transport_map


def _trace(num_steps = 5000,*, transport_map, start_cells):

    start_cells = start_cells.astype(bool).astype(int).reshape((-1,1))
    start_cells = start_cells/start_cells.sum()

    current_step = start_cells.copy()
    steps = []

    for i in range(num_steps):
        current_step = transport_map.T.dot(current_step)
        steps.append(current_step.copy())

    return np.hstack(steps).T


def _animate(frame_num, palette = 'BuPu', vmax_quantile = 0.99,*, 
        scatter, backtrace, original_ordering, basis):
    
    probs = backtrace[frame_num]
    vmax = np.quantile(probs, vmax_quantile)
    
    order = probs.argsort()
    
    probs = probs[order]
    scatter.set_offsets(basis[order])
    
    color = map_colors(None, probs, vmax = vmax, add_legend=False, 
            palette = palette)

    scatter.set_color(color)

    return scatter

@adi.wraps_functional(pti.fetch_trace_args, adi.return_output, 
    ['basis','start_cells','distance_matrix','pseudotime','transport_map'])
def trace_differentiation(
    palette = 'BuPu', add_outline = True, outline_width = (0,12), 
    outline_color = 'lightgrey',
    size = 1, figsize = (10,7), fps = 24, steps_per_frame = 1,
    num_steps = 4000, ka = 5, vmax_quantile = 0.99, 
    direction = 'forward', num_preview_frames = 4, 
    log_prob = False, sqrt_time = False, 
    distance_matrix = None, transport_map = None, pseudotime = None, *,
    basis, start_cells, save_name,
    **plot_args,
):
    '''
    
    Starting from a group of initial cells, trace the diffusion
    over time through the markov chain model of differentiation.

    In "forward" mode, traces the cells along paths from less
    to more differentiated states and elucides paths to different
    terminal states.

    In "backward" mode, start from a terminal state and find ancestor
    populations of cells.

    Parameters
    ----------

    adata : anndata.AnnData
        Anndata with pseudotime defined.
    save_name : str
        Filepath to write .gif file of trace.
    direction : {"forward","backward"}, default = "forward"
        Which direction to run the diffusion process. Forward simulates the 
        differentiation process and traces descendent cell populations. Backwards
        finds ancestral cell populations.
    palette : str or None, default = "BuPu"
        Palette of plot. Default of None will set `palette` to the style-specific default.
    fps : int > 0, default = 24
        Frames-per-second of movie. 
    setps_per_frame : int > 0, default = 1
        Number of steps to take on the transport map for each frame. Larger values will
        give a more "jumpy" trace, but will reduce the amount of frames to plot.
    num_steps : int > 0, default = 40000
        Number of steps to take through transport map. This must be set high
        enough to show the desired diffusion process, but may be trimmed to
        the appropriate length for your data.
    ka : int > 0, default = 5
        The ka-th neighbor defines the neighborhood size when creating the transport
        map. A larger neighborhood size creates a transport map that is more tolerant
        of backwards steps.
    basis : str, default = "X_umap"
        

    Other Parameters
    ----------------

    figsize : tuple(int, int), default = (10,7)
        Size of plot
    add_outline : bool, default = True
        Add outline around embedding plots of cells. Shows boundaries of cells with
        low color values more clearly.
    outline_width : tuple(int, int), default = (0,12)
        Width of the outline, if `add_outline` is True. The first value in the tuple
        is the additional size around the data point of a white outline. The second
        value is the additional size around the data point of a colored outline.
    outline_color : str, default = "lightgrey"
        Color of the outline, if `add_outline` is True.
    size : int, default = 1
        Size of datapoints on embedding plot.
    num_preview_frames : int > 0, default = 4
        Number of frames to preview.
    vmax_quantile : float > 0, < 1, default = 0.99
        At each frame, the color scale is capped at this quantile of probabilities
        in the data. Adjusting this down will show lower-probability cells in each
        frame.
    sqrt_time : bool, default = False
        Take frames at steps linearly increasing in sqrt space. This makes
        smaller jumps at lower step counts than at higher stepcounts. If your
        are tracing a process that evolves more quaickly at the onset, set this
        to True.
    log_prob : bool, default = False
        Plot log-probabilities instead of probabilities. If frames are dominated by
        high-probability cells, this may reveal other populations.

    Returns
    -------

    None, but saves a gif of the diffusion process to *save_name*.


    Examples
    --------

    .. code-block:: python
    
        >>> mira.time.trace_differentiation(data, start_lineage='IRS', num_steps=1500,
        ...          save_name='data/hf_diff.gif', direction='backward', sqrt_time=True,
        ...          log_prob=True, steps_per_frame=15, figsize=(7,4))


    .. image:: /_static/pseudotime/mira.time.trace_differentiation.gif
        :width: 600

    '''

    logger.info('Creating transport map ...')

    if transport_map is None:

        if direction == 'backward':
            pseudotime = pseudotime.max() - pseudotime

        transport_map = _make_map(ka = ka, 
            distance_matrix= distance_matrix,
            pseudotime=pseudotime)

    logger.info('Tracing ancestral populations ...')

    num_frames = num_steps//steps_per_frame
    backtrace = _trace(num_steps = num_steps, 
        transport_map= transport_map, 
        start_cells= start_cells)

    if log_prob:
        backtrace = np.log(backtrace + 1e-4)

    # select the frames to use
    if not sqrt_time:
        frame_slices = np.arange(0, num_steps, steps_per_frame)
    else:
        frame_slices = np.square(
            np.linspace(0, np.sqrt(num_steps-1), num_steps//steps_per_frame)
        ).astype(int)
    
    num_frames = len(frame_slices)
    backtrace = backtrace[frame_slices, :]

    plot_kwargs = dict(
        palette = palette, add_outline=add_outline, 
        add_legend=False, size = size, outline_width=outline_width, 
        outline_color = outline_color,
        **plot_args)

    if num_preview_frames > 0:
        
        num_partitions = num_preview_frames - 1
        preview_interval = num_frames//num_partitions
        test_frames = [1] +  list(np.arange(1, num_partitions)*preview_interval)\
            + [num_frames-1]

        map_plot(lambda ax, framenum, x : plot_umap(basis, x, **plot_kwargs, 
            ax = ax, vmax = np.quantile(x, vmax_quantile), title = 'Frame ' + str(framenum)), 
            [(frame, backtrace[frame].reshape(-1)) for frame in test_frames], plots_per_row = 4,
            height = 2.5, aspect = 1.5)
        plt.show()

    fig, ax = plt.subplots(1,1,figsize=figsize)
    
    ax, scatter, order = plot_umap(basis, backtrace[0], ax = ax, animate = True,
        **plot_kwargs)

    plot_fn = partial(_animate, 
        palette = palette, vmax_quantile = vmax_quantile,
        scatter = scatter, backtrace = backtrace, 
        original_ordering = order, basis = basis,
    )

    logger.info('Creating animation ...')
    anim = FuncAnimation(fig, plot_fn, frames=num_frames, interval=1000/fps)
    plt.close()

    logger.info('Saving animation ...')
    writergif = PillowWriter(fps=fps) 
    anim.save(save_name, writer=writergif)
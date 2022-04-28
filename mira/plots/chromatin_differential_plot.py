
import matplotlib.pyplot as plt
from mira.plots.base import map_colors, plot_umap
import numpy as np
import mira.adata_interface.core as adi
import mira.adata_interface.plots as pli

def _plot_chromatin_differential_scatter(ax, 
        title = 'NITE vs LITE Predictions',
        hue_label = 'Expression',
        size = 5,
        palette = 'Reds',
        show_legend = True,
        plot_kwargs = {},
        linecolor = 'grey',
        hue_order = None,
        na_color = 'lightgrey',
        vmin = None, vmax = None,
        *,
        hue, 
        nite_prediction,
        lite_prediction,
    ):


    plot_order = hue.argsort()
    ax.scatter(
        lite_prediction[plot_order],
        nite_prediction[plot_order],
        s = size,
        c = map_colors(
            ax, hue[plot_order], palette = palette, add_legend = show_legend, 
            vmin = vmin, vmax = vmax, na_color= na_color, hue_order = hue_order,
            cbar_kwargs = dict(orientation = 'vertical', pad = 0.01, shrink = 0.5, aspect = 15, anchor = (1.05, 0.5), label = hue_label),
            legend_kwargs = dict(loc="center left", markerscale = 4, frameon = False, title_fontsize='x-large', fontsize='large',
                        bbox_to_anchor=(1.05, 0.5), title = hue_label)
        ),
        **plot_kwargs,
    )
    ax.set(
        title = title,
        xscale = 'log', yscale = 'log',
        xlabel = 'LITE Prediction',
        ylabel = 'NITE Prediction',
        xticks = [], yticks = [],
    )
    
    line_extent = max(lite_prediction.max(), nite_prediction.max()) * 1.2
    line_min = min(lite_prediction.min(), nite_prediction.min()) * 0.8

    ax.set(ylim = (line_min, line_extent), xlim = (line_min, line_extent))
    
    ax.plot([0, line_extent], [0, line_extent], color = linecolor)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_aspect('equal', adjustable='box')


def _plot_chromatin_differential_panel(
    ax, 
    expr_pallete = 'Reds',
    lite_prediction_palette = 'viridis',
    differential_palette = 'coolwarm',
    size = 1.5, differential_range = 3,
    trim_lite_prediction = 5,
    add_outline = True,
    outline_width = (0, 10),
    outline_color = 'lightgrey',
    show_legend = True, 
    first_plot = False,*,
    gene_name,
    umap,
    chromatin_differential, 
    expression,
    lite_prediction,
    nite_prediction
):

    ax[0].text(0.5, 0.5, gene_name,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax[0].transAxes, fontsize='x-large')
    ax[0].axis('off')

    if first_plot:
        ax[0].set(title = 'Gene')

    plot_umap(umap, chromatin_differential, ax = ax[3], palette = differential_palette, add_legend = show_legend,
    size = size, vmin = -differential_range, vmax = differential_range, title = 'Chromatin Differential' if first_plot else '')

    plot_umap(umap, expression, palette = expr_pallete, ax = ax[1], add_legend = show_legend,
        size = size, title = 'Expression' if first_plot else '', 
        add_outline = add_outline, outline_width = outline_width, outline_color = outline_color)


    lite_std = np.nanstd(lite_prediction)
    lite_mean = np.nanmean(lite_prediction)

    plot_umap(umap, lite_prediction, palette = lite_prediction_palette, ax = ax[2], 
        vmin = None, vmax = min(lite_mean + trim_lite_prediction*lite_std, lite_prediction.max()),
        size = size, title = 'Local Prediction' if first_plot else '', add_legend = show_legend)

    _plot_chromatin_differential_scatter(ax[4], 
            title = 'NITE vs. LITE Predictions' if first_plot else '',
            plot_kwargs= dict(
                edgecolor = 'lightgrey',
                linewidths = 0.15,
            ),
            hue = expression,
            palette = expr_pallete,
            nite_prediction = nite_prediction,
            lite_prediction = lite_prediction,
        )
    
    plt.tight_layout()
    return ax


@adi.wraps_functional(
    pli.fetch_scatter_differential_plot, adi.return_output,
    ['hue','hue_label','lite_prediction', 'nite_prediction']
)
def plot_scatter_chromatin_differential(
        ax = None, title = '', size = 3, linecolor = 'grey',
        palette = 'viridis', show_legend = True,
        hue_order = None, na_color = 'lightgrey',
        vmax = None, vmin = None, figsize=(5,5), plot_kwargs = {},*,
        hue, hue_label, lite_prediction, nite_prediction,
        ):
    '''
    Plots chromatin differential scatterplot with more
    flexibility for coloring cells. Useful for studying temporal
    and cell-type relationships between LITE and NITE facets of
    gene expression.
    
    Parameters
    ----------

    adata : anndata.AnnData
        AnnData object, `LITE_prediction` and `NITE_prediction` layers.
    gene : str
        Gene for which to plot LITE and NITE predictions.
    color : str, default = None
        With which column to color cells on plot. If none provided,
        colors by *gene*'s values for *layer*.
    layer : str, default = None
        Which layer to access for *color*.
    plot_kwargs : dict[str, any]
        Dictionary of keyword arguments to pass to backend 
        matplotlib.pyplot.scatter function.

    Examples
    --------

    .. code-block :: python

        >>> mira.pl.plot_scatter_chromatin_differential(
        ...     data, gene='KRT23', color='Cell Type',
        ...     palette='viridis', title = 'KRT23 LITE/NITE predictions')

    .. image :: /_static/mira.pl.plot_scatter_chromatin_differential.png
    
    '''

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)


    _plot_chromatin_differential_scatter(ax, 
        plot_kwargs=plot_kwargs,
        title = title,
        hue_label = hue_label,
        size = size,
        palette = palette,
        show_legend = show_legend,
        hue_order= hue_order,
        na_color= na_color,
        vmin = vmin, vmax = vmax,
        linecolor = linecolor,
        hue = hue, 
        nite_prediction = nite_prediction,
        lite_prediction = lite_prediction,
    )

    return ax


@adi.wraps_functional(
    pli.fetch_differential_plot, adi.return_output,
    ['gene_names','umap','chromatin_differential','expression','lite_prediction', 'nite_prediction']
)
def plot_chromatin_differential(
    expr_pallete = 'Reds', 
    lite_prediction_palette = 'viridis',
    differential_palette = 'coolwarm',
    height = 3,
    aspect = 1.5, 
    differential_range = 3,
    trim_lite_prediction = 5,
    show_legend = True,
    size = 1, *,
    gene_names,
    umap,
    chromatin_differential, 
    expression,
    lite_prediction,
    nite_prediction
):
    '''
    Plot the expression, local accessibility prediction, chromatin differential, 
    and LITE vs. NITE predictions for a given gene. This is the main tool with
    which one can visually investigate gene regulatory dynamics. These plots
    are most informative when looking at NITE-regulated genes.

    .. note::
        
        *Before using this function, one must train RP models.*
        Please refer to the :ref:`LITE/NITE tutorial </notebooks/tutorial_topic_model_tuning.ipynb>`
        for instruction on training RP models and calculating NITE scores and chromatin differential.

    Parameters
    ----------
    
    adata : anndata.AnnData
        AnnData object with `chromatin_differential`, `LITE_prediction`,
        and `NITE_prediction` layers.
    gene_names : list[str], np.ndarray[str]
        List of genes for which to plot chromatin differential panels.
    expr_pallete : str, default = 'Reds'
        Pallete for plotting expression values.
    lite_prediction_palette : str, default = 'viridis'
        Palette for plotting LITE prediction values.
    differential_palette : str, default = 'coolwarm'
        Palette for plotting chromatin differential.
    height : float, default = 3
        Height of plot panels
    aspect : float, default = 1.5
        Aspect ratio of plots
    differential_range : float, default = 3
        Clamps range of color values for chromatin differential to
        +/- differential range.
    trim_lite_prediction : float, default = 5
        Clips the maximum LITE prediction value to *mean + <time_lite_prediction> std*,
        reducing the effect outliers have on plot colors.
    show_legend : boolean, default = True
        Show legend on plots.
    size : float, default = 1
        Size of points.

    Returns
    -------

    matplotlib.pyplot.axes

    Examples
    --------

    .. code-block :: python

        >>> mira.pl.plot_chromatin_differential(adata, gene_names = ['LEF1','KRT23','WNT3','MT2'],
        ...         show_legend = False)

    .. image :: /_static/mira.pl.plot_chromatin_differential.png
        :width: 1200

    '''

    num_rows = len(gene_names)
    fig, ax = plt.subplots(num_rows, 5, figsize = ( aspect * height * 4.25, num_rows * height) ,
        gridspec_kw={'width_ratios' : [0.5,2,2,2,2]})

    if num_rows == 1:
        ax = ax[np.newaxis , :]

    for i, data in enumerate(zip(
        gene_names,
        chromatin_differential.T,
        expression.T,
        lite_prediction.T,
        nite_prediction.T,
    )):

        kwargs = dict(zip(
            ['gene_name','chromatin_differential','expression','lite_prediction','nite_prediction'],
            data
        ))

        _plot_chromatin_differential_panel(ax = ax[i,:], umap = umap, expr_pallete = expr_pallete, lite_prediction_palette = lite_prediction_palette,
            size = size, differential_palette = differential_palette, show_legend = show_legend, trim_lite_prediction = trim_lite_prediction,
            differential_range = differential_range, first_plot = i == 0,
            **kwargs)

    return ax
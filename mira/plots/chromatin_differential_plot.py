
import matplotlib.pyplot as plt
from mira.plots.base import map_colors, plot_umap
import numpy as np
import mira.adata_interface.core as adi
import mira.adata_interface.plots as pli
from matplotlib.patches import Patch
import warnings

def _plot_chromatin_differential_scatter(ax, 
        title = 'LITE vs NITE Predictions',
        hue_label = 'Expression',
        size = 5,
        palette = 'Reds',
        add_legend = True,
        *,
        hue, 
        nite_prediction,
        lite_prediction,
    ):
    
    plot_order = hue.argsort()
    ax.scatter(
        nite_prediction[plot_order],
        lite_prediction[plot_order],
        s = size,
        c = map_colors(
            ax, hue[plot_order], palette = palette, add_legend = add_legend,
            cbar_kwargs = dict(
                location = 'right', pad = 0.1, shrink = 0.5, aspect = 15, label = hue_label,
            )
        ),
        edgecolor = 'lightgrey',
        linewidths = 0.15,
    )
    ax.set(
        title = title,
        xscale = 'log', yscale = 'log',
        xlabel = 'NITE Prediction',
        ylabel = 'LITE Prediction',
        xticks = [], yticks = [],
    )
    
    line_extent = max(lite_prediction.max(), nite_prediction.max()) * 1.2
    line_min = min(lite_prediction.min(), nite_prediction.min()) * 0.8
    
    '''ax[3].fill_between([line_min, line_extent],[line_min, line_extent], color = 'royalblue', alpha = 0.025)
    ax[3].fill_between([line_min, line_extent],[line_extent, line_extent],[line_min, line_extent], color = 'red', alpha = 0.025)

    ax[3].legend(handles = [
                Patch(color = 'red', label = 'Over-estimates', alpha = 0.5),
                Patch(color = 'cornflowerblue', label = 'Under-estimates', alpha = 0.5),
            ], **dict(
                loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon = False, ncol = 2, 
            ))'''

    ax.set(ylim = (line_min, line_extent), xlim = (line_min, line_extent))
    
    ax.plot([0, line_extent], [0, line_extent], color = 'grey')
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
    add_legend = True, *,
    gene_name,
    umap,
    chromatin_differential, 
    expression,
    lite_prediction,
    nite_prediction
):

    plot_umap(umap, chromatin_differential, ax = ax[2], palette = differential_palette, add_legend = add_legend,
    size = size, vmin = -differential_range, vmax = differential_range, title = gene_name + ' Chromatin Differential')

    plot_umap(umap, expression, palette = expr_pallete, ax = ax[0], add_legend = add_legend,
        size = size, title = gene_name + ' Expression', 
        add_outline = add_outline, outline_width = outline_width, outline_color = outline_color)


    lite_std = np.nanstd(lite_prediction)
    lite_mean = np.nanmean(lite_prediction)

    plot_umap(umap, lite_prediction, palette = lite_prediction_palette, ax = ax[1], 
        vmin = None, vmax = min(lite_mean + trim_lite_prediction*lite_std, lite_prediction.max()),
        size = size, title = gene_name + ' Local Prediction', add_legend = add_legend)

    _plot_chromatin_differential_scatter(ax[3], 
            title = gene_name + ' LITE vs. NITE Predictions',
            hue = expression,
            palette = expr_pallete,
            nite_prediction = nite_prediction,
            lite_prediction = lite_prediction,
        )
    
    plt.tight_layout()
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
    add_legend = True,
    size = 1, *,
    gene_names,
    umap,
    chromatin_differential, 
    expression,
    lite_prediction,
    nite_prediction
):

    num_rows = len(gene_names)
    fig, ax = plt.subplots(num_rows, 4, figsize = ( aspect * height * 4, num_rows * height) )

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
            size = size, differential_palette = differential_palette, add_legend = add_legend, trim_lite_prediction = trim_lite_prediction,
            differential_range = differential_range, 
            **kwargs)

    return ax
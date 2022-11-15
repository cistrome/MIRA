
from mira.plots.base import map_colors
import matplotlib.pyplot as plt
from mira.adata_interface.core import wraps_functional
from mira.adata_interface.plots import fetch_disentanglement_plot

@wraps_functional(fetch = fetch_disentanglement_plot,
    fill_kwargs = ['expression_rates','technical_effects','hue_values'])
def plot_disentanglement(
    expression_rates, technical_effects, hue_values,
    ax = None, 
    figsize = (5,5),
    palette = 'Greys', 
    size = 0.5, 
    add_legend = False, 
    hue_order = None, 
    vmin = None, 
    vmax = None,
    na_color = 'lightgrey',
    hue_label = '',
    **plot_kwargs):
    '''
    
    '''

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)


    ax.scatter(
        technical_effects,
        expression_rates,
        s = size,
        c = map_colors(
            ax, hue_values, palette = palette, add_legend = add_legend, 
            vmin = vmin, vmax = vmax, na_color= na_color, hue_order = hue_order,
            cbar_kwargs = dict(orientation = 'vertical', pad = 0.01, shrink = 0.5, aspect = 15, anchor = (1.05, 0.5), label = hue_label),
            legend_kwargs = dict(loc="center left", markerscale = 4, frameon = False, title_fontsize='x-large', fontsize='large',
                        bbox_to_anchor=(1.05, 0.5), title = hue_label)
        ),
        **plot_kwargs,
    )
    
    ax.set(yscale = 'log', xlabel = 'Technical effects', ylabel = 'Expression rates',
        yticks = [], xticks = [0])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax

import warnings
import numpy as np
from matplotlib.colors import Normalize, ColorConverter
from matplotlib import cm
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from math import ceil


def map_plot(func, data, plots_per_row = 3, height =4, aspect = 1.5, vertical = False):

    num_plots = len(data)

    num_rows = ceil(num_plots/plots_per_row)
    plots_per_row = min(plots_per_row, num_plots)

    fig, ax = plt.subplots(num_rows, plots_per_row, figsize = (height*aspect*plots_per_row, height*num_rows))
    if num_plots == 1:
        ax = np.array([[ax]])
    elif num_rows==1:
        ax = ax[np.newaxis, :]

    if vertical:
        ax = ax.T

    for ax_i, d in zip(ax.ravel(), data):
        
        func(ax_i, *d)

    for ax_i in ax.ravel()[num_plots:]:
        ax_i.axis('off')

    plt.tight_layout()

    return fig, ax


def map_colors(ax, c, palette, add_legend = True, hue_order = None, na_color = 'lightgrey',
        legend_kwargs = {}, cbar_kwargs = {}, vmin = None, vmax = None, log = False,
        normalizer = Normalize):

    assert(isinstance(c, (np.ndarray, list)))
    
    if isinstance(c, list):
        c = np.array(c)
    c = np.ravel(c)

    if log:
        c = np.log1p(c)

    if np.issubdtype(c.dtype, np.number):
        
        na_mask = np.isnan(c)

        colormapper=cm.ScalarMappable(normalizer(
            np.nanmin(c) if vmin is None else vmin,
            np.nanmax(c) if vmax is None else vmax), 
            cmap=palette)
        c = colormapper.to_rgba(c)

        if na_mask.sum() > 0:
            c[na_mask] = ColorConverter().to_rgba(na_color)

        if add_legend:
            plt.colorbar(colormapper, ax=ax, **cbar_kwargs)

        return c

    else:
        na_mask = c == 'nan'
        
        classes = list(
            dict(zip(c, range(len(c)))).keys()
        )[::-1] #set, order preserved

        if isinstance(palette, list):
            num_colors = len(palette)
            palette_obj = lambda i : np.array(palette)[i]
        else:
            palette_obj = cm.get_cmap(palette)
            num_colors = len(palette_obj.colors)

        if num_colors > 24:
            color_scaler = (num_colors-1)/(len(classes)-1)

            color_wheel = palette_obj(
                (color_scaler * np.arange(len(classes))).astype(int) % num_colors
            )
        else:
            color_wheel =palette_obj(np.arange(len(classes)) % num_colors)
        
        if hue_order is None:
            class_colors = dict(zip(classes, color_wheel))
        else:
            assert(len(hue_order) == len(classes))
            class_colors = dict(zip(hue_order, color_wheel))

        c = np.array([class_colors[c_class] for c_class in c])
        
        if na_mask.sum() > 0:
            c[na_mask] = ColorConverter().to_rgba(na_color)
        
        if add_legend:
            ax.legend(handles = [
                Patch(color = color, label = str(c_class)) for c_class, color in class_colors.items() if not c_class == 'nan'
            ], **legend_kwargs)

        return c


def plot_umap(X, hue, palette = 'viridis', projection = '2d', ax = None, figsize= (10,5),
        add_legend = True, hue_order = None, size = 2, title = None, vmin = None, vmax = None, 
        add_outline = False, outline_color = 'lightgrey', outline_width = (0, 0.5),
        **plot_kwargs):
    
    plot_order = hue.argsort()

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)

    if isinstance(size, (list, np.ndarray)):
        size = size[plot_order]

    colors = map_colors(ax, hue[plot_order], palette, add_legend=add_legend, hue_order = hue_order, vmin = vmin, vmax = vmax,
            cbar_kwargs = dict(orientation = 'vertical', pad = 0.01, shrink = 0.5, aspect = 15, anchor = (1.05, 0.5)),
            legend_kwargs = dict(loc="center left", markerscale = 4, frameon = False, title_fontsize='x-large', fontsize='large',
                        bbox_to_anchor=(1.05, 0.5)))

    if add_outline:
        assert(isinstance(outline_width, (tuple, list)))
        assert(len(outline_width) == 2)
        assert(outline_width[0] >= 0 and outline_width[1] >= 0)
        
        first_ring_size = size + size*outline_width[0]
        second_ring_size = first_ring_size + size*outline_width[1]

        ax.scatter(X[plot_order,0], X[plot_order,1], color = outline_color, s= second_ring_size, **plot_kwargs)
        ax.scatter(X[plot_order,0], X[plot_order,1], color = 'white', s= first_ring_size, **plot_kwargs)

    ax.scatter(X[plot_order,0], X[plot_order,1], c = colors, s= size, **plot_kwargs)
    ax.axis('off')

    if not title is None:
        ax.set_title(str(title), fontdict= dict(fontsize = 'x-large'))

    return ax
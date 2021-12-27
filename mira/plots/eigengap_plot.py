import matplotlib.pyplot as plt
import mira.adata_interface.core as adi
import mira.adata_interface.pseudotime as pti
import numpy as np
from mira.plots.base import map_plot, plot_umap

@adi.wraps_functional(pti.fetch_eigengap, adi.return_output,
    ['eigvals','eigen_gap','diffmap','umap'])
def plot_eigengap(
    height = 2, aspect = 1.5, size = 0.3,
    eigengap_figsize=(7,4), palette = 'plasma',
    plots_per_row = 5,*,
    eigvals,
    eigen_gap,
    diffmap,
    umap,
):

    fig1,ax = plt.subplots(2,1,figsize=(7,4), sharex = True, 
                        gridspec_kw={'height_ratios' : [3,2]})
    ax[0].plot(range(len(eigvals)-1), eigvals[1:], '--bo', c = 'black')
    ax[0].set(ylabel = 'Eigenvalues')
    ax[1].bar(range(len(eigen_gap)), eigen_gap, color = 'lightgrey', edgecolor = 'black', linewidth = 0.5)
    ax[1].set(ylabel = 'Eigen Gap', xlabel = 'Num Components')
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    plt.tight_layout()

    fig2, ax2 = map_plot(lambda ax, i, component : \
                        plot_umap(umap, np.array(component), palette=palette, size = size, 
                                add_legend=False, ax = ax, title='Component ' + str(i+1)), 
            list(enumerate(diffmap.T)), height = height, plots_per_row = plots_per_row,
            )

    return fig1, fig2, ax, ax2
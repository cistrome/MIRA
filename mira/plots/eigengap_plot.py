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
    '''
    Plots the eigengap, the difference between consecutive eigenvalues, 
    for estimation of the optimal number of diffusion components to 
    represent dataset. Also plots diffusion components projected onto
    the UMAP basis. This allows the user to check to make sure that each
    cell population is captured by the components.

    Parameters
    ----------

    adata : anndata.AnnData
        Adata with diffusion map in `.obsm["X_diffmap"]` and 
        eigenvalues in `.uns["diffmap_evals"]`
    basis : str, default = "X_umap"
        Representation basis for arranging cells. By default,
        uses the representation in `.obsm["X_umap"]`. Only the first two
        components will be used of any representation.
    height : float > 0, default = 2 
        Height of each UMAP plot
    aspect : float > 0, default = 1.5
        Apsect ratio of each UMAP plot
    size : float > 0, default = 0.3
        Size of points on UMAP
    eigengap_figsize : tuple(float, float), default = (7,4)
        Size of the eigengap figure
    palette : str, default = "plasma"
        Matplotlib colormap string for projecting diffusion components 
        onto UMAP plots.
    plots_per_row : int > 0, default = 5
        Number of UMAP plots per row

    Examples
    --------

    Following calculation and normalization of the diffusion map representation
    of the data using `sc.tl.diffmap` and :ref:`mira.time.normalize_diffmap`,
    respectively, one may use this function to estimate the number of diffusion
    components to use to calculate a new KNN graph based on diffusion distances.

    This new KNN graph represents a de-noised version of the joint-KNN graph
    defined from topic compositions.

    The estimate of the number of components is given by the component which
    has the largest *Eigengap*, of difference between successive components. For
    the hair follicle system, this was **5**. 

    .. code-block:: python

        >>> mira.pl.plot_eigengap(data, palette='magma')
        >>> plt.show()

    .. image:: /_static/pseudotime/mira.pl.plot_eigengap.png
        :width: 1400

    .. warning::

        The Eigengap heuristic may not always recommend the best number of components.
        For example, it may recommend fewer components than is needed to describe some
        subtypes of cells. In this case, select the fewest number of components that has a
        large Eigengap **and** includes components that describe all cell types in
        your system (as given by components projected to the UMAP basis).
    
    '''

    fig1,ax = plt.subplots(2,1,figsize=(7,4), sharex = True, 
                        gridspec_kw={'height_ratios' : [3,2]})
    ax[0].plot(range(len(eigvals)-1), eigvals[1:], '--o', c = 'black')
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
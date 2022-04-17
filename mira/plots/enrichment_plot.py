from itertools import zip_longest
import numpy as np
from mira.plots.base import map_colors, map_plot
from functools import partial

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def compact_string(x, max_wordlen = 4, join_spacer = ' ', sep = ' ', label_genes = []):
    return '\n'.join(
        [
            join_spacer.join([x + ('*' if x in label_genes else '') for x in segment if not x == '']) for segment in grouper(x.split(sep), max_wordlen, fillvalue='')
        ]
    )

def _plot_enrichment(ax, ontology, results, 
    label_genes = [], color_by_adj = True, palette = 'Reds', gene_fontsize=10, pval_threshold = 1e-5,
    show_top = 5, show_genes = True, max_genes = 20, text_color = 'black', barcolor = 'lightgrey'):

    assert(isinstance(pval_threshold, float) and pval_threshold > 0 and pval_threshold < 1)

    terms, genes, pvals, adj_pvals = [],[],[],[]
    for result in results[:show_top]:
        
        terms.append(
            compact_string(result['term'])
        )        
        genes.append(' '.join(result['genes'][:max_genes]))
        pvals.append(-np.log10(result['pvalue']))
        adj_pvals.append(-np.log10(result['adj_pvalue']))

    if color_by_adj:
        edgecolor = map_colors(ax, np.array(adj_pvals), palette, add_legend = True, 
            cbar_kwargs = dict(
                    location = 'right', pad = 0.1, shrink = 0.5, aspect = 15, label = '-log10 Adj P-value',
                ), vmin = 0, vmax = -np.log10(pval_threshold))
            
        ax.barh(np.arange(len(terms)), pvals, edgecolor = edgecolor, color = barcolor, linewidth = 2)
    else:
        ax.barh(np.arange(len(terms)), pvals, color = barcolor)

    ax.set_yticks(np.arange(len(terms)))
    ax.set_yticklabels(terms)
    ax.invert_yaxis()
    ax.set(title = ontology, xlabel = '-log10 pvalue')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    if show_genes:
        for j, p in enumerate(ax.patches):
            _y = p.get_y() + p.get_height() - p.get_height()/3
            ax.text(0.1, _y, compact_string(genes[j], max_wordlen=10, join_spacer = ', ', label_genes = label_genes), 
                ha="left", color = text_color, fontsize = gene_fontsize)


def plot_enrichments(enrichment_results, show_genes = True, show_top = 10, barcolor = 'lightgrey', label_genes = [],
        text_color = 'black', return_fig = False, plots_per_row = 2, height = 4, aspect = 2.5, max_genes = 15,
        pval_threshold = 1e-5, color_by_adj = True, palette = 'Reds', gene_fontsize = 10):
    '''
    Make plot of geneset enrichments results.

    Parameters
    ----------
    show_genes : boolean, default = True
        Whether to show gene names on enrichment barplot bars
    show_top : int > 0, default = 10
        Plot this many top terms for each ontology
    barcolor : str or tuple[int] (r,g,b,a) or tuple[int] (r,g,b)
        Color of barplot bars
    label_genes : list[str] or np.ndarray[str]
        Add an asterisc by the gene name of genes in this list. Useful for
        finding transcription factors or signaling factors of interest in
        enrichment results.
    text_color : str or tuple[int] (r,g,b,a) or tuple[int] (r,g,b)
        Color of text on plot
    plots_per_row : int > 0, default = 2
        Number of onotology plots per row in figure
    height : float > 0, default = 4
        Height of each ontology plot
    aspect : float > 0, default = 2.5
        Aspect ratio of ontology plot
    max_genes : int > 0, default = 15
        Maximum number of genes to plot on each term bar
    pval_threshold : float (0, 1), default = 1e-5
        Upper bound on color map for adjusted p-value coloring of bar
        outlines.
    color_by_adj : boolean, default = True
        Whether to outline term bars with adjusted p-value
    palette : str
        Color palette for adjusted p-value
    gene_fontsize : float > 0, default = 10
        Fontsize of gene names on term bars

    Returns
    -------
    ax : matplotlib.pyplot.axes
    
    '''

    func = partial(_plot_enrichment, text_color = text_color, label_genes = label_genes, pval_threshold = pval_threshold,
            show_top = show_top, barcolor = barcolor, show_genes = show_genes, max_genes = max_genes,
            color_by_adj = color_by_adj, palette = palette, gene_fontsize=gene_fontsize)

    fig, ax = map_plot(func, list(enrichment_results.items()), plots_per_row = plots_per_row, 
        height =height, aspect = aspect)  

    if return_fig:
        return fig, ax
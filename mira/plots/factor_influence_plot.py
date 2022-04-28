
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from mira.plots.base import map_colors
import mira.adata_interface.core as adi
import mira.adata_interface.regulators as ri
from mira.tools.tf_targeting import _driver_TF_test
from functools import partial
import logging
logger = logging.getLogger(__name__)

def layout_labels(*, ax, x, y, label, label_closeness = 5, fontsize = 11, max_repeats = 5):

    xy = np.array([x,y]).T
    scaler = MinMaxScaler().fit(xy)
    xy = scaler.transform(xy)
    scaled_x, scaled_y = xy[:,0], xy[:,1]

    G = nx.Graph()
    num_labels = Counter(label)
    encountered_labels = Counter()
    
    sort_order = np.argsort(scaled_x + scaled_y)[::-1]
    scaled_x = scaled_x[sort_order]
    scaled_y = scaled_y[sort_order]
    label = np.array(label)[sort_order]
    
    new_label_names = []
    for i,j,l in zip(scaled_x, scaled_y, label):
        if num_labels[l] > 1:
            encountered_labels[l]+=1
            if encountered_labels[l] <= max_repeats:
                l = l + ' ({})'.format(str(encountered_labels[l])) #make sure labels are unique
                G.add_edge((i,j), l)
        else:
            G.add_edge((i,j), l)

    pos_dict = nx.drawing.spring_layout(G, k = 1/(label_closeness * np.sqrt(len(x))), 
        fixed = [(i,j) for (i,j),l in G.edges], 
        pos = {(i,j) : (i,j) for (i,j),l in G.edges})
        
    for (i,j),l in G.edges:
        axi, axj = scaler.inverse_transform(pos_dict[l][np.newaxis, :])[0]
        i,j = scaler.inverse_transform([[i,j]])[0]
        ax.text(axi, axj ,l, fontsize = fontsize)
        ax.plot((i,axi), (j,axj), c = 'black', linewidth = 0.2)

    return ax


def _join_factor_meta(m1, m2, show_factor_ids = False):

    def reformat_meta(meta):
        return {factor['id'] : factor for factor in meta}

    m1 = reformat_meta(m1)
    m2 = reformat_meta(m2)

    shared_factors = np.intersect1d(list(m1.keys()), list(m2.keys()))

    m1 = [m1[factor_id] for factor_id in shared_factors]
    m2 = [m2[factor_id] for factor_id in shared_factors]

    l1_pvals = np.array([x['pval'] for x in m1]).astype(float)
    l2_pvals = np.array([x['pval'] for x in m2]).astype(float)
    factor_names = np.array([(x['id'] + ': ' if show_factor_ids else '') + x['name'] for x in m1]).astype(str)

    return factor_names, l1_pvals, l2_pvals


def _influence_plot(ax, l1_pvals, l2_pvals, factor_names, pval_threshold = (1e-5, 1e-5), label_factors = None,
    hue = None, palette = 'coolwarm', legend_label = '', hue_order = None, show_legend = True, na_color = 'lightgrey',
    label_closeness = 2, max_label_repeats = 1, axlabels = ('list1', 'list2'), color = 'grey', fontsize = 12):

    if not hue is None:
        assert(isinstance(hue, (list, np.ndarray)))
        assert(len(hue) == len(factor_names))
        
        cell_colors = map_colors(ax, hue, palette, 
            add_legend = show_legend, hue_order = hue_order, na_color = na_color,
            cbar_kwargs = dict(location = 'right', pad = 0.01, shrink = 0.5, aspect = 15, label = legend_label),
            legend_kwargs = dict(loc= "center left", bbox_to_anchor = (1.05, 0.5), 
                markerscale = 1, frameon = False, title_fontsize='x-large', fontsize='large', title = legend_label))
    else:
        cell_colors = color

    l1_pvals = np.array(l1_pvals)
    l2_pvals = np.array(l2_pvals)

    x = -np.log10(l1_pvals)
    y = -np.log10(l2_pvals)
    
    b0 = -np.log10(pval_threshold[0])
    a0 = -np.log10(pval_threshold[1])

    name_mask = y > -a0/b0 * x + a0

    if not label_factors is None:
        assert(isinstance(label_factors, (list, np.ndarray)))
        name_mask = np.isin(factor_names, label_factors)

    ax.scatter(x, y, c = cell_colors)

    if name_mask.sum() > 0:
        layout_labels(ax = ax, x = x[name_mask], y = y[name_mask], label_closeness = label_closeness, 
            fontsize = fontsize, label = np.array(factor_names)[name_mask], max_repeats = max_label_repeats)
    else:
        logger.warn('No TFs met p-value thresholds.')

    ax.set(xlabel = axlabels[0], ylabel = axlabels[1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    #ax.axis('square')
    plt.tight_layout()

    return ax


def plot_factor_influence(
    factor_list_1, factor_list_2, label_factors = None, hue = None, palette = 'coolwarm', hue_order = None, 
    figsize = (8,8), legend_label = '', show_legend = True, fontsize = 13, 
    pval_threshold = (1e-50, 1e-50), na_color = 'lightgrey',
    color = 'lightgrey', label_closeness = 3, max_label_repeats = 3, show_factor_ids = False,
    ax = None, axlabels = ('list1', 'list2'), pval_pseudocount = 1e-300,
):

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize = figsize)

    factor_names, l1_pvals, l2_pvals = _join_factor_meta(factor_list_1, factor_list_2, 
            show_factor_ids = show_factor_ids)
    
    l1_pvals= l1_pvals + pval_pseudocount
    l2_pvals= l2_pvals + pval_pseudocount

    if not hue is None:
        assert(isinstance(hue, dict)), '"hue" argument must be dictionary of format {factor : value, ... }'
        hue = [hue[factor] if factor in hue else np.nan for factor in factor_names]

    return _influence_plot(ax, l1_pvals, l2_pvals, factor_names, pval_threshold = pval_threshold, 
        label_factors = label_factors, hue = hue, palette = palette, legend_label = legend_label, 
        hue_order = hue_order, show_legend = show_legend, na_color = na_color,
        label_closeness = label_closeness, max_label_repeats = max_label_repeats, 
        axlabels = axlabels, color = color, fontsize = fontsize)


@adi.wraps_functional(
    ri.fetch_driver_TF_test, adi.return_output,
    ['isd_matrix','genes','factors']
)
def compare_driver_TFs_plot(background = None, alt_hypothesis = 'greater', factor_type = 'motifs',
    axlabels = ('Set1 Drivers', 'Set2 Drivers'), label_factors = None,
    hue = None, palette = 'coolwarm', hue_order = None, ax = None, 
    figsize = (8,8), legend_label = '', show_legend = True, fontsize = 13, 
    pval_threshold = (1e-3, 1e-3), na_color = 'lightgrey', show_factor_ids = False,
    color = 'lightgrey', label_closeness = 3, max_label_repeats = 3,*,
    geneset1, geneset2, isd_matrix, genes, factors):
    '''
    Use pISD (probabilistic *insilico* deletion) association scores between 
    transcription factors and genes to compare and contrast 
    driving regulatorys of two genesets.

    The genesets may be determined by shared regulation (e.g. topics), 
    by differential expression, or based on similar regulatory dynamcis.

    .. note::

        Please refer to the :ref:`LITE/NITE tutorial </notebooks/tutorial_topic_model_tuning.ipynb>`
        for instruction on training RP models and calculating pISD scores.

    Parameters
    ----------

    background : np.ndarrary[str], list[str], None; default = None
        List of gene names to use for background of driver TF test.
        For each transcription factor, assesses association in each
        query geneset against the provided background. If no background
        provided, all compares query against all other genes for which
        pISD scores were calculated.
    geneset_1 : np.ndarray[str], list[str]
        Query geneset 1. List of genes linked by some process (similar
        regulatory dynamics, topic activtation, etc.).
    geneset_2 : np.ndarray[str], list[str]
        Query geneset 1. List of genes linked by some contrasting
        process.
    factor_type : str, 'motifs' or 'chip', default = 'motifs'
        Which factor type to use for enrichment.
    label_factors : list[str], np.ndarray[str], None; default=None
        List of factors to label. If not provided, will label all
        factors that meet the p-value thresholds.
    hue : dict[str : {str, float}] or None
        If provided, colors the factors on the plot. The keys of the dict
        must be the names of transcription factors, and the values are
        the associated data to map to colors. The values may be 
        categorical, e.g. cluster labels, or scalar, e.g. expression
        values. TFs not provided in the dict are colored as *na_color*.
    palette : str, list[str], or None; default = None
        Palette of plot. Default of None will set `palette` to the style-specific default.
    hue_order : list[str] or None, default = None
        Order to assign hues to features provided by `data`. Works similarly to
        hue_order in seaborn. User must provide list of features corresponding to 
        the order of hue assignment. 
    ax : matplotlib.pyplot.axes, deafult = None
        Provide axes object to function to add streamplot to a subplot composition,
        et cetera. If no axes are provided, they are created internally.
    figsize : tuple(float, float), default = (8,8)
        Size of figure
    legend_label : str, None
        Label for legend.
    show_legend : boolean, default=True
        Show figure legend.
    fontsize : int>0, default=13
        Fontsize of TF labels on plot.
    pval_threshold : tuple[float, float], default=(1e-50, 1e-50)
        Threshold below with TFs will not be labeled on plot. The first and
        second positions relate p-value with respect to topic 1 and topic 2.
    na_color : str, default='lightgrey'
        Color for TFs with no provided *hue*
    color : str, default='grey'
        If *hue* not provided, colors all points on plot this color.
    label_closeness : int>0, default=3
        Closeness of TF labels to points on plot. When *label_closeness* is high,
        labels are forced to be very close to points.
    max_label_repeats : boolean, default=3
        Some TFs have multiple ChIP samples or Motif PWMs. For these factors,
        label the top *max_label_repeats* examples. This prevents clutter when
        many samples for the same TF are close together. The rank of the sample
        for each TF is shown in the label as "<TF name> (<rank>)".

    Returns
    -------

    matplotlib.pyplot.axes

    '''

    driver_test = partial(_driver_TF_test, background = background, alt_hypothesis = alt_hypothesis,
        isd_matrix = isd_matrix, genes = genes, factors = factors)

    m1, m2 = driver_test(geneset = geneset1), driver_test(geneset = geneset2)

    return plot_factor_influence(m1, m2, ax = ax, label_factors = label_factors,
            pval_threshold = pval_threshold, hue = hue, hue_order = hue_order, 
            palette = palette, legend_label = legend_label, show_legend = show_legend, 
            label_closeness = label_closeness, figsize = figsize, 
            show_factor_ids = show_factor_ids, na_color = na_color, 
            max_label_repeats = max_label_repeats, axlabels = axlabels, 
            fontsize = fontsize, color = color)        

    
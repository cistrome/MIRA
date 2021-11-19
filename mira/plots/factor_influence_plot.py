
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from mira.plots.base import map_colors

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

def plot_factor_influence(ax, l1_pvals, l2_pvals, factor_names, pval_threshold = (1e-5, 1e-5), label_factors = None,
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
    layout_labels(ax = ax, x = x[name_mask], y = y[name_mask], label_closeness = label_closeness, 
        fontsize = fontsize, label = np.array(factor_names)[name_mask], max_repeats = max_label_repeats)

    ax.set(xlabel = axlabels[0], ylabel = axlabels[1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.axes('square')

    plt.tight_layout()

    return ax
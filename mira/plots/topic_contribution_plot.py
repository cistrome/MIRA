
from mira.plots.base import map_colors
import numpy as np
import matplotlib.pyplot as plt


def plot_topic_contributions(
    topic_contributions, highlight_topics,
        palette = ['red','black'],
        min_threshold = 0.05,
        figsize = (5,3),
        ax = None,
        size = 10,
        **plot_kwargs,
    ):
    '''
    Utility plot for choosing representative number of topics for a dataset in conjuction with the `gradient_tune` method.

    Parameters
    ----------
    topic_contributions : list
        Output from `mira.topics.gradient_tune`. Sorted list of maximum topic contributions.
    highlight_topics : int > 0
        Number of topics to highlight on plot. Helps to choose the number of topics corresponding
        with the "elbow".
    min_treshold ; float in (0,1), default = 0.05
        Threshold line to draw across plot.

    '''

    assert isinstance(highlight_topics, int) and highlight_topics > 0
    assert isinstance(min_threshold, (float, int)) and 0 < min_threshold < 1
    assert isinstance(topic_contributions, (list, np.ndarray))

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)

    max_n = len(topic_contributions)
    n_topics = np.arange(max_n) + 1
    valid_topic = (n_topics <= highlight_topics).astype(str)

    ax.scatter(
        n_topics,
        topic_contributions,
        s = size,
        c = map_colors(
            ax, valid_topic, palette = palette, add_legend = True, 
            cbar_kwargs = dict(orientation = 'vertical', pad = 0.01, shrink = 0.5, aspect = 15, anchor = (1.05, 0.5), label = 'Valid topic'),
            legend_kwargs = dict(loc="center left", markerscale = 4, frameon = False, title_fontsize='large', fontsize='medium',
                        bbox_to_anchor=(1.05, 0.5), title = 'Valid topic')
        ),
        **plot_kwargs,
    )

    ax.hlines(min_threshold, 1, max_n, color = 'black', linewidth = 1)

    ax.set(yscale = 'log', xlabel = 'Topic Number', ylabel = 'Max contribution')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax
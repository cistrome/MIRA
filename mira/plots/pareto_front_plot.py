
import matplotlib
import numpy as np
from mira.plots.factor_influence_plot import layout_labels
matplotlib.rc('font', size = 20)

from matplotlib.cm import get_cmap
from optuna.trial import TrialState as ts
from mira.plots.base import map_colors
import matplotlib.pyplot as plt

def get_RD_scores(trials):
    
    trial_num, scores = list(zip(*[
        (t.number, [t.user_attrs['distortion'], t.user_attrs['rate']])
        for t in trials
        if t.state == ts.COMPLETE
    ]))
    
    scores = np.array(scores)
    
    return trial_num, scores


def get_trial_attr(trial, attr):
    
    attr = attr.lower()
    
    if attr == 'value' or attr == 'elbo':
        if trial.value is None:
            return np.nan

        return trial.value
    elif attr == 'number':
        return trial.number
    
    try:
        return trial.params[attr]
    except KeyError:
        pass

    try:
        return trial.user_attrs[attr]
    except KeyError:
        pass

    try:
        return trial.system_attrs[attr]
    except KeyError:
        pass

    try:
        return getattr(trial, attr)
    except AttributeError:
        raise KeyError('Attr {} not found in params or user_attrs of trial'.format(attr))
    

def _dominates(
    values0, values1
):

    if len(values0) != len(values1):
        raise ValueError("Trials with different numbers of objectives cannot be compared.")

    d = all(v0 <= v1 for v0, v1 in zip(values0, values1))
    return d


def _get_pareto_front_trials_nd(trial_num, scores):
    
    assert isinstance(scores, np.ndarray)
    
    pareto_front = []

    for t1, score in zip(trial_num, scores):
        dominated = False
        for t2, other_score in zip(trial_num, scores):
            if not t1 == t2 and _dominates(other_score, score):
                dominated = True
                break
        if not dominated:
            pareto_front.append(t1)

    return pareto_front


def plot_pareto_front(trials, 
      x = 'rate',
      y = 'distortion',
      hue = 'usefulness',
      ax = None, 
      figsize = (7,7),
      palette = 'Blues',
      na_color = 'lightgrey',
      size = 100,
      alpha = 0.8,
      add_legend = True,
      label_pareto_front = False,
      include_pruned_trials = True,
     ):
    
    trials = [t for t in trials if t.state in [ts.PRUNED, ts.COMPLETE]]
    
    best_trial = sorted(trials, key = lambda t : -t.values[0] if t.state == ts.COMPLETE else -np.inf)[-1]
    
    _x = np.array([ get_trial_attr(t, x) for t in trials ])
    _y = np.array([ get_trial_attr(t, y) for t in trials ])
    completed = np.array([ t.state == ts.COMPLETE for t in trials ])
    
    if label_pareto_front:
        pareto_front_trials = _get_pareto_front_trials_nd(*get_RD_scores(trials))
    else:
        pareto_front_trials = [t.number for t in trials] # just label all

    is_pareto_front = np.array([ 
        t.number in pareto_front_trials and completed[i] \
        for i, t in enumerate(trials)
    ])
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize = figsize)
    
    if not hue is None:
        _hue = [ get_trial_attr(t, hue) if front else np.nan 
                      for t, front in zip(trials, is_pareto_front) ]
        
        if ~np.isnan(_hue).all():
            color = map_colors(ax, np.array(_hue), palette = palette, na_color = na_color,
                                cbar_kwargs=dict(orientation = 'vertical', pad = 0.01, shrink = 0.5, 
                                aspect = 15, anchor = (1.05, 0.5), label = hue.capitalize()),
                                add_legend = add_legend)
    else:
        _hue = [np.nan]*len(trials)
        color = map_colors(None, np.array(_hue), palette = palette, na_color = na_color,
                                    add_legend = False)
    
    
    ax.scatter(
        _x[completed], _y[completed],
        c = color[completed],
        marker = 'o',
        s = size * (1.5*is_pareto_front[completed]*label_pareto_front + 1),
        alpha = alpha,
        edgecolor = 'black',
        label = 'completed trials',
    )
    
    if not include_pruned_trials:
        ax.margins(0.1)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    
    ax.scatter(
        _x[~completed], _y[~completed],
        c = color[~completed],
        marker = 'x',
        s = size,
        alpha = alpha,
        label = 'pruned trials',
    )

    if include_pruned_trials:
        ax.margins(0.1)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    
    ax.set(
        xlim = xlim, ylim = ylim,
        xlabel = x.capitalize(),
        ylabel = y.capitalize(),
    )
    
    layout_labels(ax = ax, 
                  x = _x[is_pareto_front], 
                  y = _y[is_pareto_front], 
                  label = np.array([str(t.number) for t in trials])[is_pareto_front], 
                  label_closeness=5, fontsize=20,
    )
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if add_legend:
        ax.legend()

    return ax
    

def plot_intermediate_values(trials, palette = 'Greys', 
                            na_color = 'lightgrey',
                            add_legend = True,
                            log_hue = False,
                            hue = 'value',
                            vmin = None,
                            vmax = None,
                            ax = None, 
                            figsize = (10,7),
                            **plot_kwargs):
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize = figsize)

    def get_or_nan(trial):
        try:
            return get_trial_attr(trial, hue)
        except KeyError:
            return np.nan

    trials = [t for t in trials if t.state in [ts.PRUNED, ts.COMPLETE]]

    trial_values = np.array([ get_or_nan(t) for t in trials] )

    cbar_params = dict(orientation = 'vertical', pad = 0.01, shrink = 0.5, 
                                 aspect = 15, anchor = (1.05, 0.5), label = hue)

    legend_params = dict(loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon = False, 
                ncol = 4, title_fontsize='x-large', fontsize='large', markerscale = 1)


    colors = map_colors(ax, trial_values, 
                palette = palette, 
                na_color = na_color,
                log = log_hue,
                vmin = vmin, vmax = vmax,
                cbar_kwargs=cbar_params, 
                legend_kwargs=legend_params,
                add_legend = add_legend)
    
    for trial, _c in zip(trials, colors):
        
        ax.plot(
            list(trial.intermediate_values.keys()),
            list(trial.intermediate_values.values()),
            c = _c,
            **plot_kwargs,
        )
        
    ax.set(yscale = 'log',
           xlabel = 'Epoch', ylabel = 'Loss',
          )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax
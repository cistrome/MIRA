
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
        return trial.value
    
    if attr == 'usefulness':
        return trial.user_attrs[attr] if attr in trial.user_attrs else np.inf
    
    try:
        return trial.params[attr]
    except KeyError:
        return trial.user_attrs[attr]
    except KeyError():
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
      color = 'usefulness',
      ax = None, 
      figsize = (7,7),
      palette = 'Blues',
      na_color = 'lightgrey',
      size = 100,
      alpha = 0.8,
      add_legend = True,
     ):
    
    trials = [t for t in trials if t.state in [ts.PRUNED, ts.COMPLETE]]
    
    best_trial = sorted(trials, key = lambda t : -t.values[0] if t.state == ts.COMPLETE else -np.inf)[-1]
    
    rates = np.array([ get_trial_attr(t, x) for t in trials ])
    distortions = np.array([ get_trial_attr(t, y) for t in trials ])
    completed = np.array([ t.state == ts.COMPLETE for t in trials ])
    
    pareto_front_trials = _get_pareto_front_trials_nd(*get_RD_scores(trials))
    is_pareto_front = np.array([ t.number in pareto_front_trials for t in trials ])
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize = figsize)
    
    if not color is None:
        usefulness = [ get_trial_attr(t, color) if front else np.nan 
                      for t, front in zip(trials, is_pareto_front) ]
        
        if ~np.isnan(usefulness).all():
            usefulness_color = map_colors(ax, np.array(usefulness), palette = palette, na_color = na_color,
                cbar_kwargs=dict(orientation = 'vertical', pad = 0.01, shrink = 0.5, 
                                 aspect = 15, anchor = (1.05, 0.5), label = color.capitalize()),
                                add_legend = add_legend)
    else:
        usefulness = [np.nan]*len(trials)
        usefulness_color = map_colors(None, np.array(usefulness), palette = palette, na_color = na_color,
                                    add_legend = False)
    
    marker = ['o' if complete else 'x' for complete in completed ]
    
    ax.scatter(
        rates[completed], distortions[completed],
        c = usefulness_color[completed],
        marker = 'o',
        s = size * (1.5*is_pareto_front[completed] + 1),
        alpha = alpha,
        edgecolor = 'black',
        label = 'completed trials',
    )
    
    ax.margins(0.1)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    
    ax.scatter(
        rates[~completed], distortions[~completed],
        c = usefulness_color[~completed],
        marker = 'x',
        s = size,
        alpha = alpha,
        label = 'pruned trials',
    )
    
    if x.lower() == 'rate' and y.lower() == 'distortion':
        R0, D0 = best_trial.user_attrs['rate'], best_trial.user_attrs['distortion']
        ax.plot(
            (R0+D0 - 1,0),(0,R0+D0 - 1), '--',
            label = 'min ELBO', color = 'black',
        )
        ylim = (ylim[0]*0.999, ylim[1])
    
    ax.set(
        xlim = xlim, ylim = ylim,
        xlabel = x.capitalize(),
        ylabel = y.capitalize(),
    )
    
    layout_labels(ax = ax, 
                  x = rates[is_pareto_front], 
                  y = distortions[is_pareto_front], 
                  label = np.array([str(t.number) for t in trials])[is_pareto_front], 
                  label_closeness=5, fontsize=20,
    )
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if add_legend:
        ax.legend()
    

def plot_intermediate_values(trials, palette = 'Greys', 
                             ax = None, figsize = (10,7),
                             linecolor = 'black'):
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize = figsize)
    
    trials = sorted(trials, key = lambda t : -t.values[0] if t.state == ts.COMPLETE else -np.inf)
    
    for i, trial in enumerate(trials):
        
        ax.plot(
            list(trial.intermediate_values.keys()),
            list(trial.intermediate_values.values()),
            c = get_cmap(palette)((i/len(trials))**2 + 0.05),
        )
        
    ax.set(yscale = 'log',
           xlabel = 'Epoch', ylabel = 'Loss',
          )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    #ax.vlines(min_resource, ymin = 0, ymax = trials[-1].values[0] * 0.97,
    #             color = linecolor)
    
    #ax.vlines(2*min_resource, ymin = 0, ymax = trials[-1].values[0] * 0.97,
    #             color = linecolor)

    return ax
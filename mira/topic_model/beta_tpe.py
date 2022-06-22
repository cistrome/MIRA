
from re import search
from optuna.trial import TrialState as ts
from optuna.samplers._tpe.sampler import default_weights
from optuna.samplers import TPESampler
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters
from optuna.samplers._tpe.sampler import _split_observation_pairs, _get_observation_pairs
import numpy as np
from collections import defaultdict
from functools import partial

def beta_stratified_terminate(trials, 
        min_trials = 16, 
        max_trials = 32, 
        n_failures = 8):

    trials = [t.state == ts.PRUNED or t.state == ts.COMPLETE for t in trials]

    if len(trials) < min_trials or len(trials) < n_failures:
        return False
    elif len(trials) >= max_trials:
        return True
    
    best_trial_no = np.argmin([t.value for t in trials])

    failures = 0
    for trial in trials[best_trial_no:]:
        if trial.state == ts.PRUNED or trial.state == ts.COMPLETE:
            failures+=1

    if failures > n_failures:
        return True

    return False


def weight_trials(n_trials,*, trials,
    reconstruction_weight, n_beta_options = 4, 
    parallel = True):

    allowed_states = [ts.PRUNED, ts.COMPLETE]
    if parallel:
        allowed_states.append(ts.RUNNING)

    mask = np.array([t.params['reconstruction_weight'] == reconstruction_weight if 'reconstruction_weight' in t.params else False
                    for t in trials if t.state in allowed_states])

    print(n_trials, allowed_states, len(mask))
    assert len(mask) == n_trials

    weights = default_weights(n_trials)
    weights[~mask]/=n_beta_options

    return weights


class BetaStratifiedTPESampler(TPESampler):

    def __init__(self, *args, 
        min_trials = 16, 
        max_trials = 32,
        n_failures = 8,
        **kwargs
        ):

        super().__init__(*args, **kwargs)

        self._min_trials = min_trials
        self._max_trials = max_trials
        self._n_failures = n_failures
    

    def _get_available_beta_options(self, study, beta_dist):
        
        beta_trials = defaultdict(list)

        for trial in study.trials:
            try:
                beta_trials[trial.params['reconstruction_weight']].append(
                    trial
                )
            except KeyError:
                pass

        can_sample = dict(zip(
            beta_dist.choices, [True]*len(beta_dist.choices)
        ))

        terminated = {
            beta : not beta_stratified_terminate(trials, 
                min_trials=self._min_trials, 
                max_trials=self._max_trials,
                n_failures=self._n_failures)
            
            for beta, trials in beta_trials.items() 
        }

        can_sample.update(terminated)

        return [beta for beta, samp in can_sample.items() if samp]


    def _set_TPE_params(self, weight_fn):

        curr = self._parzen_estimator_parameters

        self._parzen_estimator_parameters = _ParzenEstimatorParameters(
            curr.consider_prior,
            curr.prior_weight,
            curr.consider_magic_clip,
            curr.consider_endpoints,
            weight_fn,
            curr.multivariate,
        )


    def _sample_relative(
        self, study, trial, search_space,
    ):

        if search_space == {}:
            return {}
        
        beta_dist = search_space.pop('reconstruction_weight')

        beta_options = self._get_available_beta_options(study, beta_dist)

        print(beta_options)
        reconstruction_weight = self._rng.choice(beta_options)

        weight_fn = partial(
            weight_trials, parallel = self._constant_liar, 
            trials = study.trials, reconstruction_weight = reconstruction_weight, 
            n_beta_options = len(beta_dist.choices)
        )

        self._set_TPE_params(weight_fn)

        samples = super()._sample_relative(
            study, trial, search_space
        )

        samples['reconstruction_weight'] = \
                beta_dist.to_external_repr(reconstruction_weight)

        return samples
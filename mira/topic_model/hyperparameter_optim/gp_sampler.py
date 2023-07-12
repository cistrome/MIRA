from optuna._transform import _SearchSpaceTransform
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern
from functools import partial
from collections import defaultdict
from optuna.trial import TrialState as ts
from optuna.samplers import BaseSampler
from scipy.stats import norm
import optuna
import numpy as np
import matplotlib.pyplot as plt
import math


def is_allowed_trial(trial, constant_liar, search_space):
    
    if constant_liar:
        allowed_states = [ts.RUNNING, ts.COMPLETE, ts.PRUNED]
    else:
        allowed_states = [ts.COMPLETE, ts.PRUNED]
        
    return trial.state in allowed_states and \
            all([param in trial.params for param in search_space.keys()])


def iterate_rung_scores(trial):

        rung_num = 0
        while 'completed_rung_' + str(rung_num) in trial.system_attrs:
            yield trial, rung_num, trial.system_attrs['completed_rung_' + str(rung_num)]
            rung_num += 1
            
        if trial.state == ts.COMPLETE:
            yield trial, rung_num, trial.value


class SuccessiveHalvingPruner(optuna.pruners.SuccessiveHalvingPruner):

    def get_rung_score_thresholds(self, study, trial):

        rung_scores = defaultdict(list)
        
        for trial in study.trials:
            for trial, rung, score in iterate_rung_scores(trial):
                rung_scores[rung].append(score)

        rung_thresholds = defaultdict(lambda : np.inf)
        for rung, scores in rung_scores.items():
            rung_thresholds[rung] = np.quantile(scores, 1/self._reduction_factor)
            
        return rung_thresholds


class HyperbandPruner(optuna.pruners.HyperbandPruner):

    def get_rung_score_thresholds(self, study, trial):
        if len(self._pruners) == 0:
            self._try_initialization(study)
            if len(self._pruners) == 0:
                return False

        bracket_id = self._get_bracket_id(study, trial)
        bracket_study = self._create_bracket_study(study, bracket_id)

        return self._pruners[bracket_id].get_rung_score_thresholds(bracket_study, trial)


    def _try_initialization(self, study: "optuna.study.Study") -> None:
        if self._max_resource == "auto":
            trials = study.get_trials(deepcopy=False)
            n_steps = [
                t.last_step
                for t in trials
                if t.state == ts.COMPLETE and t.last_step is not None
            ]
            if not n_steps:
                return

            self._max_resource = max(n_steps) + 1

        assert isinstance(self._max_resource, int)

        if self._n_brackets is None:
            self._n_brackets = (
                math.floor(
                    math.log(self._max_resource / self._min_resource, self._reduction_factor)
                )
                + 1
            )

        #_logger.debug("Hyperband has {} brackets".format(self._n_brackets))

        for bracket_id in range(self._n_brackets):
            trial_allocation_budget = self._calculate_trial_allocation_budget(bracket_id)
            self._total_trial_allocation_budget += trial_allocation_budget
            self._trial_allocation_budgets.append(trial_allocation_budget)

            pruner = SuccessiveHalvingPruner(
                min_resource=self._min_resource,
                reduction_factor=self._reduction_factor,
                min_early_stopping_rate=bracket_id,
                bootstrap_count=self._bootstrap_count,
            )
            self._pruners.append(pruner)


def get_constant_liar_scores(trials, cl_function = np.max):
    
    rung_scores = defaultdict(list)
    for trial in trials:
        for trial, rung, score in iterate_rung_scores(trial):
            rung_scores[rung].append(score)

    constant_liar_scores = {rung_num : cl_function(scores)
                            for rung_num, scores in rung_scores.items()}
    
    return constant_liar_scores


def featurize_trials(trials, search_space, constant_liar,
                    constant_liar_scores):
    
    examples = []
    for trial in trials:

        if constant_liar and trial.state == ts.RUNNING:
            for rung, score in constant_liar_scores.items():
                examples.append(
                    (trial.params, rung, score)
                )
        
        else:
            for trial, rung, score in iterate_rung_scores(trial):
                examples.append(
                    (trial.params, rung, score)
                )
    
    # add a dummy constant liar trial to prevent GP from selecting the same
    # values repeatedly?
    #if not constant_liar:
    #    last_params = examples[-1][0]
    #    
    #    for rung, score in constant_liar_scores.items():
    #        examples.append(
    #            (last_params, rung, score)
    #        )
        
    params, rungs, scores = list(zip(*examples))
    
    return params, rungs, scores


def get_regressor(seed = 0, gpr = None):
    
    if gpr is None:
        gpr = GaussianProcessRegressor(
                kernel= Matern(nu = 5/2) + WhiteKernel() + ConstantKernel(),
                random_state=seed, 
                normalize_y=True, 
                n_restarts_optimizer=10,
                alpha = 0.
            )
    
    return Pipeline([
        ('scaler', StandardScaler()),
        ('gpr', gpr)
    ])


def aquisition_function(y_hat, y_std, best_value, temp = 0.01,
                       direction = 'minimize'):
    
    #if direction == 'minimize':
    y_hat, best_value = -y_hat, -best_value
        
    improvement = y_hat - best_value - best_value * temp
    Z = improvement/y_std
    
    return improvement * norm().cdf(Z) + y_std * norm().pdf(Z)


def format_params_as_input(params, rung, transformer, search_space):
        
    params = np.vstack(
        [transformer.transform(p)[None,:] for p in params]
    )
    
    params = np.hstack([params, np.array(rung)[:,None]])
    return params
    


def generate_candidates(sampling_fn, search_space, n_candidates):
    
    candidates = []
    for i in range(n_candidates):
        candidates.append(
            {
                param_name : sampling_fn(param_name, dist)
                for param_name, dist in search_space.items()
            }
        )
        
    return candidates


def get_P_promotion(rung_thresholds, rung, score, score_std):

    thresholds = np.array([rung_thresholds[r] for r in rung])

    Z = (score - thresholds)/score_std
    return 1-norm.cdf(Z)


class GP(BaseSampler):
    
    def __init__(self, 
                seed = 0, 
                tau = 0.01, 
                min_points = 5, 
                constant_liar = False,
                num_candidates = 100,
                debug = False,
                cl_function = np.max,
                gpr = None):
                
        self._rng = np.random.RandomState(seed)
        self._tau = tau
        self._min_points = min_points
        self._constant_liar = constant_liar
        self._num_candidates = num_candidates
        self._random_sampler = optuna.samplers.RandomSampler(seed=seed)
        self._debug = debug
        self._cl_function = cl_function
        self._gpr = gpr

        
    def reseed_rng(self):
        self._rng = np.random.RandomState()
        self._random_sampler.reseed_rng()
        
        
    def sample_relative(self, study, trial, search_space):

        if search_space == {}:
            return {}
        
        assert not any([isinstance(space, optuna.distributions.CategoricalDistribution)
                        for space in search_space.values()]), 'Gaussian process sampler not compatible with categorical hyperparameters.'
                
        transformer = _SearchSpaceTransform(search_space)
        
        trials = [t for t in study.trials if is_allowed_trial(t, self._constant_liar, search_space)]
        
        if len(trials) < self._min_points:
            return self._random_sampler.sample_relative(study, trial, search_space)
        
        params, rungs, scores = featurize_trials(trials, search_space, self._constant_liar,
                                                 get_constant_liar_scores(trials, self._cl_function))

        X = format_params_as_input(params, rungs, transformer, search_space)
        y = np.array(scores)
                
        gpr = get_regressor(seed = self._rng, gpr = self._gpr).fit(X,y)
        
        sampling_fn = partial(self._random_sampler.sample_independent, study, trial)
        sampled_candidates = generate_candidates(sampling_fn, search_space, self._num_candidates)
        
        max_rung = max(rungs)
        candidates = [candidate for candidate in sampled_candidates for i in range(max_rung+1)]
        candidate_rungs = np.array(list(range(max_rung+1))*self._num_candidates)
        
        X_hat = format_params_as_input(candidates, candidate_rungs, transformer, search_space)
        
        yhat_mu, yhat_std = gpr.predict(X_hat, return_std = True)
        
        rung_thresholds = study.pruner.get_rung_score_thresholds(study, trial)
        
        p_rung_promotion = get_P_promotion(rung_thresholds, candidate_rungs, yhat_mu, yhat_std)\
                            .reshape((-1, max_rung+1))[:,:-1]
        
        p_survives = np.exp(np.log(p_rung_promotion).sum(-1))
        
        score_mu, score_std = yhat_mu[candidate_rungs == max_rung], yhat_std[candidate_rungs == max_rung]

        best_value = min([t.value for t in trials if t.state == ts.COMPLETE])
        f = aquisition_function(score_mu, score_std,
                                best_value, temp= self._tau, direction=study.direction,)
        
        if self._debug:
            #self.debug_plot(X, y, X_hat[candidate_rungs == max_rung],
            #                score_mu, score_std, p_survives, f)

            return X, y, X_hat[candidate_rungs == max_rung],\
                            score_mu, score_std, p_survives, f
        

        return sampled_candidates[np.argmax(p_survives * f)]
    
    def infer_relative_search_space(self, study, trial):
        return optuna.samplers.intersection_search_space(study)
    
    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._random_sampler.sample_independent(study, trial, param_name, param_distribution)
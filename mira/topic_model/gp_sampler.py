from optuna.trial import TrialState as ts
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from collections import defaultdict
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from optuna.samplers import BaseSampler
import numpy as np
import optuna


class SuccessiveHalvingPruner(optuna.pruners.SuccessiveHalvingPruner):

    def get_rung_score_thresholds(self, trials):

        rungs = sorted(
            set([
                key for trial in trials for key in trial.system_attrs if 'completed_rung_' in key
            ]), 
            key = lambda x : int(x[15:])
        )

        score_matrix = defaultdict(list)
        for t in trials:
            for rung in rungs:
                if rung in t.system_attrs:
                    score_matrix[rung].append(
                            t.system_attrs[rung]
                    )

        rung_thresholds = np.zeros(len(rungs) + 1)
        for rung, scores in enumerate(score_matrix.values()):

            rung_thresholds[rung] = np.quantile(scores, 1/self._reduction_factor)
            
        return rung_thresholds


def get_P_promotion(rung_thresholds, rung, score, score_std):
    Z = (score - rung_thresholds[np.array(rung)])/score_std
    return 1-norm.cdf(Z)


def filter_trials(trials, constant_liar):
    
    if constant_liar:
        allowed_states = [ts.RUNNING, ts.COMPLETE, ts.PRUNED]
    else:
        allowed_states = [ts.COMPLETE, ts.PRUNED]
        
    return [
        t for t in trials
        if t.state in  allowed_states \
            and 'num_topics' in t.params
    ]


def get_examples(complete_and_running):
    
    examples = []
    for trial in complete_and_running:
        rung_num = 0
        while 'completed_rung_' + str(rung_num) in trial.system_attrs:
            examples.append(
                (trial.params['num_topics'], rung_num, trial.system_attrs['completed_rung_' + str(rung_num)])
            ) # score after each rung
            rung_num +=1
        
        if trial.state == ts.COMPLETE:
            examples.append(
                (trial.params['num_topics'], rung_num, trial.value)
            ) # final score
        
    num_topics, rungs, scores = list(zip(*examples))
    
    return num_topics, rungs, scores


def substitute_constant_liar(complete_and_running, num_topics, rungs, scores,
                            constant_liar = False):
    
    cl_fn = np.max
    
    collect_rungscores = defaultdict(list)
    for rung, score in zip(rungs, scores):
        collect_rungscores[rung].append(score)
        
    rung_cl_scores = {rung : cl_fn(scores) for rung, scores in collect_rungscores.items()}
    num_rungs = max(rungs)
    
    examples = []
    for i, trial in enumerate(complete_and_running):
        for rung_num in range(num_rungs + 1):
            if not constant_liar and i == (len(complete_and_running) - 1):
                examples.append(
                        (trial.params['num_topics'], rung_num, rung_cl_scores[rung_num])
                    )
            else:
                if trial.state == ts.RUNNING:
                    examples.append(
                        (trial.params['num_topics'], rung_num, rung_cl_scores[rung_num])
                    )
                elif 'completed_rung_' + str(rung_num) in trial.system_attrs:
                    examples.append(
                        (trial.params['num_topics'], rung_num, trial.system_attrs['completed_rung_' + str(rung_num)])
                    )
                elif trial.state == ts.COMPLETE:
                    examples.append(
                        (trial.params['num_topics'], rung_num, trial.value)
                    )
                
            
    num_topics, rungs, scores = list(zip(*examples))
    
    return num_topics, rungs, scores
    

def get_prediction_features(num_topics, rungs, topic_range = (5,55)):
    
    max_rungs = np.max(rungs) 
    
    num_topics, rungs = list(zip(*[
              [topic, rung]
              for topic in range(*topic_range) for rung in range(max_rungs+1)
             ]))
    
    return num_topics, rungs
            

def get_regressor(seed = 0):
    
    return GaussianProcessRegressor(
            kernel= RBF() + WhiteKernel() + ConstantKernel(),
        random_state=seed, normalize_y=True, n_restarts_optimizer=10,
        alpha = 0.
    )


def aquisition_function(y_hat, y_std, p_promoted, best_value, temp = 0.01,
                       direction = 'minimize'):
    
    if direction == 'minimize':
        y_hat, best_value = -y_hat, -best_value
        
    improvement = y_hat - best_value - best_value * temp
    Z = improvement/y_std
    
    return p_promoted * (improvement * norm().cdf(Z) + y_std * norm().pdf(Z))


class ELBOGP(BaseSampler):

    def __init__(self, 
                seed = 0, 
                tau = 0.01, 
                min_points = 5, 
                constant_liar = False):
                
        self._rng = np.random.RandomState(seed)
        self._tau = tau
        self._min_points = min_points
        self._constant_liar = constant_liar

    def reseed_rng(self):
        self._rng = np.random.RandomState()

    def sample_relative(self, study, trial, search_space):
        if search_space == {}:
            return {}
        
        topic_distribution = search_space['num_topics']
        min_topics, max_topics = topic_distribution.low, topic_distribution.high
        
        trials = filter_trials(study.trials, self._constant_liar)
        
        n_points = len(trials)
        if n_points < self._min_points:
            return {'num_topics' : np.geomspace(min_topics, max_topics, self._min_points + 2).astype(int)[1:-1][n_points]}       

        x1, x2, y = get_examples(trials)
        
        xhat_1, xhat_2 = get_prediction_features(x1, x2, topic_range=(min_topics, max_topics))

        x1, x2, y = substitute_constant_liar(trials, x1, x2, y,
                                            constant_liar = self._constant_liar)

        def prep(x):
            return np.array(x)[:,None]

        topic_scaler = StandardScaler()
        rung_scaler = MinMaxScaler()
        X = np.hstack([
            topic_scaler.fit_transform(prep(x1)), 
            rung_scaler.fit_transform(prep(x2))
        ])
        
        X_hat = np.hstack([
            topic_scaler.transform(prep(xhat_1)), 
            rung_scaler.transform(prep(xhat_2))
        ])
        
        maxrung = np.max(x2)
        y_hat, y_hat_std = get_regressor(seed = self._rng).fit(X, y).predict(X_hat, return_std = True)
        
        rung_thresholds = study.pruner.get_rung_score_thresholds(trials)

        p_promotion = get_P_promotion(rung_thresholds, xhat_2, y_hat, y_hat_std)
        p_promotion = p_promotion.reshape((-1, maxrung + 1))[:,:-1]
        p_promoted = np.exp(np.log(p_promotion).sum(-1))
        
        y_hat = y_hat[maxrung::maxrung+1]
        y_hat_std = y_hat_std[maxrung::maxrung+1]

        f = aquisition_function(y_hat, y_hat_std, p_promoted, study.best_value, temp= self._tau)
        
        if not self._constant_liar:
            import seaborn as sns
            import matplotlib.pyplot as plt
            sns.lineplot(
                y = f,
                x = np.arange(min_topics, max_topics)
            )
            plt.show()
            
            for r in range(p_promotion.shape[1]):
                sns.lineplot(
                    y = p_promotion[:,r],
                    x = np.arange(min_topics, max_topics)
                )
                
            plt.show()

            sns.lineplot(
                y = y_hat.reshape(-1),
                x = X_hat[maxrung::maxrung+1,0],
            )

            sns.scatterplot(
                y = np.array(y).reshape(-1),
                x = X[:,0],
                hue = X[:,1],
                legend=False,
                size = np.arange(len(y)),
                sizes = (10,50),
                edgecolor = 'black',
                palette='viridis',
            )

            plt.fill_between(
                x = X_hat[::maxrung+1,0], 
                y1 = y_hat + y_hat_std, 
                y2 = y_hat - y_hat_std,
                alpha = 0.2, color = 'lightgrey'
            )
            plt.show()
            
        return {
            'num_topics' : np.arange(min_topics, max_topics)[np.nanargmax(f)]
        }

    def infer_relative_search_space(self, study, trial):
        return optuna.samplers.intersection_search_space(study)

    def sample_independent(self, study, trial, param_name, param_distribution):
        assert param_name == 'num_topics'

        n_points = max(study.user_attrs['n_workers'], self._min_points) # expected number of independent draws
        split_num = trial.number

        if split_num >= n_points:
            independent_sampler = optuna.samplers.RandomSampler()
            return independent_sampler.sample_independent(study, trial, param_name, param_distribution)
        else:
            
            suggest_topics = np.geomspace(param_distribution.low, param_distribution.high, n_points + 2)\
                        [1:-1].astype(int)
            return param_distribution.to_external_repr(suggest_topics[split_num])
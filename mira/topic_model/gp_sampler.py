from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from optuna.trial import TrialState as ts
from optuna.samplers import BaseSampler
import optuna
from scipy.stats import norm


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


def featurize_trials(complete_and_running, min_topics = 3, max_topics = 55):
    
    topic_scaler, epochs_scaler = StandardScaler(), MinMaxScaler()
    
    num_topics = np.array([t.params['num_topics'] for t in complete_and_running])[:,None]
    
    max_epochs_trained = max([t.user_attrs['epochs_trained'] for t in complete_and_running
                          if 'epochs_trained' in t.user_attrs])
    
    num_epochs = np.array([t.user_attrs['epochs_trained'] if 'epochs_trained' in t.user_attrs else max_epochs_trained
                           for t in complete_and_running])[:,None]
    
    X = np.hstack([
        topic_scaler.fit_transform(num_topics),
        epochs_scaler.fit_transform(num_epochs)
    ])
    
    test_topics = np.arange(min_topics, max_topics)[:,None]
    X_hat = np.hstack([
        topic_scaler.transform(test_topics),
        epochs_scaler.transform(
            np.ones_like(test_topics) * max(num_epochs)
        ),
    ])
    
    return X, X_hat, num_topics


def normalize_target(complete_and_running, constant_liar, 
                     score_accessor = lambda x : x.value):

    cl_fn = max if constant_liar else np.mean
    
    constant_liar_score = cl_fn([
        score_accessor(t) for t in complete_and_running
        if t.state != ts.RUNNING
    ])
    
    scores = np.array([
        score_accessor(t) if t.state != ts.RUNNING else constant_liar_score
        for t in complete_and_running
    ])

    if not constant_liar:
        scores[-1] = constant_liar_score
    
    return scores


def get_regressor(seed = 0):
    
    return GaussianProcessRegressor(
            kernel= RBF(length_scale_bounds = (0.5, 1000)) + WhiteKernel() + ConstantKernel(),
        random_state=seed, alpha= 0., normalize_y=True, n_restarts_optimizer=0,
    )


def aquisition_function(y_hat, y_std, best_value, temp = 0.01,
                       direction = 'minimize'):
    
    if direction == 'minimize':
        y_hat, best_value = -y_hat, -best_value
        
    improvement = y_hat - best_value - temp
    Z = improvement/y_std
    
    return improvement * norm().cdf(Z) + y_std * norm().pdf(Z)


class GP_RateDistortionSampler(BaseSampler):
    def __init__(self, seed = 0, tau = 10, 
                min_points = 5, constant_liar = False):
        self._rng = np.random.RandomState(seed)
        self._tau = tau  # Current temperature.
        self._min_points = min_points
        self._constant_liar = constant_liar


    def sample_relative(self, study, trial, search_space):
        if search_space == {}:
            return {}
        
        topic_distribution = search_space['num_topics']
        min_topics, max_topics = topic_distribution.low, topic_distribution.high
        
        complete_and_running = filter_trials(study.trials, self._constant_liar)
        
        X, X_hat, _nt = featurize_trials(complete_and_running, 
                                    min_topics = min_topics,
                                    max_topics= max_topics)
        
        rate = normalize_target(complete_and_running, self._constant_liar,
                                score_accessor= lambda x : x.user_attrs['rate'])

        distortion = normalize_target(complete_and_running, self._constant_liar,
                                score_accessor= lambda x : x.user_attrs['distortion'])
        
        n_points = len(rate)
        if n_points < self._min_points:
            return {'num_topics' : 
                        np.geomspace(min_topics, max_topics, self._min_points + 2).astype(int)\
                        [1:-1][n_points]
                    }            
        
        mu_rate, std_rate = get_regressor(seed = 0).fit(X,rate)\
            .predict(X_hat, return_std=True)
        mu_distortion, std_distortion = get_regressor(seed = 0).fit(X,distortion)\
            .predict(X_hat, return_std=True)

        y_hat = mu_rate + mu_distortion
        y_std = np.sqrt(std_distortion**2 + std_rate**2)

        f = aquisition_function(y_hat, y_std, study.best_value, temp= self._tau)
        
        if not self._constant_liar:
            import seaborn as sns
            import matplotlib.pyplot as plt
            sns.lineplot(
                y = f,
                x = np.arange(min_topics, max_topics)
            )
            plt.show()

            sns.scatterplot(
                y = distortion,
                x = _nt.reshape(-1),
                hue = X[:,1],
                palette = 'viridis',
                legend = False,
            )

            ax = sns.lineplot(
                y = mu_distortion,
                x = np.arange(min_topics, max_topics),
                legend = False,
            )

            ax.fill_between(
                x = np.arange(min_topics, max_topics),
                y1 = mu_distortion - std_distortion,
                y2 = mu_distortion + std_distortion,
                alpha = 0.2,
                color = 'lightgrey',
            )
            plt.show()
            
        return {
            'num_topics' : np.arange(min_topics, max_topics)[f.argmax()]
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
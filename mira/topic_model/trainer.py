
import numpy as np
from sklearn.model_selection import train_test_split
from functools import partial
import os
import optuna
from optuna.trial import TrialState as ts
from optuna.study._optimize import _run_trial
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
import gc
import sys
import os
from joblib import Parallel, delayed
import fcntl
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Setup the root logger.
from mira.topic_model.base import logger as baselogger
from mira.topic_model.base import ModelParamError
from optuna.exceptions import ExperimentalWarning
import warnings
warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna")
from mira.adata_interface.topic_model import logger as interfacelogger
from mira.adata_interface.core import logger as corelogger

from torch.utils.tensorboard import SummaryWriter
import torch
from mira.plots.pareto_front_plot import plot_intermediate_values, plot_pareto_front
from mira.topic_model.gp_sampler import GP, HyperbandPruner
from optuna.storages import RedisStorage


try:
    import redis
    redis_installed = True
except ImportError:
    redis_installed = False

class Redis(RedisStorage):

    def __init__(
        self,
        url = 'redis://localhost:6379',
        heartbeat_interval = 30,
        grace_period = None,
        failed_trial_callback = None,
    ):

        assert redis_installed, 'Must have redis-py installed to use this function. Run "$ conda install -c conda-forge redis-py"'

        if heartbeat_interval is not None and heartbeat_interval <= 0:
            raise ValueError("The value of `heartbeat_interval` should be a positive integer.")
        if grace_period is not None and grace_period <= 0:
            raise ValueError("The value of `grace_period` should be a positive integer.")

        self._url = url
        self._heartbeat_interval = heartbeat_interval
        self._grace_period = grace_period
        self._failed_trial_callback = failed_trial_callback
        
    @property
    def _redis(self):
        return redis.Redis.from_url(self._url)


class Locker:

    def __init__(self, study_name):
        self.study_name = study_name
        self.lockfile = os.path.join(
            '/tmp', (self.study_name + '_lockfile.lck').replace('/','_')
        )

        if not os.path.exists(self.lockfile):
            with open(self.lockfile, 'wb') as f:
                pass

    def __enter__ (self):
        self.fp = open(self.lockfile)
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__ (self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()

class DisableLogger:
    def __init__(self, logger):
        self.logger = logger
        self.level = logger.level

    def __enter__(self):
        self.logger.setLevel(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        self.logger.setLevel(self.level)


class FailureToImproveException(Exception):
    pass

def terminate_after_n_failures(study, trial, 
    n_failures=5,
    min_trials = 32):

    trials = [trial for trial in study.trials
            if trial.state == ts.PRUNED or trial.state == ts.COMPLETE]


    if len(trials) < min_trials or len(trials) < n_failures:
        return
    
    try:
        best_trial_no = max([trial.number for trial in study.best_trials])
    except ValueError:
        return 

    failures = 0
    for trial in trials:
        if trial.number > best_trial_no:
            failures+=1

    if failures > n_failures:
        raise FailureToImproveException()


def _format_params(params):

    def _format_param_value(value):

        if type(value) == float:
            return "{:.4f}".format(value)
        elif type(value) == str:
            return "'{}'".format(value)
        else:
            return str(value)

    return '{' + \
        ', '.join(["'{}': {}".format(param, _format_param_value(value))
        for param, value in params.items()]) \
    + '}'

def _get_batches_trained(trial):
        if not trial.state == ts.FAIL:
            return max(len(trial.intermediate_values), 0)
        else:
            return 'E'

def _get_trial_desc(study, trial):

    def get_score(trial):
        try:
            if trial.value is None:
                return np.inf

            return trial.value
        except AttributeError:
           pass

        return np.inf

    best_so_far = all(
        get_score(other_trial) > get_score(trial)
        for other_trial in study.trials[:trial.number]
    )

    if best_so_far:
        was_best = '\u25CF'
    else:
        was_best = ' '

    if trial.state == ts.COMPLETE:
        return ' #{:<3} | {} | completed, score: {:.4e} | {}'\
            .format(str(trial.number), was_best, get_score(trial), _format_params(trial.params))
    elif trial.state == ts.PRUNED:
        return ' #{:<3} | {} | pruned at step: {:<12} | {}'\
            .format(str(trial.number), ' ', str(_get_batches_trained(trial)), _format_params(trial.params))
    elif trial.state == ts.FAIL:
        return ' #{:<3} | {} | ERROR                        | {}'\
            .format(str(trial.number), ' ', str(trial.params))


def _log_progress(tuner, study, trial):
    
    logger.info(_get_trial_desc(study, trial))

    if trial.number in [t.number for t in study.best_trials]:
        logger.info('New best!')

def _clear_page():

    if NOTEBOOK_MODE:
        pass
        clear_output(wait=True)
    else:
        print('------------------------------------------------------')


def _print_topic_histogram(study):

    out = ''

    study_results = sorted([
        (trial_.params['num_topics'], '\u25A0', trial_.number)
        for trial_ in study.trials 
        if trial_.state in [ts.COMPLETE, ts.PRUNED, ts.FAIL] and 'num_topics' in trial_.params
    ], key = lambda x : x[0])

    current_num_modules = 0
    for trial_result in study_results:

        if trial_result[0] > current_num_modules:
            current_num_modules = trial_result[0]
            out += '\n{:>7} | '.format(str(current_num_modules))

        out += str(trial_result[1]) + ' '
    
    out += '\n\n'

    return out


def _print_running_trial(study, trial):

    try:
        progress = max(trial.intermediate_values.keys())
    except ValueError:
        progress = 0

    num_hashtags = min(34, int(34 * progress/study.user_attrs['max_resource']))
    return ' #{:<3} |'.format(trial.number) + '\u25A0'*num_hashtags + ' '*(34-num_hashtags) + '| '\
        + _format_params(trial.params)


def _print_study(tuner, study, trial):

    if study is None:
        raise ValueError('Cannot print study before running any trials.')

    out = ''

    try:
        out+= 'Trials finished: {} | Best trial: {} | Best score: {:.4e}\nPress ctrl+C,ctrl+C or esc,I+I,I+I in Jupyter notebook to stop early.'.format(
                str(len(study.trials)),
                str(study.best_trials[0].number) if len(study.best_trials) > 0 else 'None',
                study.best_value
            ) +  '\n\n'
    except ValueError:
        out += 'Trials finished {}'.format(str(len(study.trials))) + '\n\n'

    out += 'Tensorboard logidr: ' + os.path.join(tuner.tensorboard_logdir, tuner.study_name) + '\n'
    out += '#Topics | #Trials ' + '\n'
    out += _print_topic_histogram(study)

    out += 'Trial | Result (\u25CF = best so far)         | Params' + '\n'

    for trial in study.trials:
        if trial.state in [ts.COMPLETE, ts.PRUNED, ts.FAIL]:
            out += _get_trial_desc(study, trial) + '\n'

    if tuner.parallel:
        out += '\nRunning trials:\nTrial | Progress                         | Params' + '\n'
        for trial in study.trials:
            if trial.state == ts.RUNNING:
                out += _print_running_trial(study, trial) + '\n'

    out += '\n'

    return out


try:
    from IPython.display import clear_output
    clear_output(wait=True)
    NOTEBOOK_MODE = True
except ImportError:
    NOTEBOOK_MODE = False


import contextlib
import joblib

@contextlib.contextmanager
def joblib_print_callback(tuner):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TrialCompleteCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            s = str(tuner)
            _clear_page()
            print(s)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TrialCompleteCallback

    try:
        yield None
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback


class SpeedyTuner:

    @classmethod
    def optimize():
        pass

    objective = ['minimize']
    serial_dashboard = _print_study
    parallel_dashboard = lambda s,x,y : None #_log_progress

    @classmethod
    def train_test_split(cls, adata, 
        train_size = 0.8, seed = 2556,
        stratify = None):

        return train_test_split(adata, train_size = train_size, 
            random_state = seed, shuffle = True, stratify = stratify)

    
    def __init__(self,
        model,
        save_name,
        min_topics,
        max_topics,*,
        n_jobs = 1,
        max_trials = 64,
        min_trials = 24,
        stop_condition = 8,
        seed = 2556,
        tensorboard_logdir = 'runs',
        model_dir = 'models',
        storage = 'sqlite:///mira-tuning.db',
        rigor = 1,
        pruner = None,
        sampler = None,
        log_steps = False,
        log_every = 10,
    ):
        self.model = model
        self.n_jobs = n_jobs
        self.min_topics, self.max_topics = min_topics, max_topics
        self.iters = max_trials
        self.seed = seed
        self.study_name = save_name
        self.storage = storage
        self.pruner = pruner
        self.sampler = sampler
        self.tensorboard_logdir = tensorboard_logdir
        self.stop_condition = stop_condition
        self.min_trials = min_trials
        self.optimize_usefulness = False
        self.log_steps = log_steps
        self.log_every = log_every
        self.model_dir = model_dir
        self.rigor = rigor
        self.study = self.create_study()

    def __str__(self):
        return _print_study(self, self.study, None)

    def create_study(self):

        return optuna.create_study(
            directions = self.objective,
            study_name = self.study_name,
            storage = self.storage,
            load_if_exists= True,
        )

    @property
    def n_completed_trials(self):
        return len([
            trial for trial in self.study.trials
            if trial.state == ts.PRUNED or trial.state == ts.COMPLETE
        ])

    def _set_failed(self, t):
        if t.state == optuna.trial.TrialState.RUNNING:
            self.study.tell(t.number, None, ts.FAIL)

    def purge(self):

        # remove zombie trials
        for t in self.study.trials:
            self._set_failed(t)


    def get_stop_callback(self):
        if isinstance(self.stop_condition, int) and self.stop_condition > 0:
            return partial(terminate_after_n_failures, 
                n_failures = self.stop_condition,
                min_trials = self.min_trials)
        else:
            return lambda s,t : None
            
    @property
    def parallel(self):
        return self.n_jobs > 1

    def fit(self, train, test = None):
        '''
        Run Bayesian optimization scheme for topic model hyperparameters. This
        function launches multiple concurrent training processes to evaluate 
        hyperparameter combinations. All processes are launched on the same node.
        Evaluate the memory usage of a single MIRA topic model to determine 
        number of workers. 

        Parameters
        ----------
        adata : anndata.AnnData
            Anndata of expression or accessibility data.
            Cells must be labeled with test or train set membership using
            `tuner.train_test_split`.
        self.n_jobs : int, default = 1
            Number of tuning processes to launch.
        
        Returns
        -------
        study : optuna.study.Study
            Study object summarizing results of tuning iterations.

        '''

        assert isinstance(self.n_jobs, int) and self.n_jobs > 0

        self.study.set_user_attr('n_workers', self.n_jobs)
        self.study.set_user_attr('max_resource', self.model.num_epochs)

        if self.n_jobs > 5 and not isinstance(self.storage, Redis):
            raise ValueError('Can run maximum of 5 workers with default SQLite storage backend. For more processes, use Redis storage.')
        
        self.model.cpu()

        if self.n_completed_trials > 0:
            logger.warn('Resuming study with {} trials.'.format(self.n_completed_trials))

        num_running = len([t for t in self.study.trials if t.state == ts.RUNNING])
        if num_running > 0:
            logger.warn('{} running trials. Use "purge" method to stop trails if no longer running.'.format(num_running))

        if self.parallel and self.model.dataset_loader_workers > 0:
            logger.warn('Parrallelized tuning on one node with muliprocessed data loading is not permitted. Setting "dataset_loader_workers" to zero.')
            self.model.dataset_loader_workers = 0

        if not torch.cuda.is_available():
            logger.warn('GPU is not available, will not speed up training.')
        
        self.study = optuna.create_study(
            directions = self.objective,
            pruner = self.get_pruner(),
            sampler = self.get_tuner(),
            study_name = self.study_name,
            storage = self.storage,
            load_if_exists= True,
        )

        if test is None:
            logger.warn('No test set provided, splitting data into training and test sets (80/20).')

            if isinstance(train, str):
                raise ValueError(
                    'Automatic train/test splitting not supported for on-disk dataset. Make sure to save '
                    'training and testing partitions manually.'
                )

            train, test = self.train_test_split(train, seed = self.seed)
        
        lock = Locker(self.study_name)

        tune_func = partial(
                self._tune_step,
                train = train, 
                test = test,
                lock = lock,
        )

        remaining_trials = self.iters - self.n_completed_trials
        try:
            self.get_stop_callback()(self.study, None)
        except FailureToImproveException:
            remaining_trials = 0

        if remaining_trials > 0:

            try:

                with joblib_print_callback(self):
                    Parallel(n_jobs= self.n_jobs, verbose = 0)\
                        (delayed(tune_func)() for i in range(remaining_trials))

            except (KeyboardInterrupt, FailureToImproveException):
                pass

        self.model = self.fetch_best_weights()

        return self.model


    def get_stop_callback(self):
        if isinstance(self.stop_condition, int) and self.stop_condition > 0:
            return partial(terminate_after_n_failures, 
                n_failures = self.stop_condition,
                min_trials = self.min_trials)
        else:
            return self.stop_condition


    def get_pruner(self):

        if not self.pruner is None:
            return self.pruner

        elif self.rigor == 0:

            return optuna.pruners.SuccessiveHalvingPruner(
                min_resource=8,
                reduction_factor=2,
            )

        elif self.rigor > 0:

            assert self.model.num_epochs % 3 == 0, 'Number of epochs trained must be divisible by 3, e.g. 24, 36, etc.'
            return HyperbandPruner(
                min_resource = self.model.num_epochs//3,
                max_resource = self.model.num_epochs,
                reduction_factor = 2,
            )

        #    raise ValueError('Pruner {} is not an option'.format(str(self.pruner)))


    def get_tuner(self):

        if not self.sampler is None:
            return self.sampler

        elif self.rigor < 2:
            
            return GP(
                constant_liar = self.parallel,
                tau = 0.01,
                min_points = min(self.iters//2, 
                    max(min(10 * (self.rigor + 1), 10), self.n_jobs/(2 if isinstance(self.pruner, HyperbandPruner) else 1))
                ),
                num_candidates = 300,
                cl_function = np.max
            )

        elif self.rigor >= 2:

            params = optuna.samplers.TPESampler.hyperopt_parameters()
            params['n_startup_trials'] = max(10, self.n_jobs)

            return optuna.samplers.TPESampler(
                seed = self.seed,
                constant_liar=self.parallel,
                **params,
            )

        #else:
        #    raise ValueError('Sampler {} is not an option'.format(str(self.sampler)))


    def get_model_save_name(self, trial_number):

        return os.path.join(
            self.model_dir, self.study_name, '{}.pth'.format(str(trial_number))
        )


    def save_model(self, model, savename):

        savedir = os.path.dirname(savename)

        if not os.path.isdir(savedir):
            os.makedirs(savedir)

        model.save(savename)


    def run_trial(
            self, trial,*,
            train, 
            test,
            lock,
        ):
        
        must_prune = False
        self.study.sampler.reseed_rng()

        with lock:
            params = self.model.suggest_parameters(self, trial)

        #logger.info(
        #    'New trial #{} - num_topics: {}'.format(
        #        str(trial.number),
        #        str(params['num_topics'])
        #    )
        #)

        self.model.set_params(**params, seed = self.seed + trial.number)

        with SummaryWriter(
                log_dir=os.path.join(self.tensorboard_logdir, self.study_name, str(trial.number))
            ) as trial_writer:

            if not self.parallel:
                print('Evaluating: ' + _format_params(params))

            epoch_test_scores = []
            for epoch, train_loss in self.model._internal_fit(train, 
                    writer = trial_writer if self.log_steps else None,
                    log_every = self.log_every):
                
                try:
                    distortion, rate, metrics = self.model.distortion_rate_loss(test, bar = False, 
                                                        _beta_weight = self.model._last_anneal_factor)

                    trial_score = distortion + rate

                    if not self.parallel:
                        num_hashtags = int(25 * epoch/self.model.num_epochs)
                        print('\rProgress: ' + '|' + '\u25A0'*num_hashtags + ' '*(25-num_hashtags) + '|', end = '')

                    epoch_test_scores.append(trial_score)
                    trial_writer.add_scalar('holdout_distortion', distortion, epoch)
                    trial_writer.add_scalar('holdout_rate', rate, epoch)
                    trial_writer.add_scalar('holdout_loss', trial_score, epoch)
                    trial_writer.add_scalar('holdout_KL_weight', self.model._last_anneal_factor, epoch)

                    for metric_name, value in metrics.items():
                        trial_writer.add_scalar('holdout_' + metric_name, value, epoch)
                    
                    trial.report(min(epoch_test_scores[-self.model.num_epochs//6:]), epoch)

                    if trial.should_prune() and epoch < self.model.num_epochs:
                        must_prune = True
                        break

                except ValueError as err: # if evaluation fails for some reason
                    pass # just keep going unless training fails in outer loop
                         # this is implemented because sometimes early in training the
                         # estimation of test-set topics is unstable and can cause errors.
                         # In this case, it is better to just keep going with training,
                         # which will usually stabilize the model

            metrics = {
                    'number' : trial.number,
                    'epochs_trained' : epoch,
                    'distortion' : distortion,
                    'rate' : rate,
                    'trial_score' : trial_score,
                    **metrics,
            }

            params['study_name'] = self.study_name
            trial_writer.add_hparams(params, metrics)
            trial.set_user_attr("distortion", distortion)
            trial.set_user_attr("rate", rate)
            trial.set_user_attr("epochs_trained", epoch)
            

        if must_prune:
            raise optuna.TrialPruned()
        else:
            # save weights if best value so far
            #try:
            #    best_value = self.study.best_value
            #except ValueError:
            #    best_value = np.inf

            #if trial_score <= best_value:
            path = self.get_model_save_name(trial.number)
            trial.set_user_attr("path", path)
            self.save_model(self.model, path)

        return trial_score


    def _tune_step(self,*,
            train, test,
            lock,
        ):

        with DisableLogger(baselogger), DisableLogger(interfacelogger), DisableLogger(corelogger):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                interior_func = partial(
                    self.run_trial,
                    train = train,
                    test = test,
                    lock = lock,
                )

                try:
                    trial = _run_trial(self.study, interior_func, (ModelParamError, ValueError))
                finally:
                    gc.collect()

                #print_callback = self.serial_dashboard if not self.parallel else self.parallel_dashboard
                stop_callback = self.get_stop_callback()

                #print_callback(self.study, trial)
                stop_callback(self.study, trial)

        return trial


    def fetch_weights(self, trial_num):

        try:
            path = self.study.trials[trial_num].user_attrs['path']
        except IndexError:
            raise IndexError('Trial {} does not exist.'.format(trial_num))
        except KeyError:
            raise KeyError('No model saved for trial {}. Trial was either pruned or failed.'\
                    .format(str(trial_num)))

        return self.model.load(path)


    def fetch_best_weights(self):
        try:
            self.study.best_trial
        except ValueError:
            raise ValueError('No trials completed, cannot load best weights.')

        return self.fetch_weights(self.study.best_trial.number)


    def plot_intermediate_values(self,
        palette = 'Greys', 
        ax = None, figsize = (10,7),
        log_hue = False,
        hue = 'value',
        na_color = 'lightgrey',
        add_legend = True,
        vmax = None, vmin = None,
        ):
        
        return plot_intermediate_values(self.study.trials,
            palette = palette, ax = ax, figsize = figsize,
            hue = hue, add_legend = add_legend,
            vmax = vmax, vmin = vmin,
            na_color = na_color, log_hue = log_hue,
        )


    def plot_pareto_front(self,
        x = 'num_topics',
        y = 'elbo',
        color = 'distortion',
        ax = None, 
        figsize = (7,7),
        palette = 'Blues',
        na_color = 'lightgrey',
        size = 100,
        alpha = 0.8,
        add_legend = True,
    ):
        
        return plot_pareto_front(self.study.trials, x = x, y = y, color = color,
            ax = ax, figsize= figsize, palette = palette, na_color = na_color,
            size = size, alpha = alpha, add_legend = add_legend
        )


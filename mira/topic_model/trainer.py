
from pydoc_data.topics import topics
import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, train_test_split
from functools import partial
import os
import optuna
import mira.adata_interface.core as adi
import mira.adata_interface.topic_model as tmi
from optuna.trial import TrialState as ts
from joblib import Parallel, delayed
import time

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Setup the root logger.
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
from mira.topic_model.base import logger as baselogger
from mira.topic_model.base import ModelParamError
from optuna.exceptions import ExperimentalWarning
import warnings
warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna")
from mira.adata_interface.topic_model import logger as interfacelogger
from mira.adata_interface.core import logger as corelogger

from torch.utils.tensorboard import SummaryWriter
import torch
from mira.topic_model.pruner import MemoryPercentileStepPruner
from mira.plots.pareto_front_plot import plot_intermediate_values, plot_pareto_front
from mira.topic_model.gp_sampler import GP_RateDistortionSampler

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

    if len(study.trials) < min_trials or len(study.trials) < n_failures:
        return
    
    try:
        best_trial_no = max([trial.number for trial in study.best_trials])
    except ValueError:
        return 

    failures = 0
    for trial in study.trials[best_trial_no:]:
        if trial.state == ts.PRUNED or trial.state == ts.COMPLETE:
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
            return trial.values[0]
        except TypeError:
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
        return ' #{:<3} | {} | completed, score: {:.4e} | params: {}'\
            .format(str(trial.number), was_best, trial.values[0], _format_params(trial.params))
    elif trial.state == ts.PRUNED:
        return ' #{:<3} | {} | pruned at step: {:<12} | params: {}'\
            .format(str(trial.number), ' ', str(_get_batches_trained(trial)), _format_params(trial.params))
    elif trial.state == ts.FAIL:
        return ' #{:<3} | {} | ERROR                        | params: {}'\
            .format(str(trial.number), ' ', str(trial.params))


def _log_progress(tuner, study, trial,*, worker_number):
    
    logger.info('Worker {}: '.format(worker_number) + _get_trial_desc(study, trial))

    if trial.number in [t.number for t in study.best_trials]:
        logger.info('New best!')


def _print_topic_histogram(study):

    study_results = sorted([
        (trial_.params['num_topics'], '\u25A0', trial_.number)
        for trial_ in study.trials 
        if trial_.state in [ts.COMPLETE, ts.PRUNED, ts.FAIL] and 'num_topics' in trial_.params
    ], key = lambda x : x[0])

    current_num_modules = 0
    for trial_result in study_results:

        if trial_result[0] > current_num_modules:
            current_num_modules = trial_result[0]
            print('\n{:>7} | '.format(str(current_num_modules)), end = '')

        print(str(trial_result[1]), end = ' ')
    
    print('', end = '\n\n')


def _clear_page():

    if NOTEBOOK_MODE:
        #clear_output(wait=True)
        pass
    else:
        print('------------------------------------------------------')


def _print_study(tuner, study, trial):

    if study is None:
        raise ValueError('Cannot print study before running any trials.')
    
    _clear_page()

    try:
        print('Trials finished: {} | Best trial: {} | Best score: {:.4e}\nPress ctrl+C,ctrl+C or esc,I+I,I+I in Jupyter notebook to stop early.'.format(
        str(len(study.trials)),
        str(study.best_trials[0].number) if len(study.best_trials) > 0 else 'None',
        study.best_value
    ), end = '\n\n')
    except ValueError:
        print('Trials finished {}'.format(str(len(study.trials))), end = '\n\n')

    print('#Topics | #Trials ', end = '')
    _print_topic_histogram(study) 

    print('Trial Information: (\u25CF = best trial up to that point)')
    for trial in study.trials:
        if trial.state in [ts.COMPLETE, ts.PRUNED, ts.FAIL]:
            print(_get_trial_desc(study, trial))

    print('\n')
    

def print_study(study):
    _print_study(study, None)

try:
    from IPython.display import clear_output
    clear_output(wait=True)
    NOTEBOOK_MODE = True
except ImportError:
    NOTEBOOK_MODE = False


class TopicModelTuner:

    objective = 'minimize'
    serial_dashboard = _print_study
    parallel_dashboard = _log_progress

    @classmethod
    def load_study(cls, study_name, storage = 'sqlite:///mira-tuning.db'):

        return optuna.create_study(
            directions = cls.objective,
            study_name = study_name,
            storage = storage,
            load_if_exists= True,
        )

    def __init__(self,
        test_column = 'test_set_cells',
        min_topics = 5, max_topics = 55,
        min_epochs = 20, max_epochs = 40,
        min_dropout = 0.01, max_dropout = 0.15,
        batch_sizes = [32,64,128],
        cv = 5, iters = 64,
        seed = 2556,
        pruner = 'mira',
        sampler = 'tpe',
        tune_layers = True,
        tune_kl_strategy  = True,
        tensorboard_logdir = 'runs',
        storage = 'sqlite:///mira-tuning.db',
        model_survival_rate=1/4,
        usefulness_function = None,
        optimize_usefulness = False,
        stop_condition = None,
        min_trials = 32,*,
        save_name,
    ):
        '''
        Tune hyperparameters of the MIRA topic model using iterative Bayesian optimization.
        First, the optimization engine suggests a hyperparameter combination. The model
        is then trained with those parameters for 5 folds of cross validation (default option)
        to compute the performance of that model. If the parameter combination does not 
        meet the performance of previously-trained combinations, the trial is terminated early. 

        Depending on the size of your dataset, you may change the pruning and cross validation
        schemes to reduce training time. 

        The tuner returns an ``study`` object from the package `Optuna <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study>`_.
        The study may be reloaded to resume optimization later, or printed to review results.

        After tuning, the best models compete to minimize loss on a held-out set of cells.
        The winning model is returned as the final model of the dataset.

        Parallelized training is available to speed up tuning. Metrics and results from hyperparameter
        trials and model training are saved as tensorboard log files for diagnostic evaluation.

        .. image :: /_static/tensorboard_hparams.png
            :width: 1200

        .. note::

            Please refer to the :ref:`topic model tuning tutorial </notebooks/tutorial_topic_model_tuning_full.ipynb>`
            for instruction on parallelized training and tensorboard logging.

        Parameters
        ----------
        topic_model : mira.topics.ExpressionTopicModel or mira.topics.AccessibilityTopicModel
            Topic model to tune. The provided model should have columns specified
            to retrieve endogenous and exogenous features, and should have the learning
            rate configued by ``get_learning_rate_bounds``.
        study_name : str, default = None
            Table under which to save tuning results in *storage* SQLite table. 
        test_column : str, default = 'test_set'
            Column of anndata.obs marking cells held out for validation set. 
        min_topics : int, default = 5
            Minimum number of topics to try. Useful if approximate number of topics is known
            ahead of time.
        max_topics : int, default = 55
        min_dropout : float > 0, default = 0.01
            Minimum encoder dropout
        max_dropout : float>0, default = 0.15
        batch_sizes : list[int], default = [32,64,128]
            Batch sizes to try. Higher batch sizes (e.g. 256) increase training speed, 
            but seem to drastically reduce model quality.
        cv : int > 1 or subclass of ``sklearn.model_selection.BaseCrossValidator``
            If provided int, each trial is run for this many folds of cross validation.
            If provided sklearn CV object, this will be used instead of K-fold cross validation.
        iters : int > 1, default = 64
            Number of trials to run.
        study : None or `optuna.study.Study`
            If None, begin a new hyperparameter optimization routine. If given a study, 
            resume that study. If study is provided, `save_name` need not be set.
        seed : int > 0, default = 2556
            Random seed for K-fold cross validator and optuna optimizer.
        pruner : {"mira", "original", optuna.pruner.BasePruner, or None}, default = "mira"
            Default MIRA pruner is percentile pruner with memory. Pass None
            for no pruning, else pass any object inheriting from optuna.runer.BasePruner.
            "original" specifies pruner described in MIRA manuscript, which has
            since been improved.
        sampler : None or optuna.pruner.BaseSampler, default = None
            If None, uses MIRA's default choice of the TPE sampler.
        tune_kl_strategy : boolean, default = True
            Tune KL annealing strategy of model. Options are monotonic KL annealing,
            as implemented by the MIRA paper, and cyclic KL annealing.
        tune_layers : boolean, default = True,
            Tune the number of layers in the encoder model. Options are 2 or 3.
        tensorboard_logdir : str, default = 'runs',
            Directory in which to save tensorboard log files.
        storage : str, default = 'sqlite:///mira-tuning.db'
            SQLite database name.
        model_survival_rate : float >0, <1, default = 0.25
            The rate at which a model is expected to proceed through all
            folds of training, e.g., to "survive" the trial and be considered
            in the end model selection scheme.
        

        Returns
        -------
        study : optuna.study.Study

        Raises
        ------
        ValueError : If study not provided and `save_name` not set.

        Examples
        --------
        
        The tuning workflow, very briefly, is:

        .. code-block:: python

            >>> tuner = mira.topics.TopicModelTuner(
            ...     topic_model,
            ...     save_name = 'topic-model-study',
            ... )
            >>> tuner.train_test_split(data)
            >>> tuner.tune(data, n_workers = 1)
            >>> tuner.select_best_model(data)

        For large datasets, it may be useful to skip cross validation since the
        variance of the estimate of model performance should be lower. It may also
        be appropriate to limit the model to larger batch sizes.

        .. code-block::
        
            >>> tuner = mira.topics.TopicModelTuner(
            ...    topic_model,
            ...    save_name = 'study',
            ...    cv = sklearn.model_selection.ShuffleSplit(n_splits = 1, train_size = 0.8),
            ...    batch_sizes = [64,128],
            ... )

        '''
        self.test_column = test_column
        self.min_topics, self.max_topics = min_topics, max_topics
        self.min_epochs, self.max_epochs = min_epochs, max_epochs
        self.min_dropout, self.max_dropout = min_dropout, max_dropout
        self.batch_sizes = batch_sizes
        self.cv = cv
        self.iters = iters
        self.seed = seed
        self.study_name = save_name
        self.storage = storage
        self.pruner = pruner
        self.sampler = sampler
        self.tune_layers = tune_layers
        self.tune_kl_strategy = tune_kl_strategy
        self.tensorboard_logdir = tensorboard_logdir
        self.model_survival_rate = model_survival_rate
        self.usefulness_function = usefulness_function
        self.optimize_usefulness = optimize_usefulness
        self.stop_condition = stop_condition
        self.min_trials = min_trials

        self.study = self.create_study()


    @adi.wraps_modelfunc(adi.fetch_adata_shape, tmi.add_test_column, ['shape'])
    def train_test_split(self, train_size = 0.8, *, shape):
        """
        Randomly assigns cells to training and test sets given by proption `train_size`.
        Test set cells will not be used in hyperparameter optimization, and are
        reserved for validation and selection of the top-performing model.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData of expression or accessibility data
        train_size : float between 0 and 1, default = 0.8
            Proportion of cells to use for training set.

        Returns
        -------
        adata : anndata.AnnData
            `.obs['test_set']` : np.ndarray[boolean] of shape (n_cells,)
                Boolean variable, whether cell is in test set.

        """

        assert(isinstance(train_size, float) and train_size > 0 and train_size < 1)
        num_samples = shape[0]
        assert(num_samples > 0), 'Adata must have length > 0.'
        np.random.seed(self.seed)
        return self.test_column, np.random.rand(num_samples) > train_size


    def suggest_parameters(self, trial):

        params = dict(
            num_topics = trial.suggest_int('num_topics', self.min_topics, self.max_topics, log=True),
            batch_size = trial.suggest_categorical('batch_size', self.batch_sizes),
            encoder_dropout = trial.suggest_float('encoder_dropout', self.min_dropout, self.max_dropout, log = True),
            num_epochs = trial.suggest_int('num_epochs', self.min_epochs, self.max_epochs, log = True),
            #beta = trial.suggest_float('beta', 0.90, 0.97, log = True),
        )

        if self.tune_kl_strategy:
            params['kl_strategy'] = trial.suggest_categorical('kl_strategy', ['monotonic','cyclic'])

        if self.tune_layers:
            params['num_layers'] = trial.suggest_categorical('num_layers', [2,3])

        return params


    def get_trial_score(self, cv_metrics):

        metrics = self.transpose_metrics(cv_metrics)

        if self.optimize_usefulness:
            return -np.mean(metrics['usefulness'])
        else:
            return np.mean(metrics['loss'])


    @staticmethod
    def transpose_metrics(cv_metrics):

        metric_summaries = {}
        for fold, metrics in enumerate(cv_metrics):
            for metric, value in metrics.items():
                if not metric in metric_summaries:
                    metric_summaries[metric] = []
                
                metric_summaries[metric].append(value)

        return metric_summaries


    def run_trial(
            self,
            trial,
            parallel,*,
            data,
            model,
            worker_number,
        ):
            
        params = self.suggest_parameters(trial)

        model.set_params(**params, seed = self.seed + trial.number)

        num_splits = self.cv.get_n_splits(data)
        cv_metrics = []
        must_prune = False

        with SummaryWriter(
                log_dir=os.path.join(self.tensorboard_logdir, self.study_name, str(trial.number))
            ) as trial_writer:
            
            if not parallel:
                print('Evaluating: ' + _format_params(params))

            for step, (train_idx, test_idx) in enumerate(self.cv.split(data)):

                train_counts, test_counts = data[train_idx], data[test_idx]
                
                with SummaryWriter(log_dir=os.path.join(self.tensorboard_logdir, self.study_name, str(trial.number), str(step))) as model_writer:
                    for epoch, loss in model._internal_fit(train_counts, writer = model_writer):
                        
                        if not parallel:
                            num_hashtags = int(10 * epoch/model.num_epochs)
                            print('\rProgress: ' + '|##########'*step + '|' + '#'*num_hashtags + ' '*(10-num_hashtags) + '|' + '          |'*(num_splits-step-1),
                                end = '')

                distortion, rate, loss = model.distortion_rate_loss(test_counts)
                metrics = {'rate' : rate, 'distortion' : distortion, 'loss' : loss}

                if not self.usefulness_function is None:
                    usefulness, eval_metrics = self.usefulness_function(
                                model, train_counts, test_counts, data
                            )
                    metrics.update({'usefulness' : usefulness, **eval_metrics})

                cv_metrics.append(metrics)

                try:
                    trial.report(self.get_trial_score(cv_metrics), step)
                        
                    if trial.should_prune() and step + 1 < num_splits:
                        must_prune = True
                        break
                except NotImplementedError:
                    pass

            metric_summaries = self.transpose_metrics(cv_metrics)
            metric_summaries = {metric : np.mean(values) for metric, values in metric_summaries.items()}

            metrics = {
                        'number' : trial.number,
                        **metric_summaries,
                        **{'cv_{}_{}'.format(str(fold), metric) : value 
                            for fold, metrics in enumerate(cv_metrics)
                            for metric, value in metrics.items()
                        },
                        'test_score' : 1.,
                        'worker_number' : worker_number,
                        'num_folds_tested' : len(cv_metrics),
                    }

            trial_score = self.get_trial_score(cv_metrics)

            if isinstance(trial_score, float):
                metrics['trial_score'] = trial_score
            else:
                for i, val in enumerate(trial_score):
                    metrics['trial_score_' + str(i)] = val

            params['study_name'] = self.study_name
            trial_writer.add_hparams(params, metrics)
            trial.set_user_attr("distortion", metric_summaries['distortion'])
            trial.set_user_attr("rate", metric_summaries['rate'])
            
            if 'usefulness' in metric_summaries:
                trial.set_user_attr('usefulness', metric_summaries['usefulness'])

            if must_prune:
                raise optuna.TrialPruned()

        return trial_score


    def print(self):
        '''
        Print study results from tuning to screen. Useful to see results of 
        tuning when reloading study.
        '''
        self.serial_dashboard(self.study, None)


    def create_study(self):

        return optuna.create_study(
            directions = self.objective,
            study_name = self.study_name,
            storage = self.storage,
            load_if_exists= True,
        )


    def get_pruner(self,*,n_workers):

        num_splits = self.cv.get_n_splits()

        if self.pruner == 'mira':

            assert isinstance(self.model_survival_rate, float) and self.model_survival_rate > 0. and self.model_survival_rate < 1., 'Model success rate must be float between 0 and 1'
            prune_rate = self.model_survival_rate**(1/(num_splits-1))

            num_samples_before_prune = max(int(np.ceil(1/(1-prune_rate))), 5)
            
            return MemoryPercentileStepPruner(
                memory_length = num_samples_before_prune,
                percentile = 100*prune_rate,
                n_startup_trials = max(n_workers, num_samples_before_prune)
            )

        elif self.pruner == 'original':

            return MemoryPercentileStepPruner(
                memory_length = self.iters,
                percentile = 100/3,
                n_startup_trials = 1,
                interval_steps = 2,
            )
        
        elif isinstance(self.pruner, optuna.pruners.BasePruner) or self.pruner is None:
            return self.pruner
        else:
            raise ValueError('Pruner {} is not an option'.format(str(self.pruner)))


    def get_tuner(self, worker_number = 0, parallel = False):
        if isinstance(self.sampler, optuna.samplers.BaseSampler):
            self.sampler._rng = np.random.RandomState(self.seed + worker_number)
            return self.sampler

        
        elif self.sampler == 'tpe':

            params = optuna.samplers.TPESampler.hyperopt_parameters()
            params['n_startup_trials'] = 10

            return optuna.samplers.TPESampler(
                seed = self.seed + worker_number,
                constant_liar=parallel,
                **params,
            )
        else:
            raise ValueError('Sampler {} is not an option'.format(str(self.sampler)))


    @property
    def n_completed_trials(self):
        return len([
            trial for trial in self.study.trials
            if trial.state == ts.PRUNED or trial.state == ts.COMPLETE
        ])

    def tune(self, model, adata, n_workers = 1):
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
        n_workers : int, default = 1
            Number of tuning processes to launch.
        
        Returns
        -------
        study : optuna.study.Study
            Study object summarizing results of tuning iterations.

        '''

        assert isinstance(n_workers, int) and n_workers > 0
        
        if self.optimize_usefulness:
            assert not self.usefulness_function is None

        model.cpu()

        if self.n_completed_trials > 0:
            logger.warn('Resuming study with {} trials.'.format(self.n_completed_trials))

        tune_func = partial(
                self._tune, parallel = n_workers>1, n_workers = n_workers,
                            model = model, data = adata,
            )

        remaining_trials = self.iters - self.n_completed_trials
        if remaining_trials > 0:

            worker_iters = np.array([remaining_trials//n_workers]*n_workers)
            remaining_iters = remaining_trials % n_workers
            if remaining_iters > 0:
                worker_iters[np.arange(remaining_iters)]+=1

            worker_iters = worker_iters[worker_iters > 0]

            Parallel(n_jobs=len(worker_iters), verbose = 0)\
                (delayed(tune_func)(worker_number = i, iters = n_iters) for i, n_iters in enumerate(worker_iters))

            self.print()
            time.sleep(0.05)

        return self.study


    def get_stop_callback(self):
        if isinstance(self.stop_condition, int) and self.stop_condition > 0:
            return partial(terminate_after_n_failures, 
                n_failures = self.stop_condition,
                min_trials = self.min_trials)
        else:
            return self.stop_condition


    def get_cv(self):

        if isinstance(self.cv, int):
            return KFold(self.cv, random_state = self.seed, shuffle= True)
        else:
            return self.cv


    def get_train_data(self, data):
        return tmi.fetch_split_train_test(self, data)['train_data']


    def _tune(self, worker_number = 0, iters = 5, n_workers = 1, parallel = False,
        *,model, data):

        train_data = self.get_train_data(data)

        self.cv = self.get_cv()

        if parallel and model.dataset_loader_workers > 0:
            logger.warn('Worker {}: Parrallelized tuning on one node with muliprocessed data loading is not permitted. Setting "dataset_loader_workers" to zero.'.format(str(worker_number)))
            model.dataset_loader_workers = 0
        
        self.study = optuna.create_study(
            directions = self.objective,
            pruner = self.get_pruner(n_workers = n_workers, model = model),
            sampler = self.get_tuner(worker_number = worker_number, parallel=parallel,
                n_workers = n_workers),
            study_name = self.study_name,
            storage = self.storage,
            load_if_exists= True,
        )
        
        trial_func = partial(
            self.run_trial,
            parallel = parallel,
            data = train_data,
            model = model,
            worker_number = worker_number,
        )

        if not torch.cuda.is_available():
            logger.warn('Worker {}: GPU is not available, will not speed up training.'.format(str(worker_number)))

        print_callback = self.serial_dashboard if not parallel else partial(self.parallel_dashboard, worker_number = worker_number)
        stop_callback = self.get_stop_callback()

        with DisableLogger(baselogger), DisableLogger(interfacelogger), DisableLogger(corelogger):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            
                np.random.seed(self.seed + worker_number)

                logger.info('Worker {}: Running {} trials.'\
                        .format(str(worker_number), str(iters))
                )

                try:
                    self.study.optimize(
                        trial_func, n_trials = iters, 
                        callbacks = [print_callback] + ([stop_callback] if not stop_callback is None else []),
                        catch = (ModelParamError,),
                    )
                except (KeyboardInterrupt, FailureToImproveException):
                    pass

        if not parallel:
            self.print()

        return self.study


    def get_tuning_results(self):
        '''
        Returns results from each tuning trail.
        '''

        try:
            self.study
        except AttributeError:
            raise Exception('User must run "tune_hyperparameters" before running this function')

        return self.study.trials


    def get_best_trials(self, top_n_trials = 5):
        '''
        Return results from best-performing tuning trials.

        Parameters
        ----------
        top_n_trials : int > 0, default = 5

        Returns
        -------
        top_trials : list
        '''
        
        assert(isinstance(top_n_trials, int) and top_n_trials > 0)
        try:
            self.study
        except AttributeError:
            raise Exception('User must run "tune_hyperparameters" before running this function')
        
        valid_trials = [
          trial for trial in self.study.trials if trial.state == ts.COMPLETE
        ]
        
        sorted_trials = sorted(valid_trials, key = lambda trial : trial.values[-1])

        return sorted_trials[:top_n_trials]

    def get_best_params(self, top_n_trials = 5):
        '''
        Return hyperparameters from best-performing tuning trials.

        Parameters
        ----------
        top_n_trials : int > 0, default = 5

        Returns
        -------
        top_parameters : list[dict]
            of format [{parameter combination 1}, ..., {parameter combination N}]
        '''

        return [
            {**trial.params, 'seed' : self.seed + trial.number} 
            for trial in self.get_best_trials(top_n_trials)
        ]


    @adi.wraps_modelfunc(tmi.fetch_split_train_test, adi.return_output, ['all_data', 'train_data', 'test_data'])
    def select_best_model(self, top_n_trials = 5, color_col = 'leiden', record_umaps = False,*,
        all_data, train_data, test_data):
        '''
        Retrain best parameter combinations on all training data, then 
        compare validation data performance. Best-performing model on test set 
        returned as "official" topic model representation of dataset.

        Parameters
        ----------
        adata : anndata.AnnData
            Anndata of expression or accessibility data.
            Cells must be labeled with test or train set membership using
            `tuner.train_test_split`.
        top_n_trials : int > 0, default = 5
            Number of top parameter combinations to test on validation data.
        record_umaps : boolean, default = False,
            Record ILR-transformed topics as cell embeddings in tensorboard
            embedding projector. Enables exploration of manifold for each
            trained topic model.
        color_col : str, default = 'leiden'
            With which column to color cells in tensorboard embedding projector.
            If column is not present in *adata*, skips coloring.

        Returns
        -------
        best_model : sublcass of mira.topic_model.BaseModel
            Best-performing model chosen using validation set.
            
        '''

        scores = []
        best_params = self.get_best_params(top_n_trials)

        for trial_num, params in best_params.items():
            logger.info('Training model with parameters: ' + _format_params(params))

            try:
            
                with SummaryWriter(log_dir=os.path.join(self.tensorboard_logdir, self.study_name, 'eval_' + str(trial_num))) as writer:

                    with DisableLogger(baselogger), DisableLogger(interfacelogger), DisableLogger(corelogger):
                    
                        test_score = self.model.set_params(**params, seed = self.seed + trial_num)\
                            .fit(train_data, writer = writer)\
                            .score(test_data)

                        scores.append(test_score)

                        logger.info('Score: {:.5e}'.format(test_score))
                        writer.add_hparams(params, {'test_score' : test_score})

                        if record_umaps:

                            self.model.predict(all_data)
                            self.model.get_umap_features(all_data)

                            try:
                                metadata = all_data.obs_vector(color_col)
                            except KeyError:
                                metadata = None

                            writer.add_embedding(
                                all_data.obsm['X_umap_features'], metadata= metadata,
                            )


            except (RuntimeError, ValueError) as err:
                logger.error('Error occured while training, skipping model.')
                scores.append(np.inf)

        final_choice = best_params[list(best_params.keys())[np.argmin(scores)]]
        logger.info('Set parameters to best combination: ' + _format_params(final_choice))
        self.model.set_params(**final_choice)
        
        logger.info('Training model with all data.')

        with SummaryWriter(log_dir= os.path.join(self.tensorboard_logdir, self.study.study_name, 'best_model')) as writer:
            self.model.fit(all_data, writer = writer)

        return self.model


class SpeedyTuner(TopicModelTuner):

    objective = ['minimize']
    serial_dashboard = _print_study
    parallel_dashboard = _log_progress

    @classmethod
    def train_test_split(cls, adata, 
        train_size = 0.8, seed = 2556,
        stratify = None):

        return train_test_split(adata, train_size = train_size, 
            random_state = seed, shuffle = True, stratify = stratify)

    
    def __init__(self,
        max_trials = 64,
        min_trials = 24,
        stop_condition = 8,
        seed = 2556,
        tensorboard_logdir = 'runs',
        model_dir = 'models',
        storage = 'sqlite:///mira-tuning.db',
        pruner = 'sha',
        sampler = 'gp',
        log_steps = False,
        log_every = 10,
        initial_topic_array = False,
        tune_topics_only = True,*,
        save_name,
        min_topics,
        max_topics,
    ):
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
        self.initial_topic_array = initial_topic_array
        self.optimize_usefulness = False
        self.log_steps = log_steps
        self.log_every = log_every
        self.model_dir = model_dir
        self.tune_topics_only = tune_topics_only
        self.study = self.create_study()

    @property
    def n_completed_trials(self):
        return len([
            trial for trial in self.study.trials
            if trial.state == ts.PRUNED or trial.state == ts.COMPLETE
        ])

    def tune(self, model, train, test, n_workers = 1):
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
        n_workers : int, default = 1
            Number of tuning processes to launch.
        
        Returns
        -------
        study : optuna.study.Study
            Study object summarizing results of tuning iterations.

        '''

        self.study.set_user_attr('n_workers', n_workers)

        # remove zombie trials
        for t in self.study.trials:
            if t.state == optuna.trial.TrialState.RUNNING:
                self.study.tell(t.number, None, ts.FAIL)

        assert isinstance(n_workers, int) and n_workers > 0

        if self.initial_topic_array and len(self.study.trials) == 0:
            self.enqueue_initial_trials(max(5, n_workers))
        
        model.cpu()

        if self.n_completed_trials > 0:
            logger.warn('Resuming study with {} trials.'.format(self.n_completed_trials))

        tune_func = partial(
                self._tune, parallel = n_workers>1, n_workers = n_workers,
                            model = model, train = train, test = test,
            )

        remaining_trials = self.iters - self.n_completed_trials
        try:
            self.get_stop_callback()(self.study, None)
        except FailureToImproveException:
            remaining_trials = 0

        if remaining_trials > 0:

            worker_iters = np.array([remaining_trials//n_workers]*n_workers)
            remaining_iters = remaining_trials % n_workers
            if remaining_iters > 0:
                worker_iters[np.arange(remaining_iters)]+=1

            worker_iters = worker_iters[worker_iters > 0]

            Parallel(n_jobs=len(worker_iters), verbose = 0)\
                (delayed(tune_func)(worker_number = i, iters = n_iters) for i, n_iters in enumerate(worker_iters))

            self.print()
            time.sleep(0.05)

        return self.study


    def _tune(self, worker_number = 0, iters = 5, n_workers = 1, parallel = False,
        *,model, train, test):

        time.sleep(0.5 * worker_number)

        self.cv = self.get_cv()

        if parallel and model.dataset_loader_workers > 0:
            logger.warn('Worker {}: Parrallelized tuning on one node with muliprocessed data loading is not permitted. Setting "dataset_loader_workers" to zero.'.format(str(worker_number)))
            model.dataset_loader_workers = 0
        
        self.study = optuna.create_study(
            directions = self.objective,
            pruner = self.get_pruner(n_workers = n_workers, model = model),
            sampler = self.get_tuner(worker_number = worker_number, parallel=parallel,
                n_workers = n_workers),
            study_name = self.study_name,
            storage = self.storage,
            load_if_exists= True,
        )

        self.study.set_user_attr("n_workers", n_workers)
        
        trial_func = partial(
            self.run_trial,
            parallel = parallel,
            train = train,
            test = test,
            model = model,
            worker_number = worker_number,
        )

        if not torch.cuda.is_available():
            logger.warn('Worker {}: GPU is not available, will not speed up training.'.format(str(worker_number)))

        print_callback = self.serial_dashboard if not parallel else partial(self.parallel_dashboard, worker_number = worker_number)
        stop_callback = self.get_stop_callback()

        with DisableLogger(baselogger), DisableLogger(interfacelogger), DisableLogger(corelogger):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            
                np.random.seed(self.seed + worker_number)

                logger.info('Worker {}: Running {} trials.'\
                        .format(str(worker_number), str(iters))
                )
                
                try:
                    self.study.optimize(
                        trial_func, n_trials = iters, 
                        callbacks = [print_callback] + ([stop_callback] if not stop_callback is None else []),
                        catch = (ModelParamError, ValueError),
                    )
                except (KeyboardInterrupt, FailureToImproveException):
                    pass

        if not parallel:
            self.print()

        return self.study


    def get_stop_callback(self):
        if isinstance(self.stop_condition, int) and self.stop_condition > 0:
            return partial(terminate_after_n_failures, 
                n_failures = self.stop_condition,
                min_trials = self.min_trials)
        else:
            return self.stop_condition

    def get_cv(self):
        return None

    def get_train_data(self, data):
        return data


    def suggest_parameters(self, trial):

        params = dict(        
            num_topics = trial.suggest_int('num_topics', self.min_topics, 
            self.max_topics, log=True),
        )

        if not self.tune_topics_only:
            params.update(dict(
                encoder_dropout = trial.suggest_float('encoder_dropout', 0.0001, 0.1, log = True),
                num_layers = trial.suggest_categorical('num_layers', (2,3,4,)),
                hidden = trial.suggest_categorical('hidden', [64, 128, 256]),
                min_momentum = trial.suggest_float('min_momentum', 0.8, 0.9, log = True),
                max_momentum = trial.suggest_float('max_momentum', 0.91, 0.97, log = True),
                weight_decay = trial.suggest_float('weight_decay', 0.00001, 0.01, log = True),
                decoder_dropout = trial.suggest_float('decoder_dropout', 0.15, 0.25)
            ))

        return params

    def get_pruner(self,*, model, n_workers):

        if self.pruner == 'sha':
            return optuna.pruners.SuccessiveHalvingPruner(
                min_resource = model._get_min_resources(),
                reduction_factor = 2,
            )
        elif isinstance(self.pruner, optuna.pruners.BasePruner) or self.pruner is None:
            return self.pruner
        else:
            raise ValueError('Pruner {} is not an option'.format(str(self.pruner)))


    def get_tuner(self, worker_number = 0, parallel = False,
            n_workers = 1):

        if isinstance(self.sampler, optuna.samplers.BaseSampler):
            self.sampler._rng = np.random.RandomState(self.seed + worker_number)
            return self.sampler

        elif self.sampler == 'gp':
            return GP_RateDistortionSampler(
                seed = self.seed + worker_number,
                constant_liar = parallel,
                tau = 10,
            )

        elif self.sampler == 'tpe':

            params = optuna.samplers.TPESampler.hyperopt_parameters()
            params['n_startup_trials'] = 5

            return optuna.samplers.TPESampler(
                seed = self.seed + worker_number,
                constant_liar=parallel,
                **params,
            )
        else:
            raise ValueError('Sampler {} is not an option'.format(str(self.sampler)))


    def enqueue_initial_trials(self, n_trials = 5):

        for num_topics in np.geomspace(self.min_topics, self.max_topics, n_trials + 2)[1:-1]:
            self.study.enqueue_trial(
                {
                    'num_topics' : int(num_topics),
                    'reconstruction_weight' : 0.5
                }
            )


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
            self,
            trial,
            parallel,*,
            model,
            train, test,
            worker_number,
        ):
        
        must_prune = False
        params = self.suggest_parameters(trial)

        logger.info(
            'New trial - num_topics: {}'.format(str(params['num_topics']))
        )

        model.set_params(**params, seed = self.seed + trial.number)

        with SummaryWriter(
                log_dir=os.path.join(self.tensorboard_logdir, self.study_name, str(trial.number))
            ) as trial_writer:

            if not parallel:
                print('Evaluating: ' + _format_params(params))

            epoch_test_scores = []
            for epoch, train_loss in model._internal_fit(train, 
                    writer = trial_writer if self.log_steps else None,
                    log_every = self.log_every):
                
                try:
                    distortion, rate, metrics = model.distortion_rate_loss(test, bar = False, 
                                                        _beta_weight = model._last_anneal_factor)

                    trial_score = distortion + rate

                    if not parallel:
                        num_hashtags = int(25 * epoch/model.num_epochs)
                        print('\rProgress: ' + '|' + '\u25A0'*num_hashtags + ' '*(25-num_hashtags) + '|', end = '')

                    epoch_test_scores.append(trial_score)
                    trial_writer.add_scalar('holdout_distortion', distortion, epoch)
                    trial_writer.add_scalar('holdout_rate', rate, epoch)
                    trial_writer.add_scalar('holdout_loss', trial_score, epoch)
                    trial_writer.add_scalar('holdout_KL_weight', model._last_anneal_factor, epoch)

                    for metric_name, value in metrics.items():
                        trial_writer.add_scalar('holdout_' + metric_name, value, epoch)
                    
                    trial.report(min(epoch_test_scores[-model.num_epochs//6:]), epoch)

                    if trial.should_prune() and epoch < model.num_epochs:
                        must_prune = True
                        break
                except ValueError: # if evaluation fails for some reason
                    pass

            #distortion, rate, metrics = model.distortion_rate_loss(test, bar = False)
            #trial_score = distortion + rate

            metrics = {
                    'number' : trial.number,
                    'epochs_trained' : epoch,
                    'distortion' : distortion,
                    'rate' : rate,
                    'trial_score' : trial_score,
                    **metrics,
                    'worker_number' : worker_number,
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
            self.save_model(model, path)

        return trial_score


    def fetch_weights(self, model, trial_num):

        try:
            path = self.study.trials[trial_num].user_attrs['path']
        except IndexError:
            raise IndexError('Trial {} does not exist.'.format(trial_num))
        except KeyError:
            raise KeyError('No model saved for trial {}. Trial was either pruned or failed.'\
                    .format(str(trial_num)))

        return model.load(path)


    def fetch_best_weights(self, model):
        return self.fetch_weights(model, self.study.best_trial.number)


    def plot_intermediate_values(self,
        palette = 'Greys', 
        ax = None, figsize = (10,7),
        linecolor = 'black',
        hue = 'value'):
        
        return plot_intermediate_values(self.study.trials,
            palette = palette, ax = ax, figsize = figsize,
            linecolor = linecolor, hue = hue,
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


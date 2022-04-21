
import numpy as np
from sklearn.model_selection import KFold, BaseCrossValidator
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


class DisableLogger:
    def __init__(self, logger):
        self.logger = logger
        self.level = logger.level

    def __enter__(self):
        self.logger.setLevel(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        self.logger.setLevel(self.level)


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

def _get_trial_desc(trial):

        if trial.state == ts.COMPLETE:
            return 'Trial #{:<3} | completed, score: {:.4e} | params: {}'.format(str(trial.number), trial.values[-1], _format_params(trial.params))
        elif trial.state == ts.PRUNED:
            return 'Trial #{:<3} | pruned at step: {:<12} | params: {}'.format(str(trial.number), str(_get_batches_trained(trial)), _format_params(trial.params))
        elif trial.state == ts.FAIL:
            return 'Trial #{:<3} | ERROR                        | params: {}'\
                .format(str(trial.number), str(trial.params))


def _log_progress(study, trial,*, num_trials, worker_number):
    
    logger.info('Worker {}: '.format(worker_number) + _get_trial_desc(trial))

    if study.best_trial.number == trial.number:
        logger.info('New best!')


def _print_study(study, trial):

    if study is None:
        raise ValueError('Cannot print study before running any trials.')

    if NOTEBOOK_MODE:
        clear_output(wait=True)
    else:
        print('------------------------------------------------------')

    try:
        print('Trials finished: {} | Best trial: {} | Best score: {:.4e}\nPress ctrl+C,ctrl+C or esc,I+I,I+I in Jupyter notebook to stop early.'.format(
        str(len(study.trials)),
        str(study.best_trial.number),
        study.best_value
    ), end = '\n\n')
    except ValueError:
        print('Trials finished {}'.format(str(len(study.trials))), end = '\n\n')        

    print('#Topics | Trials (number is #folds tested)', end = '')

    study_results = sorted([
        (trial_.params['num_topics'], _get_batches_trained(trial_), trial_.number)
        for trial_ in study.trials
    ], key = lambda x : x[0])

    current_num_modules = 0
    for trial_result in study_results:

        if trial_result[0] > current_num_modules:
            current_num_modules = trial_result[0]
            print('\n{:>7} | '.format(str(current_num_modules)), end = '')

        print(str(trial_result[1]), end = ' ')
        #print(trial_result[1], ('*' trial_result[2] == study.best_trial.number else ''), end = '')
    
    print('', end = '\n\n')
    print('Trial Information:')
    for trial in study.trials:
        print(_get_trial_desc(trial))

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

    @classmethod
    def load_study(cls, study_name, storage = 'sqlite:///mira-tuning.db'):

        return optuna.create_study(
            direction = 'minimize',
            study_name = study_name,
            storage = storage,
            load_if_exists= True,
        )

    def __init__(self,
        topic_model,
        test_column = None,
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
        model_survival_rate=1/4,*,
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
        self.model = topic_model
        self.test_column = test_column or 'test_set'
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


    @adi.wraps_modelfunc(adi.fetch_adata_shape, tmi.add_test_column, ['shape'])
    def train_test_split(self, train_size = 0.8, *, shape):
        '''
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

        '''

        assert(isinstance(train_size, float) and train_size > 0 and train_size < 1)
        num_samples = shape[0]
        assert(num_samples > 0), 'Adata must have length > 0.'
        np.random.seed(self.seed)
        return self.test_column, np.random.rand(num_samples) > train_size

    @staticmethod
    def _trial(
            trial,
            base_seed = 0,
            parallel = False,*,
            tensorboard_logdir,
            model, data, cv, batch_sizes,
            min_topics, max_topics,
            min_dropout, max_dropout,
            min_epochs, max_epochs,
            tune_kl_strategy,
            tune_layers,
            study_name,
        ):
            
        params = dict(
            num_topics = trial.suggest_int('num_topics', min_topics, max_topics, log=True),
            batch_size = trial.suggest_categorical('batch_size', batch_sizes),
            encoder_dropout = trial.suggest_float('encoder_dropout', min_dropout, max_dropout),
            num_epochs = trial.suggest_int('num_epochs', min_epochs, max_epochs, log = True),
            beta = trial.suggest_float('beta', 0.90, 0.99, log = True),
        )

        if tune_kl_strategy:
            params['kl_strategy'] = trial.suggest_categorical('kl_strategy', ['monotonic','cyclic'])

        if tune_layers:
            params['num_layers'] = trial.suggest_categorical('num_layers', [2,3])

        domains = {
            'kl_strategy' : ['monotonic','cyclic'],
            'batch_size' : batch_sizes,
            'num_layers' : [2,3],
        }

        model.set_params(**params, seed = base_seed + trial.number)
        cv_scores = []

        num_splits = cv.get_n_splits(data)
        cv_scores = []
        must_prune = False

        with SummaryWriter(log_dir=os.path.join(tensorboard_logdir, study_name, str(trial.number))) as trial_writer:
            
            if not parallel:
                print('Evaluating: ' + _format_params(params))

            for step, (train_idx, test_idx) in enumerate(cv.split(data)):

                train_counts, test_counts = data[train_idx], data[test_idx]
                
                with SummaryWriter(log_dir=os.path.join(tensorboard_logdir, study_name, str(trial.number), str(step))) as model_writer:
                    for epoch, loss in model._internal_fit(train_counts, writer = model_writer):
                        
                        if not parallel:
                            num_hashtags = int(10 * epoch/params['num_epochs'])
                            print('\rProgress: ' + '|##########'*step + '|' + '#'*num_hashtags + ' '*(10-num_hashtags) + '|' + '          |'*(num_splits-step-1),
                                end = '')

                cv_scores.append(
                    model.score(test_counts)
                )
                
                trial.report(np.mean(cv_scores), step)
                    
                if trial.should_prune() and step + 1 < num_splits:
                    must_prune = True
                    break

            trial_score = np.mean(cv_scores)

            metrics = {**{'cv_{}_score'.format(str(i)) : cv_score for i, cv_score in enumerate(cv_scores)}, 
                                'trial_score' : trial_score,
                                'test_score' : 1.,
                                'num_folds_tested' : len(cv_scores),
                                'number' : trial.number}

            trial_writer.add_hparams(params, metrics, domains)

            if must_prune:
                raise optuna.TrialPruned()

        return trial_score

    def print(self):
        '''
        Print study results from tuning to screen. Useful to see results of 
        tuning when reloading study.
        '''
        _print_study(self.study, None)


    def get_pruner(self,*,n_workers, num_splits):

        if self.pruner == 'mira':

            assert isinstance(self.model_survival_rate, float) and self.model_survival_rate > 0. and self.model_survival_rate < 1., 'Model success rate must be float between 0 and 1'
            prune_rate = self.model_survival_rate**(1/(num_splits-1))

            num_samples_before_prune = int(np.ceil(1/(1-prune_rate)))
            
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
            self.sampler.seed+=worker_number
            return self.sampler
        elif self.sampler == 'tpe':
            return optuna.samplers.TPESampler(
                seed = self.seed + worker_number,
                constant_liar=parallel,
            )
        else:
            raise ValueError('Sampler {} is not an option'.format(str(self.sampler)))


    def tune(self, adata, n_workers = 1):
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

        self.model.cpu()

        self.study = optuna.create_study(
            direction = 'minimize',
            study_name = self.study_name,
            storage = self.storage,
            load_if_exists= True,
        )

        if len(self.study.trials) > 0:
            logger.warn('Resuming study with {} trials.'.format(len(self.study.trials)))

        tune_func = partial(
                self._tune, adata, parallel = n_workers>1, n_workers = n_workers,
            )

        Parallel(n_jobs=n_workers, verbose = 0)\
            (delayed(tune_func)(worker_number = i) for i in range(n_workers))

        self.print()
        time.sleep(0.05)

        return self.study


    @adi.wraps_modelfunc(tmi.fetch_split_train_test, 
        fill_kwargs = ['all_data', 'train_data', 'test_data'])
    def _tune(self, worker_number = 0, n_workers = 1, parallel = False,*, 
        all_data, train_data, test_data):

        if isinstance(self.cv, int):
            self.cv = KFold(self.cv, random_state = self.seed, shuffle= True)
        else:
            assert isinstance(self.cv, BaseCrossValidator)

        if parallel and self.model.dataset_loader_workers > 0:
            logger.warn('Worker {}: Parrallelized tuning on one node with muliprocessed data loading is not permitted. Setting "dataset_loader_workers" to zero.'.format(str(worker_number)))
            self.model.dataset_loader_workers = 0
        
        self.study = optuna.create_study(
            direction = 'minimize',
            pruner = self.get_pruner(n_workers = n_workers, num_splits = self.cv.get_n_splits()),
            sampler = self.get_tuner(worker_number = worker_number, parallel=parallel),
            study_name = self.study_name,
            storage = self.storage,
            load_if_exists= True,
        )
        
        trial_func = partial(
            self._trial,
            base_seed = self.seed,
            model = self.model, data = train_data,
            cv = self.cv, batch_sizes = self.batch_sizes,
            min_dropout = self.min_dropout, max_dropout = self.max_dropout,
            min_epochs = self.min_epochs, max_epochs = self.max_epochs,
            min_topics = self.min_topics, max_topics = self.max_topics,
            tune_kl_strategy = self.tune_kl_strategy, 
            tune_layers = self.tune_layers,
            study_name = self.study_name,
            tensorboard_logdir = self.tensorboard_logdir,
            parallel = parallel
        )

        if not torch.cuda.is_available():
            logger.warn('Worker {}: GPU is not available, will not speed up training.'.format(str(worker_number)))

        remaining_trials = self.iters - len(self.study.trials)
        if remaining_trials > 0:

            with DisableLogger(baselogger), DisableLogger(interfacelogger), DisableLogger(corelogger):

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                
                    np.random.seed(self.seed + worker_number)

                    try:
                        self.study.optimize(
                            trial_func, n_trials = remaining_trials//n_workers, 
                            callbacks = [_print_study] if not parallel else [partial(_log_progress, worker_number = worker_number, num_trials = self.iters)],
                            catch = (ValueError),
                        )
                    except KeyboardInterrupt:
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

        return [trial.params for trial in self.get_best_trials(top_n_trials)]

    def _get_best_params(self, top_n_trials = 5):

        return {trial.number : trial.params for trial in self.get_best_trials(top_n_trials)}


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

        domains = {
            'kl_strategy' : ['monotonic','cyclic'],
            'batch_size' : self.batch_sizes,
            'num_layers' : [2,3],
        }

        scores = []
        best_params = self._get_best_params(top_n_trials)

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
                        writer.add_hparams(params, {'test_score' : test_score}, domains)

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

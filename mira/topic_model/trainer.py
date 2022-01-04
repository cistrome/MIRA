import numpy as np
from sklearn.model_selection import KFold
from functools import partial
import optuna
import logging
import mira.adata_interface.core as adi
import mira.adata_interface.topic_model as tmi
from optuna.trial import TrialState as ts
import joblib
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)  # Setup the root logger.
optuna.logging.set_verbosity(optuna.logging.WARN)
from mira.topic_model.base import logger as baselogger
from mira.adata_interface.topic_model import logger as interfacelogger


class DisableLogger:
    def __init__(self, logger):
        self.logger = logger
        self.level = logger.level

    def __enter__(self):
        self.logger.setLevel(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        self.logger.setLevel(self.level)


def _print_study(study, trial):

    if study is None:
        raise ValueError('Cannot print study before running any trials.')

    def get_trial_desc(trial):

        if trial.state == ts.COMPLETE:
            return 'Trial #{:<3} | completed, score: {:.4e} | params: {}'.format(str(trial.number), trial.values[-1], str(trial.user_attrs['trial_params']))
        elif trial.state == ts.PRUNED:
            return 'Trial #{:<3} | pruned at step: {:<12} | params: {}'.format(str(trial.number), str(trial.user_attrs['batches_trained']), str(trial.user_attrs['trial_params']))
        elif trial.state == ts.FAIL:
            return 'Trial #{:<3} | ERROR                        | params: {}'\
                .format(str(trial.number), str(trial.user_attrs['trial_params']))

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

    print('Modules | Trials (number is #folds tested)', end = '')

    study_results = sorted([
        (trial_.user_attrs['trial_params']['num_topics'], trial_.user_attrs['batches_trained'], trial_.number)
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
        print(get_trial_desc(trial))

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
    '''
    Tune hyperparameters of the MIRA topic model using iterative Bayesian optimization.
    First, the optimization engine suggests a hyperparameter combination. The model
    is then trained with those parameters for 5 folds of cross validation (default option)
    to compute the performance of that model. If the parameter combination does not 
    meet the performance of previously-trained combinations, the trial is terminated early. 

    Depending on the size of your dataset, you may change the pruning and cross validation
    schemes to reduce training time. 

    The tuner returns an ``study`` object from the package [Optuna](https://optuna.org/).
    The study may be reloaded to resume optimization later, or printed to review results.

    After tuning, the best models compete to minimize loss on a held-out set of cells.
    The winning model is returned as the final model of the dataset.

    Examples
    --------
    >>> tuner = mira.topics.TopicModelTuner(
                topic_model,
                save_name = 'study.pkl',
            )
    >>> tuner.train_test_split(data)
    >>> tuner.tune(data)
    >>> tuner.select_best_model(data)

    '''

    @classmethod
    def load_study(cls, filename):
        return joblib.load(filename)

    def __init__(self,
        topic_model,
        save_name = None,
        test_column = None,
        min_topics = 5, max_topics = 55,
        min_epochs = 20, max_epochs = 40,
        min_dropout = 0.01, max_dropout = 0.15,
        batch_sizes = [32,64,128],
        cv = 5, iters = 64,
        study = None,
        seed = 2556,
        pruner = 'halving',
    ):
        '''
        Initialize a new tuner.

        Parameters
        ----------
        topic_model : mira.topics.ExpressionTopicModel or mira.topics.AccessibilityTopicModel
            Topic model to tune. The provided model should have columns specified
            to retrieve endogenous and exogenous features, and should have the learning
            rate configued by ``get_learning_rate_bounds``.
        save_name : str, default = None
            Filename under which to save tuning results. After each iteration, the ``study``
            object will be saved here.
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
        study : None or `optuna.Study`
            If None, begin a new hyperparameter optimization routine. If given a study, 
            resume that study. If study is provided, `save_name` need not be set.
        seed : int > 0, default = 2556
            Random seed for K-fold cross validator and optuna optimizer.
        pruner : "halving" or "median"
            If "halving", use SuccessiveHalving Bandit pruner. Works best with default
            five folds of cross validation. If "median", use median pruner.

        Returns
        -------
        study : optuna.study

        Raises
        ------
        ValueError : If study not provided and `save_name` not set.

        Examples
        --------
        Using default parameters:

        >>> tuner = mira.topics.TopicModelTuner(
                topic_model,
                save_name = 'study.pkl',
            )

        For large datasets, it may be useful to skip cross validation since the
        variance of the estimate of model performance should be lower. It may also
        be appropriate to limit the model to larger batch sizes.

        >>> tuner = mira.topics.TopicModelTuner(
                topic_model,
                save_name = 'study.pkl',
                cv = sklearn.model_selection.ShuffleSplit(n_splits = 1, train_size = 0.8),
                batch_sizes = [64,128],
            )

        '''
        self.model = topic_model
        self.test_column = test_column or 'test_set'
        self.min_topics, self.max_topics = min_topics, max_topics
        self.min_epochs, self.max_epochs = min_epochs, max_epochs
        self.min_dropout, self.max_dropout = min_dropout, max_dropout
        self.batch_sizes = batch_sizes
        self.cv = cv
        self.iters = iters
        self.study = study
        self.seed = seed
        self.save_name = save_name
        self.pruner = pruner

        if not study is None:
            assert(not study.study_name is None), 'Provided studies must have names.'
        elif study is None and save_name is None:
            raise ValueError('Must provide a "save_name" to start a new study.')
        
        self.study = study


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
    def trial(
            trial,
            prune_penalty = 0.01,*,
            model, data, cv, batch_sizes,
            min_topics, max_topics,
            min_dropout, max_dropout,
            min_epochs, max_epochs,
        ):
        params = dict(
            num_topics = trial.suggest_int('num_topics', min_topics, max_topics, log=True),
            batch_size = trial.suggest_categorical('batch_size', batch_sizes),
            encoder_dropout = trial.suggest_float('encoder_dropout', min_dropout, max_dropout),
            num_epochs = trial.suggest_int('num_epochs', min_epochs, max_epochs, log = True),
            beta = trial.suggest_float('beta', 0.90, 0.99, log = True),
            seed = np.random.randint(0, 2**32 - 1),
        )

        model.set_params(**params)

        trial.set_user_attr('trial_params', params)
        trial.set_user_attr('completed', False)
        trial.set_user_attr('batches_trained', 0)
        cv_scores = []

        num_splits = cv.get_n_splits(data)
        cv_scores = []

        num_updates = num_splits * params['num_epochs']
        print('Evaluating: ' + str(params))
        for step, (train_idx, test_idx) in enumerate(cv.split(data)):

            train_counts, test_counts = data[train_idx], data[test_idx]
            
            for epoch, loss in model._internal_fit(train_counts):
                num_hashtags = int(10 * epoch/params['num_epochs'])
                print('\rProgress: ' + '|##########'*step + '|' + '#'*num_hashtags + ' '*(10-num_hashtags) + '|' + '          |'*(num_splits-step-1),
                    end = '')

            cv_scores.append(
                model.score(test_counts)
            )

            if step == 0:
                trial.report(1.0, 0)

            trial.report(np.mean(cv_scores) + (prune_penalty * 0.5**step), step+1)
                
            if trial.should_prune() and step + 1 < num_splits:
                trial.set_user_attr('batches_trained', step+1)
                raise optuna.TrialPruned()

        trial.set_user_attr('batches_trained', step+1)
        trial.set_user_attr('completed', True)
        trial_score = np.mean(cv_scores)

        return trial_score

    @staticmethod
    def _save_study(study, trial):
        joblib.dump(study, study.study_name)

    def save(self):
        '''
        Save study to `study_name`.

        Parameters
        ----------
        None
        '''
        self._save_study(self.study, None)

    def print(self):
        _print_study(self.study, None)

    def get_pruner(self):
        if self.pruner == 'halving':
            return optuna.pruners.SuccessiveHalvingPruner(
                        min_resource=1.0, 
                        bootstrap_count=0, 
                        reduction_factor=3)
        elif self.pruner == 'median':
            return optuna.pruners.MedianPruner(
                n_startup_trials=3,
                n_warmup_steps=0,
            )
        else:
            raise ValueError('Pruner {} is not an option'.format(str(self.pruner)))


    @adi.wraps_modelfunc(tmi.fetch_split_train_test, 
        fill_kwargs = ['all_data', 'train_data', 'test_data'])
    def tune(self,*, all_data, train_data, test_data):
        '''
        Run Bayesian optimization scheme for topic model hyperparameters. 

        Parameters
        ----------
        adata : anndata.AnnData
            Anndata of expression or accessibility data.
            Cells must be labeled with test or train set membership using
            `tuner.train_test_split`.
        
        Returns
        -------
        study : optuna.Study
            Study object summarizing results of tuning iterations.
        '''
        
        '''error_file = logging.FileHandler(self.logfile, mode="a")
        logger.addHandler(error_file)
        optuna.logging.enable_propagation()  # Propagate logs to the root logger.
        optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.'''

        if self.study is None:
            self.study = optuna.create_study(
                direction = 'minimize',
                pruner = self.get_pruner(),
                study_name = self.save_name,
            )

        if isinstance(self.cv, int):
            self.cv = KFold(self.cv, random_state = self.seed, shuffle= True)
        
        trial_func = partial(
            self.trial, 
            model = self.model, data = train_data,
            cv = self.cv, batch_sizes = self.batch_sizes,
            min_dropout = self.min_dropout, max_dropout = self.max_dropout,
            min_epochs = self.min_epochs, max_epochs = self.max_epochs,
            min_topics = self.min_topics, max_topics = self.max_topics,
        )

        with DisableLogger(baselogger), DisableLogger(interfacelogger):

            try:
                self.study.optimize(trial_func, n_trials = self.iters, callbacks = [_print_study, self._save_study],
                catch = (RuntimeError,ValueError),)
            except KeyboardInterrupt:
                pass

        self.print()
        self._save_study(self.study, None)
        #finally:
        #    error_file.close()
        
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
        
        def score_trial(trial):
            if trial.user_attrs['completed']:
                try:
                    return trial.values[-1]
                except (TypeError, AttributeError):
                    pass

            return np.inf

        sorted_trials = sorted(self.study.trials, key = score_trial)

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

        return [trial.user_attrs['trial_params'] for trial in self.get_best_trials(top_n_trials)]


    @adi.wraps_modelfunc(tmi.fetch_split_train_test, adi.return_output, ['all_data', 'train_data', 'test_data'])
    def select_best_model(self, top_n_trials = 5, *,all_data, train_data, test_data):
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

        Returns
        -------
        best_model : sublcass of mira.topic_model.BaseModel
            Best-performing model chosen using validation set.
        '''

        scores = []
        best_params = self.get_best_params(top_n_trials)
        for params in best_params:
            logging.info('Training model with parameters: ' + str(params))
            try:
                scores.append(self.model.set_params(**params).fit(train_data).score(test_data))
                logging.info('Score: {:.5e}'.format(scores[-1]))
            except (RuntimeError, ValueError) as err:
                logging.error('Error occured while training, skipping model.')
                scores.append(np.inf)

        final_choice = best_params[np.argmin(scores)]
        logging.info('Set parameters to best combination: ' + str(final_choice))
        self.model.set_params(**final_choice)
        
        logging.info('Training model with all data.')
        self.model.fit(all_data)

        return self.model
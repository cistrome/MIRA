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

        assert(isinstance(train_size, float) and train_size > 0 and train_size < 1)
        num_samples = shape[0]
        assert(num_samples > 0), 'Adata must have length > 0.'
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

        try:
            self.study
        except AttributeError:
            raise Exception('User must run "tune_hyperparameters" before running this function')

        return self.study.trials


    def get_best_trials(self, top_n_models = 5):
        
        assert(isinstance(top_n_models, int) and top_n_models > 0)
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

        return sorted_trials[:top_n_models]

    def get_best_params(self, top_n_models = 5):

        return [trial.user_attrs['trial_params'] for trial in self.get_best_trials(top_n_models)]


    @adi.wraps_modelfunc(tmi.fetch_split_train_test, adi.return_output, ['all_data', 'train_data', 'test_data'])
    def select_best_model(self,top_n_models = 5, *,all_data, train_data, test_data):

        scores = []
        best_params = self.get_best_params(top_n_models)
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
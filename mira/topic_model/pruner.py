from optuna.pruners import PercentilePruner
from optuna.trial._state import TrialState
from optuna.pruners._percentile import _is_first_in_interval_step, \
    _get_percentile_intermediate_result_over_trials
import math


class MemoryPercentileStepPruner(PercentilePruner):
    
    def __init__(self, *args, memory_length = 10, **kwargs):
            super().__init__(*args, **kwargs)
            self.memory_length = memory_length

    def prune(self, study, trial):
        
        all_trials = study.get_trials(deepcopy=False)
        n_trials = len([t for t in all_trials if t.state == TrialState.COMPLETE])

        if n_trials == 0:
            return False

        if n_trials < self._n_startup_trials:
            return False

        step = trial.last_step
        if step is None:
            return False

        n_warmup_steps = self._n_warmup_steps
        if step < n_warmup_steps:
            return False

        if not _is_first_in_interval_step(
            step, trial.intermediate_values.keys(), n_warmup_steps, self._interval_steps
        ):
            return False

        direction = study.direction
        score = trial.intermediate_values[step]
        if math.isnan(score):
            return True
        
        complete_trials = [t for t in all_trials if t.state == TrialState.COMPLETE]

        historical_percentile = _get_percentile_intermediate_result_over_trials(
            complete_trials, direction, step, self._percentile, self._n_min_trials
        )
        
        recent_memory_percentile = _get_percentile_intermediate_result_over_trials(
            complete_trials[-self.memory_length : ], direction, step, self._percentile, self._n_min_trials
        )
        
        score_threshold = min(recent_memory_percentile, historical_percentile)
        
        if math.isnan(score_threshold):
            return False

        to_prune = score > score_threshold
        
        study._storage.set_trial_system_attr(trial._trial_id, step, 
                                                {'recent_memory_threshold' : recent_memory_percentile,
                                                'historical_percentile' : historical_percentile,
                                                'score' : score, 'score_threshold' : score_threshold,
                                                'to_prune' : to_prune})
        
        return to_prune
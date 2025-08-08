import os
import json
import operator
import warnings
from functools import singledispatch
from typing import Optional, Union
from collections.abc import Callable

import pandas as pd
import numpy as np

import optuna
from optuna.study import StudyDirection
from optuna.trial import TrialState
from optuna.pruners import BasePruner, NopPruner
from optuna.exceptions import ExperimentalWarning
from optuna.logging import get_logger

from . import defaults
from .utils import dump_intermediate_results, import_source_as_module


if defaults.disable_experimental_warnings:
    warnings.filterwarnings("ignore", category=ExperimentalWarning)


logger = get_logger('optuna')


class TrialStopException(Exception):
    """Generic exception raised for errors in trials."""
    def __init__(self):
        self.message = f'Stopping the trial. Reason: {self.__class__.__name__}'
        super().__init__(self.message)

class InvalidExperimentConfig(TrialStopException): pass
class EarlyStopping(TrialStopException): pass
class EarlyPruning(TrialStopException): pass
class DuplicateConfigExists(TrialStopException): pass


class ImpatientPruner(BasePruner):
    '''
    Pruner that helps to either prun or safely stop iterations within a trial.
    =========================================================================

    Behavior is somewhat opposit to Optuna's `PatientPruner` that forbids wrapped
    pruner to interfere within a patience interval. In contrast, `ImpatientPruner`
    attempts to stop trials as soon as the intermediate scores start stagnating.
    
    Moreover, this pruner allows to not only prune trials but also to normally
    complete them by skipping the remaning iterations. For that purpose, it provides 
    the `is_impatient` attribute which can be used to make the prune-or-complete
    decision. If `is_impatient` is `True` and if pruning decision is positive, one
    can simply collect results and stop iterations without raising `optuna.TrialPruned()`.
    
    Example:
    >>> ... # somewhere in a loop
    >>>     if trial.should_prune():
    >>>         if getattr(trial.study.pruner, 'is_impatient', False): # handle other pruners
    >>>             break # complete trial normally but do not iterate anymore    
    >>>         raise optuna.TrialPruned() # abrupt trial
    '''
    def __init__(self, wrapped_pruner: BasePruner, patience: int, min_delta: Optional[float] = 0.0):
        self._main_pruner = wrapped_pruner or NopPruner()
        self._patience = patience
        self._min_delta = min_delta
        self.is_impatient = None

    def prune(self, study: optuna.Study, trial: optuna.Trial):
        if (trial.last_step is None) or (self._patience < 0):
            return False
                
        self.is_impatient = False
        if self._patience == 0:
            self.is_impatient = True
            return True
        
        
        intermediate_values = trial.intermediate_values
        steps = np.fromiter(intermediate_values.keys(), dtype=int)

        # Do not prune if the number of steps to decide is insufficient.
        if steps.size <= self._patience:
            return False

        steps.sort()
        # Scores within patience interval
        patience_steps = steps[-self._patience-1:]
        patience_scores = np.fromiter(
            (intermediate_values[step] for step in patience_steps), dtype=float
        )
        if study.direction == StudyDirection.MAXIMIZE:
            do_prune = np.nanmax(patience_scores) <= (patience_scores[0] + self._min_delta)
            if do_prune:
                self.is_impatient = True
                return True
        else:
            raise NotImplementedError
        return self._main_pruner.prune(study, trial)


def check_duplicates(trial: optuna.Trial):
    trials = trial.study.get_trials(
        deepcopy = False,
        states = (TrialState.COMPLETE, TrialState.PRUNED)
    )
    for t in trials:
        if t.params == trial.params:
            raise DuplicateConfigExists


def prepare_pruner(args, main_pruner: Optional[BasePruner] = None):
    pruner = main_pruner or NopPruner()
    # check early stopping settings:
    patience = getattr(args, 'es_max_steps', -1)
    min_delta = getattr(args, 'es_tol', 0.0)
    if patience < 0:
        return pruner
    return ImpatientPruner(pruner, patience=patience, min_delta=min_delta)    


def validation_callback(
        trial: Union[optuna.Trial, optuna.trial.FrozenTrial, optuna.trial.FixedTrial],
        target_metric: str
    ) -> Callable[[dict, int, float], None]:
    '''
    Evaluate model, report progress on metrics, and also maybe prune.
    ================================================================
    
    **Notes**:
    1) Good reading on pruning essentials:
    https://tech.preferred.jp/en/blog/how-we-implement-hyperband-in-optuna/
    2) Some issues on appropriate reporting and a workaround (implemented here as well):
    https://github.com/optuna/optuna/issues/2575#issuecomment-1203699079
    '''    
    def callback(results: dict, step: int, objective: float):
        step_results = results[step]
        step_results.loc['objective', 'score'] = objective
        report_progress(results, step, target_metric, trial)
        if trial.should_prune(): # prune unpromising trial w.r.t other trials
            pruner = trial.study.pruner
            if getattr(pruner, 'is_impatient', False):
                logger.info(
                    'Trial stopped due to impatient pruning within the last '
                    f'{pruner._patience+1} steps with {pruner._min_delta} tolerance.'
                )
                raise EarlyStopping  # allow completing trial normally, discard remaining steps
            raise EarlyPruning # prune trial
    return callback


def report_progress(
    results: dict, step: int, target_metric: str, trial: optuna.Trial
) -> None:
    '''
    Implements a running best score reporting.
    =========================================

    The reporting is managed in such a way so that optuna's pruners do not 
    prune models when they already started overfitting. Only best scores 
    and the corresponding step the're achived at are reported.
    More on this workaround:
    https://github.com/optuna/optuna/issues/2575#issuecomment-1203699079
    
    **Note**:
    It also means that the intermediate values history of a trial is not
    representative, as it doesn't show real values, only running best.
    Special measures must be taken to save the actual values at each step.
    For example, save all results with trials' `set_user_attributes`. See
    `log_attributes` function as an example.
    '''
    scores = results[step]['score'].to_dict()
    n_steps = step + 1 # indexing starts from 0
    scores_str = " | ".join(f'{name}: {val:.6f}' for name, val in scores.items())
    logger.info(f'steps: {n_steps} | {scores_str}')
    target_score = scores[target_metric]
    is_maximize = defaults.study_direction == 'maximize'
    try:
        best_results = results['best']
    except KeyError:
        default_score = float('-inf') if is_maximize else float('inf')
        best_results = {target_metric: default_score, 'step': None}
        results['best'] = best_results
    current_best_score = best_results[target_metric]
    op = operator.gt if is_maximize else operator.lt
    if op(target_score, current_best_score):
        current_best_score = target_score
        best_results.update({target_metric: current_best_score, 'step': step})
    trial.report(current_best_score, step=step)


@singledispatch
def log_attributes(results, trial):
    pass # do not log on unrecognized types

@log_attributes.register
def log_attributes_df(results: pd.DataFrame, trial: optuna.Trial) -> None:
    '''Generic logging interface for non-iterative learning algorithms.'''
    trial.set_user_attr('metrics', results['score'].to_dict())

@log_attributes.register(dict)
def log_attributes_dict(results: dict[int, pd.DataFrame], trial: optuna.Trial) -> None:
    '''Advanced logging interface for iterative learning algorithms.'''
    best_results = results.pop('best')
    try:
        trial.set_user_attr('best', best_results)
        all_results = pd.concat(results, axis=0, names=['steps', 'metrics'])
        trial.set_user_attr('history', all_results['score'].unstack('metrics').to_dict())
    finally:
        results.update({'best': best_results})


def create_test_trial(study: optuna.Study):
    best_trial = study.best_trial
    test_trial = optuna.create_trial(
        params = best_trial.params,
        distributions = best_trial.distributions,
        user_attrs = {
            'fixed_params': best_trial.user_attrs.get('fixed_params'),
            'best': best_trial.user_attrs['best'],
        },
        value = 0
    )
    study.add_trial(test_trial)
    return test_trial


def dump_trial_results(trial: optuna.Trial, study_name: Optional[str] = None):
    trial_id = trial.number
    study_name = study_name or trial.study.study_name
    filename = os.path.join(defaults.data_dir, f'results/{study_name}_results.txt')
    try:
        results = trial.user_attrs['history']
    except KeyError: # handle non-iterative learning
        results = trial.user_attrs['metrics']
        results_dict = {'results': results}
    else:
        results_dict = {'results': results, 'best': trial.user_attrs['best']}
    results_dict['config'] = get_trial_config(trial)
    results_dict['trial_id'] = trial_id
    dump_intermediate_results(results_dict, filename)


def read_trial_results(filename):
    with open(filename, 'r') as f:
        results_list = json.load(f)
    
    metrics_data = {}
    config_dict = {}

    for i, result in enumerate(results_list):
        trial_id = f'{i}{result["trial_id"]}' # workaround for duplicate trial ids
        # Extract metrics
        metrics = result['results']
        metrics_data[trial_id] = pd.DataFrame.from_dict(metrics)
        # Store configuration
        config = result['config']
        config_dict[trial_id] = config

    metrics_df = pd.concat(metrics_data, names=['trial_id', 'step'])
    return metrics_df, config_dict


def get_trial_config(trial: optuna.Trial):
    fixed_params = trial.user_attrs.get('fixed_params', {})
    return {**trial.params, **fixed_params}


def config_from_file(trial: Optional[optuna.Trial], path: str) -> dict:
    '''
    Suggest a new config for training based on a specification
    provided in a .py file available by path.

    If `trial` is None, the correposing config must not use `suggest` methods,
    otherwise an error will be thrown.
    '''
    grid = import_source_as_module(path)
    return grid.generate_config(trial)


def save_config(trial: optuna.Trial, config: dict) -> None:
    """
    Saves a configuration and its associated trial ID to a file.

    Parameters
    ----------
    config : dict
        The configuration to save.
    trial : optuna.Trial
        The trial associated with the configuration.

    Returns
    -------
    None

    """
    trial_id = trial.number
    study_name = trial.study.study_name
    filename = os.path.join(defaults.data_dir, f'results/{study_name}_config.txt')
    dump_intermediate_results({'trial_id': trial_id, 'config': config}, filename)


def load_config(study_name: str) -> list:
    """
    Loads a list of configurations and their associated trial IDs from a file.

    Parameters
    ----------
    study_name : str
        The name of the study a configuration belongs to.

    Returns
    -------
    list
        A list of dictionaries, each containing a 'trial_id' key and a 'config' key.
    """    
    filename = os.path.join(defaults.data_dir, f'results/{study_name}.txt')
    with open(filename, 'r') as f:
        return json.load(f)


def get_storage_name(storage_db: str, study_name: str) -> str:
    if storage_db == 'sqlite':
        storage_name = os.path.join(defaults.data_dir, f'results/{study_name}.db')
        return f'{storage_db}:///{storage_name}'
    if storage_db == 'redis': # "redis://127.0.0.1:6379/db"
        return f'{storage_db}://localhost:6379/{study_name}'


def check_step_replacement(trial: Union[optuna.trial.FrozenTrial, optuna.trial.FixedTrial]) -> None:
    fixed_config = trial.user_attrs.get('fixed_params', {})
    try:
        step_key = fixed_config['step_replacement']
    except KeyError:
        logger.warning(f'Unable to identify the optimal number of steps. Will run exhaustively.')
        return
    
    best_results = trial.user_attrs['best']
    num_steps = best_results['step'] + 1
    max_steps = fixed_config[step_key]
    if num_steps != max_steps:
        fixed_config[step_key] = num_steps
        logger.info(f'Max number of steps changed from {max_steps} to {num_steps}.')


def set_max_trials(n_steps):
    if n_steps is None:
        return defaults.grid_steps_limit
    if n_steps == 0:
        return np.iinfo('i8').max
    return n_steps


class GridExtractor:
    def __init__(self):
        self.extracted = []
    
    def suggest_categorical(self, name: str, values: list):
        self.extracted.append(name)
        return values
    
def full_grid_sampler(config_path):
    grid_extractor = GridExtractor()
    grid = config_from_file(grid_extractor, config_path)
    search_space = {key: grid[key] for key in grid_extractor.extracted}
    return optuna.samplers.GridSampler(search_space)
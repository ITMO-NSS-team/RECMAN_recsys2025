import argparse
from datetime import datetime
from functools import singledispatchmethod
from typing import Union

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from lib import defaults, logger
from lib.data.processor import DataSet, prepare_data, get_sequence_length
from lib.evaluation import Evaluator
from lib.models.learning import import_model_routine, train_data_format, test_data_format
from lib.optunatools import config_from_file, create_test_trial, dump_trial_results, full_grid_sampler, get_storage_name, get_trial_config, log_attributes, prepare_pruner, save_config, set_max_trials, validation_callback
from lib.optunatools import check_duplicates, check_step_replacement, EarlyPruning, InvalidExperimentConfig, DuplicateConfigExists
from lib.argparser import parse_args
from lib.utils import show_average_time


class Objective:
    '''Optuna-compatible objective constructor for hyperparameters tuning.'''
    def __init__(self, datapack: Union[list, tuple], run_args: argparse.Namespace) -> None:
        self.run_args = run_args
        # make a statefull data instance to avoid redundant processing of the dataset in each call
        self.dataset = DataSet(
            datapack,
            name = run_args.dataset,
            # declare default dataset formats (different from original pandas dataframes format)
            train_format = train_data_format(run_args.model), # needed for model training
            test_format = test_data_format(run_args), # needed for evaluation
            # ensure all formats are stored, avoid recomputing them:
            is_persistent = True,
        )
        # declare additional dataset formats
        self.dataset.initialize_formats({'train': 'sequential'}) # user-seq series, for evaluation
        self.dataset.info()
        # make model-specific preparations
        self.train_validate = import_model_routine(run_args.model, 'train_validate')
        # set evaluation-specific attributes
        self.topn = run_args.topn
        self.target_metric = f'{run_args.target_metric}@{self.topn}'.upper()
        self.results = None

    @singledispatchmethod
    def __call__(self, trial: optuna.Trial) -> float:
        '''Standard call to objective within optuna's `study.optimize` method.'''
        optuna_callback = validation_callback(trial, self.target_metric)
        self.evaluator = Evaluator(self.dataset, self.topn, evaluation_callback=optuna_callback)
        try: # main training program
            trial_config = config_from_file(trial, self.run_args.config_path)
            if trial_config.get('maxlen') is None: # TODO: implement as postprocessing callback
                trial_config['maxlen'] = get_sequence_length(self.dataset.name)
            if defaults.drop_duplicated_trials:
                check_duplicates(trial)
            fixed_params = {k:v for k,v in trial_config.items() if k not in trial.params}
            if fixed_params:
                trial.set_user_attr('fixed_params', fixed_params)
            if self.run_args.save_config:
                save_config(trial, trial_config)
            logger.info(f'Starting a new trial with configuration: {trial_config}')
            logger.info(f'Target metric: {self.target_metric}')
            self.train_validate(trial_config, self.evaluator)
            logger.info(f'Average evaluation time: {show_average_time(self.evaluator.evaluation_time)}')
        except (EarlyPruning, InvalidExperimentConfig, DuplicateConfigExists) as e:
            logger.info(e)
            raise optuna.TrialPruned()
        except KeyboardInterrupt: # stop gracefully, allow running final test
            logger.warning('Interrupted by user.')
            trial.study.stop()
            return
        finally:
            log_attributes(self.evaluator.results, trial)  # make sure to log current results
            if self.run_args.dump_results:
                dump_trial_results(trial)
        # extract target score - depends on whether there's a history of results
        # or a single dataframe (accessed via 'most_recent_results' attribute):
        try:
            best_results = self.evaluator.results['best']
        except KeyError:
            self.results = self.evaluator.most_recent_results
        else:
            self.results = self.evaluator.results[best_results['step']]
        score = self.results.loc[self.target_metric, 'score']
        return score

    @__call__.register(optuna.trial.FrozenTrial)
    @__call__.register(optuna.trial.FixedTrial)
    def _(self, trial: Union[optuna.trial.FrozenTrial, optuna.trial.FixedTrial]) -> float:
        '''
        Objective overload to perform evaluation on fixed parameters.
        ============================================================

        It relies on the type of submitted trial, which must be a `FrozenTrial` ro `FixedTrial`.
        Tyically, it would be just the best trial of a study.
        From the Optuna docs:
        `best_trial` is a FrozenTrial. The FrozenTrial is different from an active trial
        and behaves differently - pruning does not work, `should_prune` always returns False,
        changes in user_attrs won't affect source storage.
        Source:
          - https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/010_reuse_best_trial.html
          - https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.FrozenTrial.html
          - https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.FixedTrial.html

        This property affects `validation_callback` behavior.
        It is convenient for testing purposes as the number of epochs will remain fixed.
        '''
        optuna_callback = validation_callback(trial, self.target_metric)
        self.evaluator = Evaluator(self.dataset, self.topn, evaluation_callback=optuna_callback)
        check_step_replacement(trial)
        config = get_trial_config(trial)
        logger.info(f'Running test trial with configuration: {config}')
        self.train_validate(config, self.evaluator)
        logger.info(f'Average evaluation time: {show_average_time(self.evaluator.evaluation_time)}')
        self.results = self.evaluator.most_recent_results
        score = self.results.loc[self.target_metric, 'score']
        return score


if __name__ == "__main__":
    run_args = parse_args()
    ts = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    study_name = run_args.study_name or f'{run_args.model}_{run_args.dataset}_{run_args.target_metric}_{ts}'
    storage_name = get_storage_name(run_args.storage, study_name)

    if run_args.exhaustive:
        sampler = full_grid_sampler(run_args.config_path)
        run_args.grid_steps = 0
    else:
        sampler = optuna.samplers.RandomSampler()

    study = optuna.create_study(
        load_if_exists = True,
        study_name = study_name,
        direction = defaults.study_direction,
        sampler = sampler,
        pruner = prepare_pruner(run_args),
        storage = storage_name,
    )

    tune_datapack, test_datapack = prepare_data(run_args.dataset, time_offset_q=[run_args.time_offset]*2)
    objective = Objective(tune_datapack, run_args)
    study.optimize(
        objective,
        callbacks = [
            MaxTrialsCallback(set_max_trials(run_args.grid_steps), states=(TrialState.COMPLETE,)),
        ],
        n_trials = run_args.grid_steps * defaults.max_attempts_multiplier if run_args.grid_steps else None
    )

    if run_args.check_best: # see test scores for current best configuration
        test_trial = create_test_trial(study)
        test_objective = Objective(test_datapack, run_args)
        test_objective(test_trial) # objective overload on FrozenTrial
        logger.info(f'Test results for the provided parameters:\n{test_objective.results}')
        log_attributes(test_objective.evaluator.results, test_trial)  # make sure to log current results
        if run_args.dump_results:
            dump_trial_results(test_trial, f'{study_name}_TEST')

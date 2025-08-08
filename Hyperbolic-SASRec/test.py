import optuna

from lib import defaults, logger
from lib.data.processor import prepare_data
from lib.optunatools import config_from_file, get_storage_name
from lib.argparser import parse_args

from tune import Objective


if __name__ == "__main__":
    run_args = parse_args(is_test=True)
    test_datapack, = prepare_data(run_args.dataset, time_offset_q=run_args.time_offset)
    test_objective = Objective(test_datapack, run_args)
    storage_name = get_storage_name(run_args.storage, run_args.study_name)
    
    if run_args.study_name:
        study = optuna.create_study(
            load_if_exists = True,
            study_name = run_args.study_name,
            direction = defaults.study_direction,
            storage = storage_name,
        )
        test_trial = study.best_trial
        config = {**test_trial.params, **test_trial.user_attrs.get('fixed_params', {})}
    else:
        # the config file must not contain any `suggest` methods, only pure key: value dict is allowed
        config = config_from_file(None, run_args.config_path)
        test_trial = optuna.trial.FixedTrial({}) # distributions for fixed params are not available
        test_trial.set_user_attr('fixed_params', config)
    
    test_objective(test_trial) # objective overload on FrozenTrial
    logger.info(f'Test results for the provided parameters {config}:\n{test_objective.results}')

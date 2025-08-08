import optuna
from grids.core import hypsasrec, generate_base_config


def generate_config(trial: optuna.Trial) -> dict:
    trial_params = hypsasrec.trial_params
    fixed_params = {
        **hypsasrec.fixed_params,
        'c': 1, 
    }
    suggest = trial.suggest_categorical
    config = generate_base_config(trial_params, fixed_params, suggest)
    return config
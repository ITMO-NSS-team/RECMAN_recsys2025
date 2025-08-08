import optuna
from grids.core import sasrec, generate_base_config


def generate_config(trial: optuna.Trial) -> dict:
    trial_params = sasrec.trial_params
    fixed_params = sasrec.fixed_params
    suggest = trial.suggest_categorical
    config = generate_base_config(trial_params, fixed_params, suggest)
    return config

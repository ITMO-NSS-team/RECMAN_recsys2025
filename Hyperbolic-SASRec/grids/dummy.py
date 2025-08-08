import optuna
from grids.core import dummy, generate_base_config

def generate_config(trial: optuna.Trial) -> dict:
    trial_params = dummy.trial_params
    fixed_params = dummy.fixed_params
    suggest = trial.suggest_int
    config = generate_base_config(trial_params, fixed_params, suggest)
    return config
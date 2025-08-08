import optuna
from grids.core import sasrec_manifold, generate_base_config


def generate_config(trial: optuna.Trial) -> dict:
    trial_params = sasrec_manifold.trial_params
    fixed_params = sasrec_manifold.fixed_params
    suggest = trial.suggest_categorical
    config = generate_base_config(trial_params, fixed_params, suggest)
    return config

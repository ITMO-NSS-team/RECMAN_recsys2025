import optuna
from grids.core import svd, generate_base_config


def generate_config(trial: optuna.Trial) -> dict:
    trial_params = svd.trial_params
    fixed_params = svd.fixed_params
    suggest = trial.suggest_categorical
    config = generate_base_config(trial_params, fixed_params, suggest)
    if config['randomized']:
        config['rnd_svd_seed'] = 0
    return config
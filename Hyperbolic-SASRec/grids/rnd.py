import optuna


def generate_config(trial: optuna.Trial) -> dict:
    config = dict(
        seed = 135,
    )
    return config
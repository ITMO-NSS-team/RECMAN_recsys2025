def generate_base_config(trial_params, fixed_params, suggest_method):
    trial_config = {}
    for name, values in trial_params.items():
        if isinstance(values, tuple):
            trial_config[name] = suggest_method(
                name, *values
            )  # e.g., suggest_int('param', 0, 10)
        else:
            trial_config[name] = suggest_method(
                name, values
            )  # e.g., suggest_categorical('param', [0, 10])

    fixed_config = {name: value for name, value in fixed_params.items()}
    config = {**trial_config, **fixed_config}
    return config


def generate_best_config(fixed_params):
    fixed_config = {name: value for name, value in fixed_params.items()}
    config = {**fixed_config}
    return config

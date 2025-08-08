trial_params = dict(
    seed = (0, 2**31-1),
)

fixed_params = dict(
    max_epochs = 240,
    step_replacement = None # during tests, max value will be replaced with an optimal one
)
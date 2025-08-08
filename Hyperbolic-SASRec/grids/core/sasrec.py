trial_params = dict(
    batch_size = [128, 256, 512],
    learning_rate = [0.005, 0.001, 0.0005],
    hidden_units = [32, 64, 128, 256, 512],
    num_blocks = [1, 2, 3],
    dropout_rate = [0.2, 0.4, 0.6],
)

fixed_params = dict(
    num_heads = 1,
    l2_emb = 0.0,
    maxlen = None,
    batch_quota = None,
    seed = 0,
    sampler_seed = 789,
    device = None,
    max_epochs = 400,
    # step_replacement = None, # during tests, max value will be replaced with an optimal one
)
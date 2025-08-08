import optuna
from grids.core import generate_best_config


def generate_config(trial: optuna.Trial) -> dict:

    fixed_params = dict(
        batch_size=128,
        learning_rate=0.005,
        hidden_units=32,
        num_blocks=2,
        dropout_rate=0.6,
        num_heads=1,
        l2_emb=0.0,
        maxlen=50,
        batch_quota=None,
        seed=0,
        sampler_seed=789,
        device="cuda",
        max_epochs=400,
        c=0.39242668485954607,  # obtained from SVD embeddings at 1e^{-12} tolerance
        geom = 'ball',
        bias = True,
        pos_lambda_reg =  1e-06,
        neg_lambda_reg =  0.0001,
        lambda_reg =  1e-08,
        dump_distance_graph = True,
        dump_exp_name = 'lux_sasrecb'
    )
    config = generate_best_config(fixed_params)
    return config

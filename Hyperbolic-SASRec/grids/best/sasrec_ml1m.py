import optuna
from grids.core import generate_best_config


def generate_config(trial: optuna.Trial) -> dict:

    fixed_params = dict(
        batch_size=128,
        learning_rate=0.005,
        hidden_units=32,
        num_blocks=3,
        dropout_rate=0.2,
        num_heads=1,
        l2_emb=0.0,
        maxlen=200,
        batch_quota=None,
        seed=0,
        sampler_seed=789,
        device="cuda",
        max_epochs=400,
        c=0.015283691692054992, # obtained from SVD embeddings at 1e^{-12} tolerance
        geom = 'ball',
        bias = True,
        num_items_sampled = 200,
        pos_lambda_reg = 1e-06,
        neg_lambda_reg = 0.001,
        geometry_c = 0.015283691692054992,
        lambda_reg = 1e-7,
        dump_distance_graph = True,
        dump_exp_name = 'ML1M_sasrecb'
    )
    config = generate_best_config(fixed_params)
    return config

import optuna
from grids.core import generate_best_config


def generate_config(trial: optuna.Trial) -> dict:

    fixed_params = dict(
        batch_size=128,
        learning_rate=0.0005,
        hidden_units=256,
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
        c=0.04589246894247849,  # obtained from SVD embeddings at 1e^{-6} tolerance
    )
    config = generate_best_config(fixed_params)
    return config

# trial_params = dict(
#     #batch_size = [128, 256, 512],
#     batch_size = [128],
#     learning_rate = [0.005, 0.001, 0.0005],
#     hidden_units = [32, 64, 128, 256, 512],
#     num_blocks = [1, 2, 3],
#     dropout_rate = [0.2, 0.4, 0.6],
#     #pos_lambda_reg = [1e-5,1e-4,1e-3,5e-3,1e-2,5e-2,0.1,0.5,1],
#     #neg_lambda_reg = [1e-5,1e-4,1e-3,5e-3,1e-2,5e-2,0.1,0.5,1]
# )

# fixed_params = dict(
#     num_heads = 1,
#     l2_emb = 0.0,
#     maxlen = None,
#     batch_quota = None,
#     seed = 0,
#     sampler_seed = 789,
#     device = None,
#     max_epochs = 400,
#     pos_lambda_reg = 0,
#     neg_lambda_reg = 0
#     # step_replacement = None, # during tests, max value will be replaced with an optimal one
# )



from tkinter import N


trial_params = dict(
    #pos_lambda_reg = [1e-5,1e-4,1e-3,5e-3,1e-2,5e-2,0.1],
    #neg_lambda_reg = [1e-5,1e-4,1e-3,5e-3,1e-2,5e-2,0.1],
    #num_items_sampled = [10,20,30,40,50,60,70,80,90,100]
    pos_lambda_reg = [1e-8,5e-7,1e-7,5e-6,1e-6],
    neg_lambda_reg = [1e-7,1e-6,1e-5,1e-4],
)

fixed_params = dict(
        batch_size=128,
        learning_rate=0.0005,
        hidden_units=64,
        num_blocks=3,
        dropout_rate=0.4,
        num_heads=1,
        l2_emb=0.0,
        maxlen=50,
        batch_quota=None,
        seed=0,
        sampler_seed=789,
        device="cuda",
        max_epochs=400,
        c=0.0408063380676231,
    # step_replacement = None, # during tests, max value will be replaced with an optimal one
)
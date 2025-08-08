
# ========= DATA-RELEATED =========
# data location
data_dir = './data'

# data description
timeid = 'timestamp'
userid = 'userid'

# data preprocessing
use_cached_pcore = True # download ready-to-use amazon files instead of preprocessing
sequence_length_movies = 200
sequence_length_amazon = 50

# data splits
time_offset_q = 0.95
max_test_interactions = 50_000


# ========= MODEL-RELEATED =========
validation_interval = 20 # frequency of validation for iterative models; 1 means validate on each iteration


# ========= STUDY-RELEATED =========
study_direction = 'maximize'
target_metric = 'HR'
max_attempts_multiplier = 3
drop_duplicated_trials = False


# ========= OPTUNA-RELEATED =========
grid_steps_limit = 60
disable_experimental_warnings = True


# databases
# redis_bind_ip = '10.2.1.51' # infiniband ninterface, '127.0.0.1'
# redis_bind_ip = '10.1.255.12' # dedicated VM with redis
# redis_bind_port = 6379
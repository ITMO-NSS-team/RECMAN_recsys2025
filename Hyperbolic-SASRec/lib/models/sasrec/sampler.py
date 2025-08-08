import numpy as np
from numba import njit


@njit
def packed_sequence_batch_sampler(indices, sizes, n_items, batch_size, maxlen, seed, pad_token=None):
    if pad_token == None:
        pad_token = n_items
    n_users = len(sizes) - 1
    users_dtype = np.intp
    items_dtype = indices.dtype
    batch_shape = (batch_size, maxlen)
    numba_seed(seed)
    while True:
        usr = np.empty(batch_size, dtype=users_dtype)
        seq = np.full(batch_shape, pad_token, dtype=items_dtype)
        pos = np.full(batch_shape, pad_token, dtype=items_dtype)
        neg = np.full(batch_shape, pad_token, dtype=items_dtype)
        for i in range(batch_size):
            usr[i] = user = np.random.randint(n_users)
            user_items = indices[sizes[user]:sizes[user+1]]
            sample_fill(user_items, n_items, maxlen, seq[i], pos[i], neg[i])
        yield usr, seq, pos, neg


@njit
def typed_sequence_batch_sampler(user_train, n_items, batch_size, maxlen, seed, pad_token=None):
    if pad_token == None:
        pad_token = n_items
    n_users = len(user_train)
    users_dtype = np.intp
    items_dtype = np.int32
    batch_shape = (batch_size, maxlen)
    numba_seed(seed)
    while True:
        usr = np.empty(batch_size, dtype=users_dtype)
        seq = np.full(batch_shape, pad_token, dtype=items_dtype)
        pos = np.full(batch_shape, pad_token, dtype=items_dtype)
        neg = np.full(batch_shape, pad_token, dtype=items_dtype)
        for i in range(batch_size):
            usr[i] = user = np.random.randint(n_users)
            user_items = user_train[user]
            sample_fill(user_items, n_items, maxlen, seq[i], pos[i], neg[i])
        yield usr, seq, pos, neg

@njit
def numba_seed(seed):
    np.random.seed(seed)

@njit
def sample_fill(user_items, n_items, maxlen, seq, pos, neg):
    nxt = user_items[-1]
    idx = maxlen - 1
    ts = set(user_items)
    for i in user_items[:-1][::-1]:
        seq[idx] = i
        pos[idx] = nxt
        neg[idx] = random_neq(n_items, ts)
        nxt = i
        idx -= 1
        if idx == -1:
            break

@njit
def random_neq(n, s):
    t = np.random.randint(n)
    while t in s:
        t = np.random.randint(n)
    return t

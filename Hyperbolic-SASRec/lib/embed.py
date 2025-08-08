import numpy as np
from scipy.sparse.linalg import svds
from lib.data.processor import DataSet, prepare_data


def get_item_representations(dataset_name, dim, time_offset=None, return_singular_values=False):
    datapack, *_ = prepare_data(dataset_name, time_offset)
    dataset = DataSet(datapack, name=dataset_name, train_format='sparse')
    n_samples, n_objects = dataset.train.shape
    if dim >= min(n_objects, n_samples):
        _, S, VT = np.linalg.svd(dataset.train.A, full_matrices=False)
    else:
        _, S, VT = svds(dataset.train, k=dim, return_singular_vectors='vh')
    # the order in which the singular values are returned is not guaranteed
    sorted_idx = np.argsort(S)[::-1] # ensuring descending order for correct truncation
    sigmas = S[sorted_idx]
    coeffs = VT[sorted_idx].T
    embeddings = np.ascontiguousarray(coeffs * sigmas)
    if not return_singular_values:
        return embeddings
    return embeddings, sigmas
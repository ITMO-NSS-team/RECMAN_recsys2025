import json
import os
import sys
import importlib
from contextlib import contextmanager
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from numba import types
from numba.typed import Dict
import torch

from polara.tools.timing import format_elapsed_time


def validate_list(arg: Union[int,list,tuple]) -> list:
    '''
    Validates if an argument is a list or tuple.
    If passed an integer, it converts it into a single element list. 
    Returns the validated list/tuple.
    '''
    if isinstance(arg, int):
        arg = [arg]
    if not isinstance(arg, (list, tuple)):
        raise ValueError('must be a list or tuple')
    return arg


def try_tuncate(sizes: list, max_size: int, include_boundary: bool=True) -> list:
    '''
    Truncates a list of values if any of the values exceeds the maximum size. 
    If include_boundary is True, then the largest value that exceeds the maximum size
    is included in the returned list.
    
    '''
    if max(sizes) > max_size:
        truncated = [n for n in sizes if n <= max_size]
        if not truncated:
            truncated = [max_size]
        if include_boundary and (max(truncated) < max_size):
            truncated.append(max_size)
        sizes = truncated
    return sizes


def fix_torch_seed(seed, conv_determinism=True):
    '''
    Notes:
    -----
    It doesn't fix the CrossEntropy loss non-determinism, to check it set `torch.use_deterministic_algorithms(True)`.
    For more details, see
    https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180
    
    The `conv_determinism` settings may affect computational performance, see
    https://pytorch.org/docs/stable/notes/randomness.html:
    
    Also note that it doesn't fix possible non-determinism in loss calculation, see:
    https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180/8    
    
    For debugging use torch.use_deterministic_algorithms(True)
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if conv_determinism:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_torch_device(device_name=None):
    if device_name is None:
        device_name = 'cpu'
        if torch.cuda.is_available():
            device_name = f'cuda:{torch.cuda.current_device()}'
    device = torch.device(device_name)
    return device


def topidx(arr, topn):
    parted = np.argpartition(arr, -topn)[-topn:]
    return parted[np.argsort(-arr[parted])]


# taken from https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
# folowing open issue https://github.com/pytorch/pytorch/issues/18220
@contextmanager
def evaluating(net):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


def matrix_from_observations(data: pd.DataFrame, userid: str, itemid: str, dtype=None):
    useridx = data[userid]
    itemidx = data[itemid]
    values = np.ones_like(useridx)
    if dtype is None:
        dtype = 'f8'
    return csr_matrix((values, (useridx, itemidx)), dtype=dtype)


def matrix_from_sequences(data: Union[dict, pd.Series], data_index: dict, dtype=None):
    shape = (len(data_index['users']), len(data_index['items']))
    indices = []
    indptr = [0]
    for user in range(shape[0]):
        seq = data.get(user, [])
        indices.extend(seq)
        indptr.append(indptr[user] + len(seq)) 
    values = np.ones_like(indices)
    if dtype is None:
        dtype = 'f8'
    return csr_matrix((values, indices, indptr), shape=shape, dtype=dtype)


def rescale_matrix(matrix, scaling):
    '''
    Function to normalize sparse rating matrix according to item popularity.
    '''
    if scaling == 1: # no scaling (standard SVD case)
        result = matrix
    # we calculate item popularity according to the frequency of interactions:
    item_popularity = matrix.getnnz(axis=0)
    norm = np.sqrt(item_popularity)
    scaling_values = np.power(norm, scaling-1, where=norm != 0)
    scaling_matrix = diags(scaling_values)
    result = matrix.dot(scaling_matrix)
    return result, scaling_values


def import_source_as_module(source_path: str, verbose: Optional[bool] = False):
    'Importing module from a specified path.'
    'See https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly'
    _, file_name = os.path.split(source_path)
    module_name = os.path.splitext(file_name)[0]
    module_spec = importlib.util.spec_from_file_location(module_name, source_path)
    if module_name in sys.modules:
        if verbose:
            print(f'Module {module_name} is already imported!')
        module = sys.modules[module_name]
    else:
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
    return module

def to_numba_dict(data: Union[dict, pd.Series]):
    numba_data = Dict.empty(
        key_type=types.int64, # userid
        value_type=types.int32[:], # items
    )
    for userid, items in data.items():
        numba_data[userid] = np.array(items, dtype=np.int32)
    return numba_data


def dump_intermediate_results(results: dict, filename: str, indent: int=4):
    with open(filename, 'a+') as f:
        if os.stat(filename).st_size == 0:
            f.write('[')
        else:
            f.seek(0, os.SEEK_END)
            f.seek(f.tell() - 1, os.SEEK_SET)
            f.truncate()
            f.write(', ')
        json.dump(results, f, indent=indent)
        f.write(']')

def show_average_time(timings):
    time_mean = format_elapsed_time(np.mean(timings))
    time_std = format_elapsed_time(np.std(timings))
    return f'{time_mean}±{time_std}'
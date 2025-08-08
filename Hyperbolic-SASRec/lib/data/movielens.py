import os
from typing import Optional
from io import BytesIO
from zipfile import ZipFile
from urllib.parse import urlparse
from urllib import request

import pandas as pd

from lib import defaults


BASE_URL = 'https://files.grouplens.org/datasets/movielens'
DATASETS = set([
    'ml-1m',
    'ml-10m',
    'ml-20m',
    'ml-25m',
    'ml-latest',
])

def get_movielens_data(dataset_id: Optional[str] = None, pcore=None):
    '''Downloads movielens data and stores it in pandas dataframe.
    '''
    zip_contents, data_name = get_movielens_data_info(dataset_id)
    # loading data into memory
    with ZipFile(zip_contents) as zfile:
        zip_files = pd.Series(zfile.namelist())
        zip_file = zip_files[zip_files.str.contains('ratings')].iat[0]
        is_new_format = ('latest' in zip_file) or ('20m' in zip_file) or ('25m' in zip_file)
        header = 0 if is_new_format else None
        delimiter = ','
        zdata = zfile.read(zip_file).replace(b'::', delimiter.encode()) # makes data compatible with pandas c-engine
        fields = [defaults.userid, 'movieid', 'rating', defaults.timeid]
        dtypes = {defaults.userid: int, 'movieid': int, 'rating': float, defaults.timeid: int}
        ml_data = pd.read_csv(
            BytesIO(zdata),
            sep = delimiter,
            header = header,
            engine = 'c',
            names = fields,
            usecols = fields,
            dtype = dtypes
        )
    return ml_data, data_name


def get_movielens_data_info(dataset: Optional[str] = None):
    if dataset is None:
        dataset = 'ml-1m'
    
    if not isinstance(dataset, str):
        raise ValueError(f'Expected a string variable, got {type(dataset)=}.')
    
    if dataset.endswith('.zip'):
        if os.path.exists(dataset): # if it's a local file - return path to it
            data_name, _ = os.path.splitext(os.path.basename(dataset))
            return dataset, data_name
        zip_file_url = dataset # assume it's a path to a file on remote server
        data_name, _ = os.path.splitext(os.path.basename(urlparse(zip_file_url).path))
    else: # handle short dataset names, e.g. ML-1M, ML-20M
        data_name = dataset.lower()
        zip_file_url = f'{BASE_URL}/{data_name}.zip'
    
    with request.urlopen(zip_file_url) as zip_response:
        zip_contents = BytesIO(zip_response.read())
    return zip_contents, data_name

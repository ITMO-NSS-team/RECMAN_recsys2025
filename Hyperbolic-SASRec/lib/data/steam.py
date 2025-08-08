from ast import literal_eval
import gzip
from typing import Optional
from urllib import request

import pandas as pd

from lib import defaults
from .tools import dates_to_timestamps, pcore_filter


BASE_URL = 'http://cseweb.ucsd.edu/~wckang'
DATASETS = set([
    'steam'
])

def get_steam_data(dataset_id: Optional[str] = None, pcore: Optional[bool] = None):
    contents, data_name = get_steam_data_info(dataset_id)
    fields = ['username', 'product_id', 'date']
    raw_data = (
        pd.DataFrame
        .from_records(
            parse_lines(contents, fields),
            columns = fields
        )
        .drop_duplicates(subset=['username', 'product_id'], keep='last')
        .rename(columns={'username': defaults.userid})
    )
    raw_data.loc[:, defaults.timeid] = dates_to_timestamps(raw_data['date'])
    if (pcore is None) or (pcore <= 1):
        return raw_data, data_name

    data_name = f'{data_name}_{pcore}'
    pcore_data = pcore_filter(raw_data, pcore, defaults.userid, 'product_id')
    return pcore_data, data_name


def get_steam_data_info(dataset_id: Optional[str] = None):
    if dataset_id is None:
        dataset_id = 'steam'
    
    if not isinstance(dataset_id, str):
        raise ValueError(f'Expected a string variable, got {type(dataset_id)=}.')
    
    data_name = dataset_id
    url = f'{BASE_URL}/{data_name}_reviews.json.gz'
    contents, _ = request.urlretrieve(url) # this may take some time depending on your internet connection    
    return contents, data_name

def parse_lines(path, fields):
    with gzip.open(path, 'rt') as gz:
        for line in gz:
            dct = literal_eval(line.strip())
            yield {key: dct[key] for key in fields}

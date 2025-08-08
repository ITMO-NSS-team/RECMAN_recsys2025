import gzip
import json
from typing import Optional
from urllib import request
from urllib.error import URLError
import warnings
import pandas as pd

from lib import defaults
from .tools import pcore_filter


BASE_URL = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/'
BASE_URL_ALT = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles'
DATASETS = set([
    'AMAZON_FASHION',
    'All_Beauty',
    'Appliances',
    'Arts_Crafts_and_Sewing',
    'Automotive',
    'Books',
    'CDs_and_Vinyl',
    'Cell_Phones_and_Accessories',
    'Clothing_Shoes_and_Jewelry',
    'Digital_Music',
    'Electronics',
    'Gift_Cards',
    'Grocery_and_Gourmet_Food',
    'Home_and_Kitchen',
    'Industrial_and_Scientific',
    'Kindle_Store',
    'Luxury_Beauty',
    'Magazine_Subscriptions',
    'Movies_and_TV',
    'Musical_Instruments',
    'Office_Products',
    'Patio_Lawn_and_Garden',
    'Pet_Supplies',
    'Prime_Pantry',
    'Software',
    'Sports_and_Outdoors',
    'Tools_and_Home_Improvement',
    'Toys_and_Games',
    'Video_Games', 
    'Digital_Music_2023',
    'pinterest20'
])
ALIAS = {
    'amz-b': 'All_Beauty',
    'amz-g': 'Toys_and_Games',
    'amz-vg': 'Video_Games',
    'amz-e': 'Electronics',
    'amazon_fashion': 'AMAZON_FASHION',
}


def get_amazon_data(dataset_id: Optional[str] = None, pcore: Optional[bool] = None):
    data_name = get_amazon_data_info(dataset_id)
    # data_name_stripped = data_name.split('Amazon_')[-1]
    if pcore is None or pcore <= 1:
        amz_data = get_ratings(data_name)
    elif pcore == 5 and defaults.use_cached_pcore:
        data_name = f'{data_name}_5'
        amz_data = get_reviews(data_name)
    else:
        raw_data = get_ratings(data_name)
        amz_data = pcore_filter(raw_data, pcore, 'userid', 'asin')
        data_name = f'{data_name}_{pcore}'
    return amz_data, data_name


def get_amazon_data_info(dataset_id: Optional[str] = None):
    if dataset_id is None:
        dataset_id = 'amz-b'
    
    if not isinstance(dataset_id, str):
        raise ValueError(f'Expected a string variable, got {type(dataset_id)=}.')
    
    try:
        data_name = ALIAS[dataset_id.lower()]
    except KeyError:
        data_name = dataset_id
    return data_name


def get_reviews(data_name):
    data_url = f'{BASE_URL}/{data_name}.json.gz'
    try:
        tmp_file, _ = request.urlretrieve(data_url)
    except URLError:
        import ssl
        warnings.warn(
            'Unable to load data securely due to a cetificate problem! '
            'Disabling SSL certificate check.', UserWarning
        )
        # potentially unsafe
        ssl._create_default_https_context = ssl._create_unverified_context
        tmp_file, _ = request.urlretrieve(data_url)
    fields = ['reviewerID', 'asin', 'unixReviewTime']
    pcore_data = pd.DataFrame.from_records(
        parse_lines(tmp_file, fields),
        columns = fields,
    ).rename(columns={'reviewerID': defaults.userid, 'unixReviewTime': defaults.timeid})
    return pcore_data

def parse_lines(path, fields):
    with gzip.open(path, 'rt') as gz:
        for line in gz:
            yield json.loads(line, object_hook=lambda dct: tuple(dct.get(key, dct) for key in fields))


def get_ratings(data_name):
    data_url = f'{BASE_URL}/{data_name}.csv'
    try:
        ratings_data = pd.read_csv(
            data_url,
            header=None,
            # fields are item, user, rating, timestamp, see:
            # https://nijianmo.github.io/amazon/index.html#:~:text=(item%2Cuser%2Crating%2Ctimestamp)%20tuples
            names=['asin', defaults.userid, 'rating', defaults.timeid]
        )
    except URLError:
        import ssl
        warnings.warn(
            'Unable to load data securely due to a cetificate problem! '
            'Disabling SSL certificate check.', UserWarning
        )
        # potentially unsafe
        ssl._create_default_https_context = ssl._create_unverified_context
        ratings_data = pd.read_csv(
            data_url,
            header=None,
            names=[defaults.userid, 'asin', 'rating', defaults.timeid]
        )        
    return ratings_data
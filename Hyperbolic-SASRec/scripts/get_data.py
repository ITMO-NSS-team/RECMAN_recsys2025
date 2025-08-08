import os
import argparse

import libcontext
from lib import defaults
from lib.data.processor import entity_names
from lib.data import movielens as ml, amazon as amz, steam


def check_dirs():
    data_dir = defaults.data_dir
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    for path in ['raw', 'clean', 'results']:
        dir_path = os.path.join(data_dir, path)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)


def load_data(dataset, loader, pcore=None):
    data, data_name = loader(dataset, pcore)
    dest = os.path.join(defaults.data_dir, f'raw/{data_name}.gz')
    data.loc[:, entity_names(data_name)].to_csv(dest, index=False)
    print(f'{data_name} data is processed and saved to {dest}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m', type=str)
    parser.add_argument('--pcore', default=5, type=int)
    args = parser.parse_args()
    
    check_dirs()
    dataset_id = args.dataset.strip().replace(' ', '_')
    if dataset_id in ml.DATASETS:
        loader = ml.get_movielens_data
    elif (dataset_id.lower() in amz.ALIAS) or (dataset_id in amz.DATASETS):
        loader = amz.get_amazon_data
    elif dataset_id in steam.DATASETS:
        loader = steam.get_steam_data        
    else:
        raise ValueError(f'Unrecognized dataset: {args.dataset}')
    load_data(dataset_id, loader, args.pcore)
    
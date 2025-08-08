import os
import argparse
import logging
from datetime import datetime

import numpy as np
from tqdm import tqdm

import libcontext
from lib import defaults, logger
from lib.delta import batched_delta_hyp
from lib.utils import dump_intermediate_results, try_tuncate, validate_list
from lib.embed import get_item_representations


def calculate_deltas(filename, embeddings, dims, sample_size, ntrials, seed=None, economic=True, max_workers=0):
    for dim in dims:
        vs_emb = embeddings[:, :dim]
        batch_results = batched_delta_hyp(vs_emb, ntrials, sample_size, seed, economic, max_workers)
        batch_results = dict(zip(['delta', 'diam'], zip(*batch_results)))
        delta_mean = np.mean(batch_results['delta'])
        delta_std = np.std(batch_results['delta'])
        diam_mean = np.mean(batch_results['diam'])
        diam_std = np.std(batch_results['diam'])
        logger.info(f'dim {dim} delta = {delta_mean:.4f}±{delta_std:.4f} with diameter = {diam_mean:.4f}±{diam_std:.4f}')
        intermediate_results = {
            "sample_size": sample_size,
            "dim": dim,
            **batch_results
        }
        dump_intermediate_results(intermediate_results, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--time_offset', default=0.95, type=float)
    parser.add_argument('--nsamples', nargs='*', type=int, default=[500, 1000, 1500, 2000, 2500, 3000, 4000])
    parser.add_argument('--ranks', nargs='*', type=int, default=[32, 64, 128, 256, 512, 768, 1024])
    parser.add_argument('--ntrials', default=48, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--full_matrix', default=False, action="store_true")
    parser.add_argument('--max_workers', default=0, type=int)
    parser.add_argument('--filename', default=None, type=str)
    parser.add_argument('--verbose', default=False, action="store_true")    
    run_args = parser.parse_args()

    num_samples = validate_list(run_args.nsamples)
    svd_ranks = validate_list(run_args.ranks)
    
    logger.info(f'Calculating {run_args.dataset} embeddings of dim {max(svd_ranks)}')
    embeddings = get_item_representations(run_args.dataset, max(svd_ranks), run_args.time_offset)
    n_objects, max_dim = embeddings.shape
    logger.info(f'Computed representation for {n_objects} samples of {max_dim} size from the dataset.')
    
    num_samples = try_tuncate(num_samples, n_objects, include_boundary=True)
    svd_ranks = try_tuncate(svd_ranks, max_dim, include_boundary=True)
    
    filename = run_args.filename
    if not filename:
        ts = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        filename = os.path.join(defaults.data_dir, f'results/delta_{run_args.dataset}_{ts}.txt')
    logger.info(f'Results will be saved to {filename}.')
    
    iterator = lambda x: x
    if not run_args.verbose:
        logger.setLevel(logging.WARNING)
        iterator = tqdm

    for sample_size in iterator(num_samples):
        calculate_deltas(
            filename, embeddings, svd_ranks, sample_size, run_args.ntrials,
            run_args.seed, not run_args.full_matrix, run_args.max_workers
        )


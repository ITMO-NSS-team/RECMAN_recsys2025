from collections import defaultdict
from collections.abc import Iterable, Callable
from math import sqrt, log2
from typing import Optional

import numpy as np
import pandas as pd

from polara.tools.timing import track_time

from lib.data.processor import DataSet
from lib.models.base import RecommenderModel


class Evaluator:
    def __init__(
        self,
        dataset: DataSet,
        topn: int,
        evaluation_callback: Optional[Callable[..., None]] = None
    ) -> None:
        self.dataset = dataset
        self.topn = topn
        self.evaluation_callback = evaluation_callback
        self._results = {}
        self._last_used_key = None
        self.evaluation_time = []
    
    def submit(
        self,
        model: RecommenderModel,
        step: Optional[int] = None,
        args: Optional[tuple] = ()
    ) -> None:
        with track_time(self.evaluation_time, verbose=False):
            if self.dataset.format_exists('test', 'sequential'):
                self._results[step] = evaluate_on_sequences(model, self.dataset, self.topn)
            else:
                self._results[step] = evaluate(model, self.dataset, self.topn)
        self._last_used_key = step
        if (step is not None) and (self.evaluation_callback is not None): # suppot iterative algorithms
            self.evaluation_callback(self._results, step, *args)
    
    @property
    def results(self):
        if not self._results: # handle uninitialized evaluator (i.e. before calling `submit`)
            return None # will prevent `log_attributes` from failing due to empty dict
        
        if self._last_used_key is None: # handle non-iterative algorithms results
            assert len(self._results) == 1, 'Iterative algorithms must provide only integer step values.'
            return self._results[None] # directly return results dataframe
        return self._results # return history of results

    @property
    def most_recent_results(self):
        return self._results[self._last_used_key]


def evaluate(model: RecommenderModel, dataset: DataSet, topn: int):
    with dataset.formats(train='sequential'): # model may have used a different format - restoring
        train_data = dataset.train
    test_data = dataset.test
    if isinstance(test_data, pd.DataFrame): # pack all data in a single step
        test_data = {0: test_data.itertuples(index=False, name=None)} 
    # collect evaluation results at each step
    step_scores, step_stder2, n_unique_recs =  list(zip(
        *(evaluate_step(model, train_data, test_seq, topn)
        for step, test_seq in test_data.items())
    ))
    # compute averages
    average_scores = pd.DataFrame.from_records(step_scores).mean()
    average_errors = pd.DataFrame.from_records(step_stder2).mean().pow(0.5) # s = sqrt(sum((si / n)**2)
    average_scores.loc['COV'] = np.mean(n_unique_recs) / len(dataset.item_index)
    average_errors.loc['COV'] = sample_ci(n_unique_recs)
    averaged_results = pd.concat(
        [average_scores, average_errors],
        keys = ['score', 'error'],
        axis = 1,
    ).rename(index=lambda x: f'{x}@{topn}'.upper())
    return averaged_results


def evaluate_step(model: RecommenderModel, train: dict, test_seq: Iterable, topn: int):
    results = []
    unique_recommendations = set()
    seen_test = defaultdict(list)
    for user, test_item in test_seq:
        seen_test_items = seen_test[user]
        seq = train.get(user, []) + seen_test_items
        if seq:
            predicted_items = model.recommend(seq, topn, user=user)
            (hit_index,) = np.where(predicted_items == test_item)
            hit_scores = compute_metrics(hit_index)
            results.append(hit_scores.values())
            unique_recommendations.update(predicted_items)
        seen_test_items.append(test_item) # extend seen items for next step prediction
    results = pd.DataFrame.from_records(results, columns=hit_scores.keys())
    step_scores = results.mean()
    step_stder2 = (results - step_scores).pow(2).mean() / (results.shape[0] - 1) # sum(xi - x)**2/(n*(n-1))
    return step_scores, step_stder2, len(unique_recommendations)


def compute_metrics(hits):
    try:
        hit_index = hits.item()
    except ValueError: # expected single element - got 0 or >1
        if hits.size > 1:
            raise ValueError("Holdout must contain single item only!")
        return {'hr': 0., 'mrr': 0., 'ndcg': 0.}
    return {'hr': 1., 'mrr': 1. / (hit_index+1.), 'ndcg': 1. / log2(hit_index+2.)}    


def sample_ci(scores, coef=2.776):
    n = len(scores)
    if n < 2: # unable to estimate ci
        return np.nan
    return coef * np.std(scores, ddof=1) / sqrt(n)


def evaluate_on_sequences(model, dataset, topn=10):
    with dataset.formats(train='sequential', test='sequential'): # get data in appropiate format
        train_data = dataset.train
        test_data = dataset.test
    
    cum_hits = 0
    cum_reciprocal_ranks = 0.
    cum_discounts = 0.
    unique_recommendations = set()
    total_count = 0
    for user, test_seq in test_data.items():
        try:
            seen_seq = train_data[user]
        except KeyError: # handle users with no history - advance by 1 item
            seen_seq = test_seq[:1]
            test_seq = test_seq[1:]
        num_predictions = len(test_seq)
        if not num_predictions: # if no test items left - skip user
            continue
        predicted_items = model.recommend_sequential(test_seq, seen_seq, topn, user=user)
        hit_steps, hit_index = np.where(predicted_items == np.atleast_2d(test_seq).T)
        unique_recommendations.update(predicted_items.ravel())

        num_hits = hit_index.size
        if num_hits:
            cum_hits += num_hits
            cum_reciprocal_ranks += np.sum(1. / (hit_index+1))
            cum_discounts += np.sum(1. / np.log2(hit_index+2))
        total_count += num_predictions

    hr = cum_hits / total_count
    mrr = cum_reciprocal_ranks / total_count
    dcg = cum_discounts / total_count
    cov = len(unique_recommendations) / len(dataset.item_index)
    results = pd.DataFrame(
        data = {'score': [hr, mrr, dcg, cov]},
        index = [f'{metric}@{topn}' for metric in ['HR', 'MRR', 'NDCG', 'COV']]
    )
    return results
from __future__ import annotations
from abc import ABC, abstractmethod
import typing
from typing import Any, Optional, Union

import numpy as np
from lib.utils import topidx

if typing.TYPE_CHECKING:
    from lib.evaluation import Evaluator


class InvalidInputData(Exception): pass


class RecommenderModel(ABC):
    @abstractmethod
    def fit(self, data: Any, evaluator: Optional[Evaluator] = None):
        '''
        Fit model on train data.
        =========================
        If the model is iterative, `evaluator` argument must also be provided.
        `data` can be of any type, which is define by subclass requirements.
        Specifications of input data types are defined in `train_data_format`
        function from `lib.models.learning`.
        '''

    def recommend(self, seen_seq: Union[list, np.ndarray], topn: int, *, user: Optional[int] = None):
        '''Given an item sequence, predict top-n candidates for the next item.'''
        predictions = self.predict(seen_seq, user=user)
        np.put(predictions, seen_seq, -np.inf)
        predicted_items = topidx(predictions, topn)
        return predicted_items

    @abstractmethod
    def predict(self, seen_seq: Union[list, np.ndarray], *, user: Optional[int] = None):
        '''Takes 1d sequence as input and returns prediction scores.'''


    def recommend_sequential(
        self,
        target_seq: Union[list, np.ndarray],
        seen_seq: Union[list, np.ndarray],
        topn: int,
        *,
        user: Optional[int] = None
    ):
        '''Given an item sequence and a sequence of next target items,
        predict top-n candidates for each next step in the target sequence.
        '''
        predictions = self.predict_sequential(target_seq[:-1], seen_seq, user=user)
        predictions[:, seen_seq] = -np.inf
        for k in range(1, predictions.shape[0]):
            predictions[k, target_seq[:k]] = -np.inf
        predicted_items = np.apply_along_axis(topidx, 1, predictions, topn)
        return predicted_items

    @abstractmethod
    def predict_sequential(
        self,
        target_seq: Union[list, np.ndarray],
        seen_seq: Union[list, np.ndarray],
        *,
        user: Optional[int] = None
    ):
        '''
        Returns scores array which rows correspond to predictions on sequences 
        obtained by successively extending `seen_seq` with items from `test_seq`
        one by one.
        '''

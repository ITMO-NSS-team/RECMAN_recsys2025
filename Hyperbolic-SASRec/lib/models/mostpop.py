import numpy as np
from scipy.sparse.base import spmatrix

from .base import RecommenderModel
from lib.evaluation import Evaluator


def train_validate(config: dict, evaluator: Evaluator) -> None:
    dataset = evaluator.dataset
    model = MostPopModel(config)
    model.fit(dataset.train)
    evaluator.submit(model)


class MostPopModel(RecommenderModel):
    def __init__(self, config) -> None:
        self.config = config
    
    def fit(self, data: spmatrix):
        self.item_popularity = data.getnnz(axis=0)

    def predict(self, seq, user):
        return np.asarray(self.item_popularity, dtype='f4')

    def predict_sequential(self, target_seq, seen_seq, user):
        scores = np.repeat(
            np.asarray([self.item_popularity], dtype='f4'),
            len(target_seq) + 1,
            axis=0
        )
        return scores
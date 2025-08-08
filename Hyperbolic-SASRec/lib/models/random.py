import numpy as np
from scipy.sparse.base import spmatrix

from .base import RecommenderModel
from lib.evaluation import Evaluator


def train_validate(config: dict, evaluator: Evaluator) -> None:
    dataset = evaluator.dataset
    model = RandomRecModel(config)
    model.fit(dataset.train)
    evaluator.submit(model)


class RandomRecModel(RecommenderModel):
    def __init__(self, config) -> None:
        self.config = config
        self.n_items = None
        self.random_state = None
    
    def fit(self, data: spmatrix):
        _, self.n_items = data.shape
        self.random_state = np.random.RandomState(self.config.get('seed', None))

    def predict(self, seq, user):
        return self.random_state.rand(self.n_items)

    def predict_sequential(self, target_seq, seen_seq, user):
        return self.random_state.rand(len(target_seq) + 1, self.n_items)
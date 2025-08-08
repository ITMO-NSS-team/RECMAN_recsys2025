import numpy as np

from lib.models.base import RecommenderModel
from lib.models.learning import trainer


def train_validate(config, evaluator):
    '''
    Train model and return best scores across all iterations.
    '''
    dataset = evaluator.dataset
    n_items = len(dataset.item_index)
    model = BlankModel(config, n_items)
    model.fit(dataset.train, evaluator)


class BlankModel(RecommenderModel):
    def __init__(self, config, n_items):
        self.config = config
        self.n_items = n_items
        self.random_state = np.random.RandomState(self.config.get('seed', None))
    
    def fit(self, data, evaluator): # from base abstract class
        trainer(self, evaluator) # common boilerplate for neural networks
    
    def train_epoch(self): # to be used within the trainer
        return self.random_state.rand()

    def predict(self, seq, user): # from base abstract class
        return self.random_state.rand(self.n_items)

    def predict_sequential(self, target_seq, seen_seq, user):
        '''
        Note: even with fixed seed, `predict` and `predict_sequential`
        are still incompatible and won't produce same results by design;
        fixed seed only guarantees reproducibility withing either consequtive 
        `predict` or consequtive `predict_sequential` calls.
        '''
        return self.random_state.rand(len(target_seq) + 1, self.n_items)        
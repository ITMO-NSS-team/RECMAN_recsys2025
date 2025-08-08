import numpy as np
from scipy.sparse import isspmatrix
from scipy.sparse.base import spmatrix as sparse_matrix
from scipy.sparse.linalg import svds

try:
    from sklearn.utils.extmath import randomized_svd
except ImportError:
    randomized_svd = None

from .base import RecommenderModel
from lib.evaluation import Evaluator
from lib.utils import rescale_matrix

class SVDError(Exception): pass


def train_validate(config: dict, evaluator: Evaluator) -> None:
    dataset = evaluator.dataset
    model = SVDModel(config)
    model.fit(dataset.train)
    evaluator.submit(model)


class ItemProjectorMixin:
    def store_item_projector(self):
        item_factors = self.item_factors
        if self.rescaled_prediction:
            self.projector = {
                'item_side': item_factors / self.scaling_weights[:, np.newaxis],
                'user_side': item_factors * self.scaling_weights[:, np.newaxis]
            }

    def get_item_projector(self):
        if self.rescaled_prediction:
            vi = self.projector['item_side']
            vu = self.projector['user_side']
        else:
            vi = vu = self.item_factors
        return vi, vu


class SVDModel(ItemProjectorMixin, RecommenderModel):
    def __init__(self, config) -> None:
        self.rank = config['rank']
        self.scaling = config['scaling']
        randomized = config['randomized']
        if randomized and randomized_svd is None:
            raise SVDError('Randomized SVD is unavailable')
        self.randomized = randomized and (randomized_svd is not None)
        self.seed = config['rnd_svd_seed'] if self.randomized else None
        self.scaling_weights = None
        self.rescaled_prediction = False
        self.projector = None
    
    def fit(self, data: sparse_matrix):
        matrix = self.data_to_matrix(data)
        scaled_matrix, self.scaling_weights = rescale_matrix(matrix, self.scaling)
        if self.randomized:
            _, sval, item_factors_t = randomized_svd(scaled_matrix, self.rank, random_state=self.seed)
        else:
            _, sval, item_factors_t = svds(scaled_matrix, self.rank, return_singular_vectors='vh')
        sidx = np.argsort(sval)[::-1]
        self.item_factors = np.ascontiguousarray(item_factors_t[sidx, :].T)
        self.store_item_projector()
    
    def data_to_matrix(self, data):
        if isspmatrix(data):
            return data.tocsr()
        raise NotImplementedError

    def predict(self, seq, user):
        vi, vu = self.get_item_projector()
        user_profile = vu[seq, :].sum(axis=0)
        scores = vi @ user_profile
        return scores

    def predict_sequential(self, target_seq, seen_seq, user):
        vi, vu = self.get_item_projector()
        user_profile = vu[seen_seq, :].sum(axis=0)
        test_sequence = np.vstack([user_profile, vu[target_seq, :]])
        test_profile = test_sequence.cumsum(axis=0)
        scores = test_profile @ vi.T
        return scores
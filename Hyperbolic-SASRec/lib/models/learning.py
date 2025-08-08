import importlib
from collections.abc import Callable

from lib import defaults
from .base import RecommenderModel
from lib.evaluation import Evaluator
from lib.optunatools import EarlyStopping


def trainer(model: RecommenderModel, evaluator: Evaluator) -> None:
    '''
    A standard boilerplate for training a neural network.
    ====================================================

    Supports pruning/reporting based on callbacks. Callback
    must be available in the `evaluator`.
    '''
    max_epochs = model.config['max_epochs']
    validation_interval = defaults.validation_interval
    assert validation_interval <= max_epochs, 'Number of epochs is too small. Won\'t validate.'
    # TODO warn if max_epochs % validation_interval != 0
    for epoch in range(max_epochs):
        loss = model.train_epoch()
        if (epoch + 1) % validation_interval == 0:
            try:
                evaluator.submit(model, step=epoch, args=(loss,))
            except EarlyStopping:
                break


def import_model_routine(model_name: str, name: str) -> Callable[..., None]:
    try:
        module_name = f'lib.models.{model_name.lower()}'
    except KeyError:    
        raise ValueError(f'Unrecognized model name {model_name}.')
    module = importlib.import_module(module_name)
    return getattr(module, name)


def train_data_format(model_name):
    sparse = ['svd', 'random', 'mostpop']
    sequential_packed = ['sasrec', 'sasrecb', 'hypsasrec', 'hypsasrecb','sasrec_manifold','sasrecb_manifold']
    sequential = []
    sequential_typed = []
    if model_name in sparse:
        return 'sparse'
    if model_name in sequential:
        return 'sequential' # pandas Series
    if model_name in sequential_packed:
        return 'sequential_packed' # csr-like format
    if model_name in sequential_typed:
        return 'sequential_typed' # numba dict
    return 'default'

def test_data_format(args):
    if args.next_item_only: # e.g., lorentzfm is not suitable for sequential prediction
        return ('interactions', dict(stepwise=True, max_steps=1))
    return 'sequential' # 'sequential' will enable vectorized evaluation, if a model supports it
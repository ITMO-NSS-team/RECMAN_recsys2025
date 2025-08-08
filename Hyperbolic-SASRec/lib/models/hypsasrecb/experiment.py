from functools import partial

import numpy as np
import torch

from lib.models.sasrec.sampler import packed_sequence_batch_sampler
from lib.utils import fix_torch_seed, get_torch_device
from lib.models.base import RecommenderModel
from lib.models.learning import trainer
from lib.evaluation import Evaluator
from .source import HypSASRecBCE


def train_validate(config: dict, evaluator: Evaluator) -> None:
    dataset = evaluator.dataset
    n_items = len(dataset.item_index)
    fix_torch_seed(config.get('seed', None))
    model = HypSASRecBCEModel(config, n_items)
    model.fit(dataset.train, evaluator)


class HypSASRecBCEModel(RecommenderModel):
    def __init__(self, config: dict, n_items: int):
        self.n_items = n_items
        self.config = config
        self.device = get_torch_device(self.config.pop('device', None))
        self._model = HypSASRecBCE(self.config, self.n_items).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.config['learning_rate'], betas=(0.9, 0.98)
        )
        self.sampler = None
        self.n_batches = None
    
    @property
    def model(self):
        return self._model

    def fit(self, data: tuple, evaluator: Evaluator):
        indices, sizes = data
        self.sampler = packed_sequence_batch_sampler(
            indices, sizes, self.n_items,
            batch_size = self.config['batch_size'],
            maxlen = self.config['maxlen'],
            seed = self.config['sampler_seed'],
        )
        self.n_batches = (len(sizes) - 1) // self.config['batch_size']
        trainer(self, evaluator)

    def train_epoch(self):
        model = self.model
        pad_token = model.pad_token
        criterion, optimizer, sampler, device, n_batches = [
            getattr(self, a) for a in ['criterion', 'optimizer', 'sampler', 'device', 'n_batches']
        ]
        l2_emb = self.config['l2_emb']
        as_tensor = partial(torch.as_tensor, dtype=torch.long, device=device)
        loss = 0
        model.train()
        for _ in range(n_batches):
            _, *seq_data = next(sampler)
            # convert batch data into torch tensors
            seq, pos, neg = [as_tensor(arr) for arr in seq_data]
            pos_logits, neg_logits = model(seq, pos, neg)
            pos_labels = torch.ones(pos_logits.shape, device=device)
            neg_labels = torch.zeros(neg_logits.shape, device=device)
            indices = torch.where(pos != pad_token)
            batch_loss = criterion(pos_logits[indices], pos_labels[indices])
            batch_loss += criterion(neg_logits[indices], neg_labels[indices])
            if l2_emb != 0:
                for param in model.item_emb.parameters():
                    batch_loss += l2_emb * torch.norm(param)**2
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
        model.eval()
        return loss


    def predict(self, seq, user):
        model = self.model
        maxlen = self.config['maxlen']
        device = self.device

        with torch.no_grad():
            log_seqs = torch.full([maxlen], model.pad_token, dtype=torch.int64, device=device)
            log_seqs[-len(seq):] = torch.as_tensor(seq[-maxlen:], device=device)
            log_feats = model.log2feats(log_seqs)
            hyp_feats = model.manifold.expmap0(log_feats[:, -1, :])
            logits = model.poincare_hyperplane(hyp_feats)
        return logits.cpu().numpy()

    def predict_sequential(self, target_seq, seen_seq, user):
        model = self.model
        maxlen = self.config['maxlen']
        device = self.device

        n_seen = len(seen_seq)
        n_targets = len(target_seq)
        seq = np.concatenate([seen_seq, target_seq])

        with torch.no_grad():
            pad_seq = torch.as_tensor(
                np.pad(
                    seq, (max(0, maxlen-n_seen), 0),
                    mode = 'constant',
                    constant_values = model.pad_token
                ),
                dtype = torch.int64,
                device = device
            )
            log_seqs = torch.as_strided(pad_seq[-n_targets-maxlen:], (n_targets+1, maxlen), (1, 1))
            log_feats = model.log2feats(log_seqs)
            hyp_feats = model.manifold.expmap0(log_feats[:, -1, :])
            logits = model.poincare_hyperplane(hyp_feats)
        return logits.cpu().numpy()
        

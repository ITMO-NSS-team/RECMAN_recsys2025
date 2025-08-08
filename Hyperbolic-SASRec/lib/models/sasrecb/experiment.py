from functools import partial

import numpy as np
import torch

from lib.models.sasrec.sampler import packed_sequence_batch_sampler
from lib.utils import fix_torch_seed, get_torch_device
from lib.models.base import RecommenderModel
from lib.models.learning import trainer
from lib.evaluation import Evaluator
from .source import SASRecCE


def train_validate(config: dict, evaluator: Evaluator) -> None:
    dataset = evaluator.dataset
    n_items = len(dataset.item_index)
    fix_torch_seed(config.get('seed', None))
    model = SASRecCEModel(config, n_items)
    model.fit(dataset.train, evaluator)


class SASRecCEModel(RecommenderModel):
    def __init__(self, config: dict, n_items: int):
        self.n_items = n_items
        self.config = config
        self.device = get_torch_device(self.config.pop('device', None))
        self._model = SASRecCE(self.config, self.n_items).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self._model.pad_token).to(self.device)
        self.optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.config['learning_rate'], betas=(0.9, 0.98)
        )
        self.sampler = None
        self.n_batches = None
        self.batch_size = None
        self.gradient_accumulation_steps = None
        self.set_batch_size()

    def set_batch_size(self):
        quota = self.config.get('batch_quota', None)
        batch_size = self.config['batch_size']
        if (not quota) or (quota >= batch_size):
            self.batch_size = batch_size
            self.gradient_accumulation_steps = 1
        elif batch_size % quota == 0:
            self.batch_size = quota
            self.gradient_accumulation_steps = batch_size // quota
        else:
            raise NotImplementedError

    @property
    def model(self):
        return self._model

    def fit(self, data: tuple, evaluator: Evaluator):
        indices, sizes = data
        self.sampler = packed_sequence_batch_sampler(
            indices, sizes, self.n_items,
            batch_size = self.batch_size,
            maxlen = self.config['maxlen'],
            seed = self.config['sampler_seed'],
        )

        self.n_batches = (len(sizes) - 1) // self.batch_size
        trainer(self, evaluator)

    def train_epoch(self):
        model = self.model
        criterion, optimizer, sampler, device, n_batches = [
            getattr(self, a) for a in ['criterion', 'optimizer', 'sampler', 'device', 'n_batches']
        ]
        l2_emb = self.config['l2_emb']
        as_tensor = partial(torch.as_tensor, device=device)

        loss = 0
        model.train()
        for index in range(n_batches):
            _, inputs, target, _ = next(sampler)
            # convert batch data into torch tensors
            inputs = as_tensor(inputs, dtype=torch.int32) # batch x seq.len
            target = as_tensor(target, dtype=torch.long)  # batch x seq.len, CrossEntropy requires `long` ints
            # need to permute output to comply with CrossEntropy inputs shape requirement
            logits = model(inputs).permute(0, 2, 1)  # batch x num.items x seq.len
            batch_loss = criterion(logits, target)
            batch_loss = batch_loss / self.gradient_accumulation_steps
            if l2_emb != 0:
                for param in model.item_emb.parameters():
                    batch_loss += l2_emb * torch.norm(param)**2
            batch_loss.backward()
            if (index + 1) % self.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
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
            logits = model.head(log_feats[:, -1, :])
        return logits.cpu().numpy().squeeze()

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
            logits = model.head(log_feats[:, -1, :])
        return logits.cpu().numpy()
        

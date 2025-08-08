import os

from functools import partial
from math import exp

import numpy as np
import torch

from .source import SASRec
from .sampler import packed_sequence_batch_sampler
from lib.utils import fix_torch_seed, get_torch_device
from lib.models.base import RecommenderModel
from lib.models.learning import trainer
from lib.evaluation import Evaluator
from lib.manifold import hyperbolic_dist, pairseq_dist_affinity, project_embeddings,estimate_curvature_SVD



def train_validate(config: dict, evaluator: Evaluator) -> None:
    dataset = evaluator.dataset
    n_items = len(dataset.item_index)
    fix_torch_seed(config.get('seed', None))
    model = SASRecModel(config, n_items)
    model.fit(dataset.train, evaluator)



class SASRecModel(RecommenderModel):
    def __init__(self, config: dict, n_items: int):
        self.n_items = n_items
        self.config = config
        self.device = get_torch_device(self.config.pop('device', None))
        self._model = SASRec(self.config, self.n_items).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.config['learning_rate'], betas=(0.9, 0.98)
        )
        self.sampler = None
        self.n_batches = None
        self.epoch_n = 0
    
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
        as_tensor = partial(torch.as_tensor, dtype=torch.int32, device=device)
        loss = 0
        model.train()

        single_dist = lambda u,v: hyperbolic_dist(u,v, c = self.config['c']) 

        #clean_dist = lambda u,v: hyperbolic_dist(u,v, c = 1.0) 

        try:
            dump_distance_graph = self.config['dump_distance_graph']
        except Exception:
            dump_distance_graph = False

        for batch_number in range(n_batches):
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


            pos_lambda_man_reg = self.config['pos_lambda_reg']
            try:
                n_items_sampled = self.config['num_items_sampled']
            except Exception:
                n_items_sampled = self.config['maxlen']

            if pos_lambda_man_reg > 0:
                with torch.amp.autocast('cuda'):
                    idx_samples = torch.randint(0, self.config['maxlen'], (n_items_sampled,), device=pos.device)
                    pos_sub = pos[:, idx_samples]
                    #pos_eseq = model.item_emb(pos)

                    pos_eseq = model.item_emb(pos_sub.detach())

                    if dump_distance_graph:
                        
                        pos_dist,pos_affinity = pairseq_dist_affinity(pos_eseq,single_dist,return_dist=True)

                    else:

                        pos_affinity = pairseq_dist_affinity(pos_eseq,single_dist)

                    pos_logits_dist = torch.cdist(pos_logits,pos_logits)**2

                    pos_lap = pos_affinity*pos_logits_dist

                    pos_man_reg = pos_lap.sum()
               
            else:
                pos_man_reg=torch.tensor(0)

            neg_lambda_man_reg = self.config['neg_lambda_reg']

            if neg_lambda_man_reg > 0:
                with torch.amp.autocast('cuda'):

                    idx_samples = torch.randint(0, self.config['maxlen'], (n_items_sampled,), device=pos.device)
                    neg_sub = neg[:, idx_samples]

                    #neg_eseq = model.item_emb(neg)

                    neg_eseq = model.item_emb(neg_sub.detach())

                    if dump_distance_graph:
                        
                        neg_dist,neg_affinity = pairseq_dist_affinity(pos_eseq,single_dist,return_dist=True)

                    else:

                        neg_affinity = pairseq_dist_affinity(pos_eseq,single_dist)

                    neg_logits_dist = torch.cdist(neg_logits,neg_logits)**2

                    neg_lap = neg_affinity*neg_logits_dist

                    neg_man_reg = neg_lap.sum()

            else:
                neg_man_reg=torch.tensor(0)


            if dump_distance_graph:

                dump_dir = 'D:\\recsys_datasets\\exp_{}_dists'.format(self.config['dump_exp_name'])

                if not os.path.exists(dump_dir): os.mkdir(dump_dir)

                with torch.no_grad():
                    torch.save(pos_affinity, f'{dump_dir}/p_aff_{self.epoch_n}_{batch_number}.pt')
                    torch.save(pos_dist, f'{dump_dir}/p_dist_{self.epoch_n}_{batch_number}.pt')
                    torch.save(neg_affinity, f'{dump_dir}/n_aff_{self.epoch_n}_{batch_number}.pt')
                    torch.save(neg_dist, f'{dump_dir}/n_dist_{self.epoch_n}_{batch_number}.pt')

            if pos_lambda_man_reg > 0:
                batch_loss += pos_lambda_man_reg*pos_man_reg
            if neg_lambda_man_reg > 0:
                batch_loss += neg_lambda_man_reg*neg_man_reg
            #print('emb pos norms = {:.4f} emb neg norms = {:.4f} loss = {:.4f} pos_reg = {:.4f} neg_reg = {:.4f}'.format(torch.mean(torch.linalg.norm(pos_eseq,dim=-1),dim=-1).mean(),torch.mean(torch.linalg.norm(neg_eseq,dim=-1),dim=-1).mean(),criterion(pos_logits[indices], pos_labels[indices])+criterion(neg_logits[indices], neg_labels[indices]),pos_lambda_man_reg*pos_man_reg,neg_lambda_man_reg*neg_man_reg))
            
            
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()




            for param in model.item_emb.parameters():
                param.data = project_embeddings(param.data, c=self.config['c'], eps=1e-5)

            loss += batch_loss.item()

        # with torch.no_grad():
        #     emb = model.item_emb.weight.data  # or any embedding you want
        #     estimate_curvature_SVD(emb, c=self.config['geometry_c'])
        self.epoch_n+=1

        model.eval()
        return loss


    def predict(self, seq, user):
        model = self.model
        maxlen = self.config['maxlen']
        device = self.device

        with torch.no_grad():
            log_seqs = torch.full([maxlen], model.pad_token, dtype=torch.int64, device=device)
            log_seqs[-len(seq):] = torch.as_tensor(seq[-maxlen:], device=device)
            log_feats = model.log2feats(log_seqs.unsqueeze(0))
            final_feat = log_feats[:, -1, :] # only use last QKV classifier
            item_embs = model.item_emb.weight
            logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
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
            final_feat = log_feats[:, -1, :] # only use last QKV classifier
            item_embs = model.item_emb.weight
            logits = final_feat.matmul(item_embs.T)
        return logits.cpu().numpy()
        

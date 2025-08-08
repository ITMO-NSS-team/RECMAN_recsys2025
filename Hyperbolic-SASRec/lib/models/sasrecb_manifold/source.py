import torch.nn as nn
from lib.models.sasrec.source import SASRec


class SASRecCE(SASRec):
    def __init__(self, config, item_num):
        super().__init__(config, item_num)
        self.head = nn.Linear(in_features=config['hidden_units'], out_features=self.item_num + 1)
        self.head.weight = self.item_emb.weight

    def forward(self, log_seqs):
        log_feats = self.log2feats(log_seqs)
        logits = self.head(log_feats)
        return logits

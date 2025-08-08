import torch
import torch.nn as nn

from geoopt import PoincareBallExact

from lib.models.sasrec.source import SASRec
from lib.models.hypsasrec.source import unidirectional_poincare_mlr


class HypSASRecBCE(SASRec):
    def __init__(self, config, item_num):
        super().__init__(config, item_num)
        
        self.geom = config['geom']
        self.c = torch.tensor(config['c'])
        if self.geom == "ball":
            self.manifold = PoincareBallExact(c=config['c'])
            self.scaler = nn.Parameter(
                torch.zeros(self.item_emb.num_embeddings),
                requires_grad=False
            )
        else:
            raise NotImplementedError
    
    def forward(self, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)   # batch x seq.length x emb.dim
        hyp_feats = self.manifold.expmap0(log_feats)
        pos_logits = self.poincare_hyperplane(hyp_feats, pos_seqs)
        neg_logits = self.poincare_hyperplane(hyp_feats, neg_seqs)
        return pos_logits, neg_logits

    def poincare_hyperplane(self, hyp_feats, index=None):
        if index is not None:
            weight = self.item_emb(index)
            scaler = self.scaler[index]
            hyperplane = unidirectional_poincare_lr
        else:
            weight = self.item_emb.weight
            scaler = self.scaler
            hyperplane = unidirectional_poincare_mlr
        w_norm = weight.norm(dim=-1)
        w_unit = weight / w_norm.unsqueeze(-1)
        return hyperplane(hyp_feats, w_norm, w_unit, scaler, self.c)


@torch.jit.script
def unidirectional_poincare_lr(x, z_norm, z_unit, r, c):
    # parameters
    rc = c.sqrt()
    drcr = 2. * rc * r

    # input
    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1)

    denom_safe = torch.clamp_min(1. - cx2, 1e-15)

    return 2 * z_norm / rc * torch.arcsinh(
        (2. * torch.sum(rcx*z_unit, dim=-1) * drcr.cosh() - (1. + cx2) * drcr.sinh()) 
        / denom_safe
    )
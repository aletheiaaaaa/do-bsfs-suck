import torch
from torch import nn

from do_bsfs_suck.config import FeaturizerConfig
from do_bsfs_suck.featurizers.base import Featurizer, block_topk


class VanillaBSF(Featurizer):
    """z = Pi_k(xW + b). Free encoder and decoder, hard block top-k.

    At b=1 this is a magnitude-TopK SAE; see tests.
    """

    def __init__(self, cfg: FeaturizerConfig) -> None:
        super().__init__(cfg)
        self.W_enc = nn.Parameter(self.W_dec.detach().clone())
        self.b_enc = nn.Parameter(torch.zeros(cfg.n_blocks, cfg.block_dim))

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nd,gbd->ngb", x, self.W_enc) + self.b_enc

    def sparsify(self, pre: torch.Tensor) -> torch.Tensor:
        mask = block_topk(self.block_norms(pre), self.k)
        return pre * mask.unsqueeze(-1)

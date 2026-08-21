import torch
from torch import nn
from torch.nn import functional as F

from do_bsfs_suck.config import FeaturizerConfig
from do_bsfs_suck.featurizers.base import Featurizer, block_topk


class TopKSAE(Featurizer):
    """Canonical ReLU-TopK SAE. Not a BSF: b is forced to 1."""

    def __init__(self, cfg: FeaturizerConfig) -> None:
        if cfg.block_dim != 1:
            raise ValueError(f"topk_sae is b=1 only, got block_dim={cfg.block_dim}")
        super().__init__(cfg)
        self.W_enc = nn.Parameter(self.W_dec.detach().clone())
        self.b_enc = nn.Parameter(torch.zeros(cfg.n_blocks, cfg.block_dim))

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(torch.einsum("nd,gbd->ngb", x, self.W_enc) + self.b_enc)

    def sparsify(self, pre: torch.Tensor) -> torch.Tensor:
        mask = block_topk(self.block_norms(pre), self.k)
        return pre * mask.unsqueeze(-1)

import math

import torch
from torch import nn
from torch.nn import functional as F

from do_bsfs_suck.config import FeaturizerConfig
from do_bsfs_suck.featurizers.base import Featurizer, block_soft_threshold

THETA_INIT = 1e-2


def inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class GroupLassoBSF(Featurizer):
    """z = sh_theta(xW + b), loss ||x - zD||^2 + lambda ||z||_2,1.

    No k: sparsity is an outcome of lambda and the learned thresholds, so the
    grid's active-dims axis is measured after the fact and matched over lambda.
    At b=1 this is a soft-threshold (JumpReLU-like) SAE.
    """

    def __init__(self, cfg: FeaturizerConfig) -> None:
        super().__init__(cfg)
        self.W_enc = nn.Parameter(self.W_dec.detach().clone())
        self.b_enc = nn.Parameter(torch.zeros(cfg.n_blocks, cfg.block_dim))
        self.raw_theta = nn.Parameter(
            torch.full((cfg.n_blocks,), inv_softplus(THETA_INIT))
        )

    @property
    def theta(self) -> torch.Tensor:
        return F.softplus(self.raw_theta)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nd,gbd->ngb", x, self.W_enc) + self.b_enc

    def sparsify(self, pre: torch.Tensor) -> torch.Tensor:
        return block_soft_threshold(pre, self.theta)

    def loss(
        self, x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        recon = (x_hat.float() - x.float()).pow(2).sum(-1).mean()
        lasso = self.block_norms(z).sum(-1).mean()
        return {
            "recon": recon,
            "lasso": lasso,
            "loss": recon + self.cfg.lasso_coeff * lasso,
        }

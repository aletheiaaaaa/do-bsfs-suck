import torch
from torch import nn

from do_bsfs_suck.config import FeaturizerConfig
from do_bsfs_suck.featurizers.base import Featurizer, block_topk


def qr_retract(w: torch.Tensor) -> torch.Tensor:
    """(G, b, d) -> orthonormal rows, via batched QR."""
    q, r = torch.linalg.qr(w.transpose(-1, -2))
    # pin QR's sign ambiguity so frames don't flip between steps
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return (q * sign.unsqueeze(-2)).transpose(-1, -2)


class GrassmannBSF(Featurizer):
    """z = Pi_k(gamma x D^T), D_g in St(b, d), encoder tied to the decoder."""

    def __init__(self, cfg: FeaturizerConfig) -> None:
        super().__init__(cfg)
        with torch.no_grad():
            self.W_dec.copy_(qr_retract(self.W_dec))
        self.log_gamma = nn.Parameter(torch.zeros(()))

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nd,gbd->ngb", x, self.W_dec) * self.log_gamma.exp()

    def sparsify(self, pre: torch.Tensor) -> torch.Tensor:
        mask = block_topk(self.block_norms(pre), self.k)
        return pre * mask.unsqueeze(-1)

    def frames(self) -> torch.Tensor:
        return self.W_dec

    @torch.no_grad()
    def constrain(self) -> None:
        self.W_dec.copy_(qr_retract(self.W_dec))

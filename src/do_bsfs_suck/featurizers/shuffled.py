import torch
from torch import nn

from do_bsfs_suck.config import FeaturizerConfig
from do_bsfs_suck.featurizers.base import Featurizer, block_topk


class ShuffledBSF(Featurizer):
    """Control: blocks grouping random directions of a trained b=1 dictionary.

    A b-dim block fires whenever any of its b directions fire, so firing rate
    (and anything derived from it, absorption included) drifts with b for
    reasons unrelated to geometry. This reproduces that drift with no manifold
    structure. Decoder frozen, encoder trains.
    """

    def __init__(self, cfg: FeaturizerConfig, directions: torch.Tensor) -> None:
        super().__init__(cfg)
        if directions.shape != (cfg.dict_dims, cfg.d_in):
            raise ValueError(
                f"expected directions {(cfg.dict_dims, cfg.d_in)}, "
                f"got {tuple(directions.shape)}"
            )

        gen = torch.Generator().manual_seed(cfg.seed + 1)
        perm = torch.randperm(cfg.dict_dims, generator=gen)
        grouped = directions[perm].view(cfg.n_blocks, cfg.block_dim, cfg.d_in)

        with torch.no_grad():
            self.W_dec.copy_(grouped)
        self.W_dec.requires_grad_(False)

        self.W_enc = nn.Parameter(self.W_dec.detach().clone())
        self.b_enc = nn.Parameter(torch.zeros(cfg.n_blocks, cfg.block_dim))

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nd,gbd->ngb", x, self.W_enc) + self.b_enc

    def sparsify(self, pre: torch.Tensor) -> torch.Tensor:
        mask = block_topk(self.block_norms(pre), self.k)
        return pre * mask.unsqueeze(-1)

    @torch.no_grad()
    def constrain(self) -> None:
        pass

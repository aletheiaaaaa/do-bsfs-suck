from abc import ABC, abstractmethod

import torch
from torch import nn

from do_bsfs_suck.config import FeaturizerConfig


def block_topk(norms: torch.Tensor, k: int) -> torch.Tensor:
    """Pi_k: 0/1 mask keeping the k blocks of largest norm."""
    idx = norms.topk(k, dim=-1, sorted=False).indices
    return torch.zeros_like(norms).scatter_(-1, idx, 1.0)


def block_soft_threshold(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """sh_theta(z)_g = max(1 - theta_g/||z_g||, 0) z_g, the prox of ||.||_2,1."""
    norms = z.float().norm(dim=-1, keepdim=True)
    scale = (1.0 - theta.view(1, -1, 1) / norms.clamp_min(1e-8)).clamp_min(0.0)
    return scale * z


class Featurizer(nn.Module, ABC):
    """x_hat = zD, codes z of shape (N, G, b), decoder D of shape (G, b, d).

    Evals only ever touch block_norms/frames/project, which at b=1 collapse to
    |z|, the unit decoder direction, and |cos(d_g, v)|*||v|| -- so SAE and BSF
    are scored by one code path.
    """

    def __init__(self, cfg: FeaturizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        self.n_blocks = cfg.n_blocks
        self.block_dim = cfg.block_dim
        self.k = cfg.k

        gen = torch.Generator().manual_seed(cfg.seed)
        dec = torch.randn(cfg.n_blocks, cfg.block_dim, cfg.d_in, generator=gen)
        dec /= dec.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
        self.W_dec = nn.Parameter(dec)

        # drives AuxK revival; see aux_loss
        self.register_buffer("tokens_since_fired", torch.zeros(cfg.n_blocks))

    @abstractmethod
    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Dense codes, before sparsification."""

    @abstractmethod
    def sparsify(self, pre: torch.Tensor) -> torch.Tensor: ...

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.sparsify(self.encode_pre(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ngb,gbd->nd", z, self.W_dec)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    @staticmethod
    def block_norms(z: torch.Tensor) -> torch.Tensor:
        """Always fp32. Under bf16 autocast a b=16 norm carries 8 mantissa bits,
        and Pi_k ranks blocks by exactly this value -- ties there change which
        blocks fire, not just by how much."""
        return z.float().norm(dim=-1)

    def frames(self) -> torch.Tensor:
        """(G, b, d), orthonormal rows spanning each block."""
        q, _ = torch.linalg.qr(self.W_dec.transpose(-1, -2))
        return q.transpose(-1, -2)

    def project(self, v: torch.Tensor) -> torch.Tensor:
        """||P_g v|| per block; replaces cosine similarity in the eval metrics."""
        coords = torch.einsum("...d,gbd->...gb", v, self.frames())
        return coords.norm(dim=-1)

    def loss(
        self, x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # fp32: a 768-term sum of bf16 squares loses most of its low end
        recon = (x_hat.float() - x.float()).pow(2).sum(-1).mean()
        return {"recon": recon, "loss": recon}

    def aux_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        pre: torch.Tensor,
        dead_after: float,
        aux_k: int,
    ) -> torch.Tensor:
        """AuxK: let dead blocks reconstruct the residual, reviving them.

        Without it dead fractions reach ~70% at high b, where k = A/b is small.
        """
        dead = self.tokens_since_fired > dead_after
        n_dead = int(dead.sum())
        if n_dead == 0:
            return x.new_zeros(())

        norms = self.block_norms(pre) * dead
        idx = norms.topk(min(aux_k, n_dead), dim=-1).indices
        keep = torch.zeros_like(norms).scatter_(-1, idx, 1.0)
        aux = self.decode(pre * keep.unsqueeze(-1)).float()
        return (aux - (x.float() - x_hat.float())).pow(2).sum(-1).mean()

    @torch.no_grad()
    def constrain(self) -> None:
        """Project blocks onto the unit ball. A clamp, not a renormalization:
        short blocks are left free to shrink."""
        norm = self.W_dec.flatten(1).norm(dim=1)
        self.W_dec.div_(norm.clamp_min(1.0).view(-1, 1, 1))

    @torch.no_grad()
    def track_dead(self, z: torch.Tensor, n_tokens: int) -> None:
        fired = self.block_norms(z).gt(0).any(dim=0)
        self.tokens_since_fired += n_tokens
        self.tokens_since_fired[fired] = 0.0

import torch

from do_bsfs_suck.featurizers import Featurizer


class ReconStats:
    """Streaming FVU and sparsity, accumulated over an evaluation pass."""

    def __init__(self) -> None:
        self.sq_err = 0.0
        self.sq_tot = 0.0
        self.n_blocks = 0.0
        self.n_dims = 0.0
        self.tokens = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor) -> None:
        self.sq_err += (x_hat - x).pow(2).sum().item()
        self.sq_tot += (x - x.mean(0, keepdim=True)).pow(2).sum().item()
        self.n_blocks += z.norm(dim=-1).gt(0).sum().item()
        self.n_dims += z.ne(0).sum().item()
        self.tokens += x.shape[0]

    def summary(self) -> dict[str, float]:
        n = max(self.tokens, 1)
        return {
            "fvu": self.sq_err / max(self.sq_tot, 1e-12),
            "l0_blocks": self.n_blocks / n,
            "l0_dims": self.n_dims / n,
            "tokens": self.tokens,
        }


@torch.no_grad()
def evaluate(
    model: Featurizer, batches, accumulators: list | None = None
) -> dict[str, float]:
    stats = ReconStats()
    model.eval()
    for x in batches:
        x_hat, z = model(x)
        stats.update(x, x_hat, z)
        for acc in accumulators or []:
            acc.update(z)
    return stats.summary()

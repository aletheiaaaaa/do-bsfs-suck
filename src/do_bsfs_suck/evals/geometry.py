import torch

from do_bsfs_suck.featurizers import Featurizer


class BlockGram:
    """Accumulates sum_n z_ng z_ng^T per block, in (G, b, b).

    Everything the stable rank needs comes out of this: singular values of the
    stacked code matrix M_g are the square roots of eigenvalues of the Gram, so
    srank = tr(C)/lambda_max(C) without ever materializing M_g.
    """

    def __init__(self, n_blocks: int, block_dim: int, device: str = "cpu") -> None:
        # float64: this sums over ~50M tokens
        self.gram = torch.zeros(n_blocks, block_dim, block_dim, device=device, dtype=torch.float64)
        self.fired = torch.zeros(n_blocks, device=device, dtype=torch.float64)
        self.tokens = 0

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
        self.gram += torch.einsum("ngb,ngc->gbc", z, z).double()
        self.fired += z.norm(dim=-1).gt(0).sum(0)
        self.tokens += z.shape[0]


def _srank(gram: torch.Tensor) -> torch.Tensor:
    eig = torch.linalg.eigvalsh(gram).clamp_min(0)
    top = eig[:, -1]
    out = eig.sum(-1) / top.clamp_min(1e-12)
    return torch.where(top > 1e-12, out, torch.zeros_like(out)).float()


@torch.no_grad()
def stable_ranks(acc: BlockGram, model: Featurizer) -> dict[str, torch.Tensor]:
    """Stable ranks under both readings of M_g: codes and contributions."""
    codes = _srank(acc.gram)

    d = model.W_dec.detach()
    contrib_gram = torch.einsum("gbc,gcd,ged->gbe", acc.gram, d.double(), d.double())
    # symmetrize: the product above is only symmetric up to float error
    contrib = _srank(0.5 * (contrib_gram + contrib_gram.transpose(-1, -2)))

    alive = acc.fired > 0
    return {
        "srank_codes": codes,
        "srank_contributions": contrib,
        "alive": alive,
        "fire_rate": acc.fired / max(acc.tokens, 1),
    }


def summarize_ranks(ranks: dict[str, torch.Tensor]) -> dict[str, float]:
    alive = ranks["alive"]
    if not bool(alive.any()):
        return {"srank_codes": 0.0, "srank_contributions": 0.0, "alive_frac": 0.0}
    return {
        "srank_codes": ranks["srank_codes"][alive].mean().item(),
        "srank_contributions": ranks["srank_contributions"][alive].mean().item(),
        "alive_frac": alive.float().mean().item(),
        "dead_frac": 1.0 - alive.float().mean().item(),
    }


@torch.no_grad()
def principal_angles(model: Featurizer, sample: int = 512, seed: int = 0) -> torch.Tensor:
    """Largest cosine between subspaces, over a random sample of block pairs."""
    frames = model.frames()
    g = frames.shape[0]
    gen = torch.Generator(device=frames.device).manual_seed(seed)
    i = torch.randint(g, (sample,), generator=gen, device=frames.device)
    j = torch.randint(g, (sample,), generator=gen, device=frames.device)
    keep = i != j
    i, j = i[keep], j[keep]

    # singular values of Q_i Q_j^T are the cosines of the principal angles
    m = torch.einsum("sbd,scd->sbc", frames[i], frames[j])
    return torch.linalg.svdvals(m)[:, 0]

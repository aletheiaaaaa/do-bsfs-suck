import math

import torch

from do_bsfs_suck.featurizers import Featurizer, GrassmannBSF

BITS_PER_PARAM = 32
HIST_RANGE = 64.0
HIST_BINS = 8192


def dictionary_params(model: Featurizer) -> int:
    """Free parameters in D. A Stiefel frame has b*d - b(b+1)/2, not b*d."""
    g, b, d = model.W_dec.shape
    if isinstance(model, GrassmannBSF):
        return g * (b * d - b * (b + 1) // 2)
    return g * b * d


class MDLStats:
    """Accumulates what Eq. 5 needs, over an evaluation pass."""

    def __init__(self, n_blocks: int) -> None:
        self.n_blocks = n_blocks
        self.hist = torch.zeros(HIST_BINS, dtype=torch.float64)
        self.support_bits = 0.0
        self.active_scalars = 0.0
        self.resid_sq = 0.0
        self.tokens = 0
        self.d_in = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor) -> None:
        self.d_in = x.shape[-1]
        active = z.norm(dim=-1).gt(0)

        # log2 C(G, k) per token, since k varies for the group lasso
        k = active.sum(-1)
        for kk, count in zip(*k.unique(return_counts=True)):
            self.support_bits += count.item() * _log2_comb(self.n_blocks, int(kk))

        vals = z[active]
        self.active_scalars += vals.numel()
        idx = (
            ((vals.flatten() + HIST_RANGE) / (2 * HIST_RANGE) * HIST_BINS)
            .long()
            .clamp(0, HIST_BINS - 1)
        )
        self.hist += torch.bincount(idx, minlength=HIST_BINS).double()

        self.resid_sq += (x - x_hat).pow(2).sum().item()
        self.tokens += x.shape[0]

    def code_entropy(self, delta: float) -> float:
        """Bits per active scalar, from the empirical distribution at step delta."""
        step = 2 * HIST_RANGE / HIST_BINS
        group = max(int(round(delta / step)), 1)
        binned = self.hist[: (HIST_BINS // group) * group].view(-1, group).sum(-1)
        p = binned / binned.sum().clamp_min(1)
        p = p[p > 0]
        return float(-(p * p.log2()).sum())

    def bits(self, model: Featurizer, delta: float, n_samples: int | None = None) -> dict[str, float]:
        n = max(self.tokens, 1)
        n_samples = n_samples or n

        support = self.support_bits / n
        code = (self.active_scalars / n) * self.code_entropy(delta)

        # residual coded at per-dim distortion delta^2, Gaussian rate, clipped
        sigma_sq = self.resid_sq / (n * max(self.d_in, 1))
        resid = 0.5 * self.d_in * math.log2(max(sigma_sq / delta**2, 1.0))

        dict_bits = dictionary_params(model) * BITS_PER_PARAM / n_samples

        return {
            "delta": delta,
            "bits_support": support,
            "bits_code": code,
            "bits_residual": resid,
            "bits_dict": dict_bits,
            "bits_total": support + code + resid + dict_bits,
        }


def _log2_comb(n: int, k: int) -> float:
    if k <= 0 or k > n:
        return 0.0
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def mdl_curve(
    stats: MDLStats, model: Featurizer, deltas: tuple[float, ...], n_samples: int | None = None
) -> list[dict[str, float]]:
    """MDL across distortion levels; the paper's delta is task dependent, so the
    curve is reported rather than a single point."""
    return [stats.bits(model, d, n_samples) for d in deltas]

import contextlib

import torch
from torch import nn

from do_bsfs_suck.config import FeaturizerConfig
from do_bsfs_suck.featurizers.base import Featurizer, block_topk

# min/max of a Cholesky factor's diagonal, below which one pass loses accuracy;
# retracted frames sit near 1, and rows going parallel is what drives it down
WELL_CONDITIONED = 0.1


def _qr_pass(w: torch.Tensor) -> torch.Tensor:
    """Householder QR: the reference, and the fallback when a Gram is singular."""
    q, r = torch.linalg.qr(w.transpose(-1, -2))
    # pin QR's sign ambiguity so frames don't flip between steps
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return (q * sign.unsqueeze(-2)).transpose(-1, -2)


@contextlib.contextmanager
def _exact_matmul():
    """Form the Gram in true fp32.

    Squaring the condition number only pays off if the Gram is exact: bf16
    autocast caps it near 4e-3 and tf32 near 5e-4, and no extra pass recovers
    either. QR needs none of this -- torch promotes it to fp32 on its own.
    """
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        # autocast is keyed by device type, and `@` is on its bf16 list
        with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _cholesky_pass(w: torch.Tensor) -> tuple[torch.Tensor | None, float]:
    """One Cholesky-QR pass, with the worst block's diagonal ratio."""
    with _exact_matmul():
        gram = w @ w.transpose(-1, -2)
        chol, info = torch.linalg.cholesky_ex(gram)
        if bool(info.any()):
            return None, 0.0
        # gram = L L^T, so L^-1 w has orthonormal rows, and L's positive diagonal
        # pins the same canonical sign the QR path fixes by hand
        out = torch.linalg.solve_triangular(chol, w, upper=False, left=True)
    diag = torch.diagonal(chol, dim1=-2, dim2=-1)
    return out, float((diag.amin(-1) / diag.amax(-1).clamp_min(1e-30)).amin())


def qr_retract(w: torch.Tensor) -> torch.Tensor:
    """(G, b, d) -> orthonormal rows, via Cholesky-QR on the (b, b) Grams."""
    if w.shape[-2] == 1:
        # one row is its own frame; a factorization would return exactly this
        return w / w.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    out, conditioning = _cholesky_pass(w)
    if out is None:
        return _qr_pass(w)
    if conditioning >= WELL_CONDITIONED:
        return out
    # forming the Gram squared the condition number; a second pass recovers it
    refined, _ = _cholesky_pass(out)
    return _qr_pass(w) if refined is None else refined


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

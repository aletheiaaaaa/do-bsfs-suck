import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from do_bsfs_suck.evals.probes import LETTERS
from do_bsfs_suck.featurizers import Featurizer

COS_THRESHOLD = 0.025
# an additive margin in probe-logit units, not a ratio -- and our activations
# are rescaled, so the paper's absolute value does not carry over
ABLATION_MARGIN = 1.0
K_MAIN = 3


def main_budget(model: Featurizer, k_main: int = K_MAIN) -> int:
    """Chanin's k_main=3, capped at the featurizer's own k.

    At matched A = k*b a high-b run has k = A/b as low as 2, and asking for 3
    main blocks when only 2 can ever fire at once guarantees a silent parent --
    absorption_rate would then be measuring the sparsity budget, not absorption.
    """
    return max(1, min(k_main, model.k))


CHUNK = 128
PREFILTER = 256


@torch.no_grad()
def block_activations(model: Featurizer, acts: torch.Tensor) -> torch.Tensor:
    """Block activations, chunked to bound memory."""
    acts = acts.to(model.W_dec.device)
    return torch.cat(
        [model.block_norms(model.encode(c)) for c in acts.split(CHUNK)]
    )


def select_main_blocks(
    block_acts: torch.Tensor, y: np.ndarray, k: int, prefilter: int = PREFILTER
) -> np.ndarray:
    """k-sparse probing: the k blocks that positively indicate the concept."""
    x = block_acts.numpy()
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return np.zeros(0, dtype=int)

    if x.shape[1] > prefilter:
        gap = x[pos].mean(0) - x[neg].mean(0)
        cand = np.argsort(-gap)[:prefilter]
    else:
        cand = np.arange(x.shape[1])

    clf = LogisticRegression(
        l1_ratio=1.0, C=0.01, solver="liblinear", max_iter=2000, class_weight="balanced"
    ).fit(x[:, cand], y)
    return cand[np.argsort(-clf.coef_[0])[:k]]


@torch.no_grad()
def _contributions(model: Featurizer, acts: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Each block's signed contribution to the probe logit."""
    dev = model.W_dec.device
    wd = torch.einsum("gbd,d->gb", model.W_dec, w.to(dev))
    return torch.cat(
        [
            torch.einsum("ngb,gb->ng", model.encode(c), wd)
            for c in acts.to(dev).split(CHUNK)
        ]
    )


@torch.no_grad()
def absorption_for_letter(
    model: Featurizer,
    acts: torch.Tensor,
    y: np.ndarray,
    predicted: np.ndarray,
    w: torch.Tensor,
    k_main: int | None = None,
    cos_threshold: float = COS_THRESHOLD,
    ablation_margin: float = ABLATION_MARGIN,
) -> dict[str, float]:
    dev = model.W_dec.device
    acts, w = acts.to(dev), w.to(dev)
    k_main = main_budget(model) if k_main is None else k_main
    # numpy indexing and sklearn both need this side of the device boundary
    block_acts = block_activations(model, acts).cpu()
    main = select_main_blocks(block_acts, y, k_main)

    true_pos = np.flatnonzero((y == 1) & (predicted == 1))
    if len(true_pos) == 0:
        return {}

    main_fires = block_acts[:, main].gt(0).any(-1).numpy()
    silent = true_pos[~main_fires[true_pos]]

    # how much of the probe direction the main blocks already span
    cos = (model.project(w) / w.norm().clamp_min(1e-8)).cpu()
    containment = float(cos[main].max())

    absorptions = 0
    if len(silent):
        # ablating a block changes the probe logit by -w.(z_g D_g)
        contrib = _contributions(
            model, acts[torch.as_tensor(silent, device=dev)], w
        ).cpu()
        top2 = contrib.topk(2, dim=-1)
        best, runner_up = top2.values[:, 0], top2.values[:, 1]
        # the runner-up is over all blocks, not the cos-eligible ones
        absorptions = int(
            (
                (best > 0)
                & (cos[top2.indices[:, 0]] > cos_threshold)
                & (best - runner_up >= ablation_margin)
            ).sum()
        )

    return {
        "absorption_rate": absorptions / len(true_pos),
        "main_fire_rate": float(main_fires[true_pos].mean()),
        "containment": containment,
        "true_positives": len(true_pos),
        "n_silent": len(silent),
        "k_main": k_main,
    }


def absorption(
    model: Featurizer,
    acts: torch.Tensor,
    letters: list[str],
    probes: dict[str, tuple[torch.Tensor, np.ndarray]],
    **kw,
) -> dict[str, dict[str, float]]:
    """`probes` maps a letter to its probe direction and its test predictions."""
    out = {}
    for letter in LETTERS:
        if letter not in probes:
            continue
        w, predicted = probes[letter]
        y = np.array([c == letter for c in letters], dtype=int)
        got = absorption_for_letter(model, acts, y, predicted, w, **kw)
        if got:
            out[letter] = got
    return out


def summarize_absorption(per_letter: dict[str, dict[str, float]]) -> dict[str, float]:
    """Read absorption_rate next to main_fire_rate and containment."""
    if not per_letter:
        return {}
    keys = ("absorption_rate", "main_fire_rate", "containment")
    return {k: float(np.mean([v[k] for v in per_letter.values()])) for k in keys} | {
        "n_letters": len(per_letter),
        "k_main": next(iter(per_letter.values()))["k_main"],
    }

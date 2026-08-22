import numpy as np
import torch

from do_bsfs_suck.featurizers import Featurizer

TAUS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)


def _first_letter(s: str) -> str | None:
    t = s.strip()
    return t[0].lower() if t and t[0].isalpha() else None


ATTRIBUTES = {
    "first_letter": _first_letter,
    "is_digit": lambda s: s.strip().isdigit() if s.strip() else None,
    "is_capitalized": lambda s: s.strip()[0].isupper() if s.strip() else None,
    "has_leading_space": lambda s: s.startswith(" "),
    "is_punctuation": lambda s: (
        not any(c.isalnum() for c in s.strip()) if s.strip() else None
    ),
}


def vocab_sample(tokenizer, n: int = 8000, seed: int = 0) -> tuple[list[int], list[str]]:
    """Unfiltered vocabulary sample, so non-alphabetic attributes vary."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(tokenizer.vocab_size, min(n, tokenizer.vocab_size), replace=False)
    return list(map(int, ids)), [tokenizer.decode([i]) for i in ids]


def supervised_directions(
    acts: torch.Tensor, strings: list[str], min_count: int = 30
) -> dict[str, torch.Tensor]:
    """u_v = E[a | attr = v] - E[a], the Makelov-style mean-difference dictionary."""
    mean = acts.mean(0)
    out: dict[str, torch.Tensor] = {}

    for attr, fn in ATTRIBUTES.items():
        values = [fn(s) for s in strings]
        for v in sorted({v for v in values if v is not None}, key=str):
            mask = torch.tensor([x == v for x in values])
            if int(mask.sum()) < min_count:
                continue
            out[f"{attr}={v}"] = acts[mask].mean(0) - mean
    return out


@torch.no_grad()
def split_counts(
    model: Featurizer, directions: dict[str, torch.Tensor], taus: tuple[float, ...] = TAUS
) -> dict[str, dict[float, int]]:
    """How many blocks align with each supervised feature, as a curve over tau."""
    out: dict[str, dict[float, int]] = {}
    for name, u in directions.items():
        u = u.to(model.W_dec.device)
        cos = model.project(u) / u.norm().clamp_min(1e-8)
        out[name] = {t: int((cos > t).sum()) for t in taus}
    return out


def summarize_splits(counts: dict[str, dict[float, int]]) -> dict[str, float]:
    if not counts:
        return {}
    taus = next(iter(counts.values())).keys()
    out = {f"splits@{t}": float(np.mean([c[t] for c in counts.values()])) for t in taus}
    out["n_features"] = len(counts)
    return out

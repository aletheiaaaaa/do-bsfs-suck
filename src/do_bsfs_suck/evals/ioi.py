from dataclasses import dataclass

import numpy as np
import torch

from do_bsfs_suck.evals.splitting import TAUS
from do_bsfs_suck.featurizers import Featurizer

TEMPLATE = "When {a} and {b} went to the store, {s} gave a book to"

# single-token names, filtered against the tokenizer and against substrings
NAMES = (
    "John Mary Tom James Dan Paul Alice Bob Carl Emma Jack Kate Luke Nick "
    "Rose Sam Will Anna Chris David Eric Frank Grace Henry Julia Kevin Laura "
    "Mark Nancy Oliver Peter Rachel Simon Tina Victor Wendy"
).split()


@dataclass
class IOIData:
    prompts: list[str]
    io: list[str]
    s: list[str]
    pos: list[str]
    io_ids: list[int]

    def __len__(self) -> int:
        return len(self.prompts)


def usable_names(tokenizer) -> list[str]:
    """Single-token, space-prefixed names that are not substrings of each other."""
    single = [
        n for n in NAMES
        if len(tokenizer(" " + n)["input_ids"]) == 1
    ]
    return [
        n for n in single
        if not any(other != n and n in other for other in single)
    ]


def make_ioi(
    tokenizer, n: int = 512, seed: int = 0, n_names: int = 12
) -> IOIData:
    """IOI prompts parameterized by S (repeated), IO (not repeated), and Pos."""
    names = usable_names(tokenizer)[:n_names]
    if len(names) < 2:
        raise ValueError("need at least two single-token names")

    rng = np.random.default_rng(seed)
    prompts, ios, subjects, poss, ids = [], [], [], [], []

    for _ in range(n):
        io, s = rng.choice(names, size=2, replace=False)
        pos = "ABB" if rng.random() < 0.5 else "BAB"
        a, b = (io, s) if pos == "ABB" else (s, io)
        prompts.append(TEMPLATE.format(a=a, b=b, s=s))
        ios.append(str(io))
        subjects.append(str(s))
        poss.append(pos)
        ids.append(tokenizer(" " + io)["input_ids"][0])

    return IOIData(prompts, ios, subjects, poss, ids)


@torch.no_grad()
def ioi_activations(
    model,
    tokenizer,
    data: IOIData,
    layers: tuple[int, ...],
    device: str = "cpu",
    batch: int = 32,
    scale: dict[int, float] | None = None,
    mean: dict[int, torch.Tensor] | None = None,
) -> dict[int, torch.Tensor]:
    """Residual activation at the final position, where IO is predicted."""
    from do_bsfs_suck.stream import _capture

    out: dict[int, list[torch.Tensor]] = {i: [] for i in layers}
    for start in range(0, len(data), batch):
        chunk = data.prompts[start : start + batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=False)
        ids = enc["input_ids"].to(device)
        with _capture(model, layers) as caught:
            model(ids)
        for i, a in caught.items():
            out[i].append(a[:, -1].float().cpu())

    acts = {i: torch.cat(v) for i, v in out.items()}
    if mean:
        acts = {i: a - mean[i].cpu() for i, a in acts.items()}
    if scale:
        acts = {i: a * scale[i] for i, a in acts.items()}
    return acts


def ioi_directions(
    acts: torch.Tensor, data: IOIData, min_count: int = 30
) -> dict[str, torch.Tensor]:
    """Mean-difference supervised dictionary over the three IOI attributes."""
    mean = acts.mean(0)
    out: dict[str, torch.Tensor] = {}

    for attr, values in (("S", data.s), ("IO", data.io), ("Pos", data.pos)):
        for v in sorted(set(values)):
            mask = torch.tensor([x == v for x in values])
            if int(mask.sum()) < min_count:
                continue
            out[f"{attr}={v}"] = acts[mask].mean(0) - mean
    return out


@torch.no_grad()
def pos_split_counts(
    model: Featurizer, directions: dict[str, torch.Tensor], taus: tuple[float, ...] = TAUS
) -> dict[str, dict[float, int]]:
    """Blocks aligned with each supervised IOI feature, as a curve over tau."""
    out: dict[str, dict[float, int]] = {}
    for name, u in directions.items():
        u = u.to(model.W_dec.device)
        cos = model.project(u) / u.norm().clamp_min(1e-8)
        out[name] = {t: int((cos > t).sum()) for t in taus}
    return out


def summarize_ioi(counts: dict[str, dict[float, int]]) -> dict[str, float]:
    """Per attribute: mean blocks per supervised feature."""
    out: dict[str, float] = {}
    for attr in ("S", "IO", "Pos"):
        rows = [c for k, c in counts.items() if k.startswith(f"{attr}=")]
        if not rows:
            continue
        for tau in rows[0]:
            out[f"ioi_{attr}_splits@{tau}"] = float(np.mean([r[tau] for r in rows]))
        out[f"ioi_{attr}_features"] = len(rows)
    return out

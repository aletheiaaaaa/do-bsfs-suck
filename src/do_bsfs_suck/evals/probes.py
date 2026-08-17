import re
import string
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from do_bsfs_suck.stream import _capture

TEMPLATE = "{token} has the first letter:"
WORDLIKE = re.compile(r"^ ?[A-Za-z]{3,}$")
LETTERS = string.ascii_lowercase


@dataclass
class SpellingSet:
    token_ids: list[int]
    strings: list[str]
    letters: list[str]

    def __len__(self) -> int:
        return len(self.token_ids)


def spelling_tokens(tokenizer, limit: int | None = 4000, seed: int = 0) -> SpellingSet:
    """Single-token wordlike vocabulary entries, labelled by first letter."""
    ids, strs, letters = [], [], []
    for tok_id in range(tokenizer.vocab_size):
        s = tokenizer.decode([tok_id])
        if not WORDLIKE.match(s):
            continue
        ids.append(tok_id)
        strs.append(s)
        letters.append(s.strip()[0].lower())

    if limit is not None and len(ids) > limit:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(ids), limit, replace=False)
        ids = [ids[i] for i in keep]
        strs = [strs[i] for i in keep]
        letters = [letters[i] for i in keep]
    return SpellingSet(ids, strs, letters)


SUBJECT_POS = 1  # after BOS: position 0 is the attention sink, not token identity


@torch.no_grad()
def token_activations(
    model: nn.Module,
    tokenizer,
    tokens: SpellingSet,
    layers: tuple[int, ...],
    device: str = "cpu",
    batch: int = 64,
    scale: dict[int, float] | None = None,
    mean: dict[int, torch.Tensor] | None = None,
) -> dict[int, torch.Tensor]:
    """Residual activation at the subject-token position of the spelling prompt.

    ids are assembled by hand so the subject is exactly the labelled vocabulary
    token, not whatever re-tokenizing its string would produce.
    """
    out: dict[int, list[torch.Tensor]] = {i: [] for i in layers}
    bos = tokenizer.bos_token_id or tokenizer.eos_token_id
    suffix = tokenizer(TEMPLATE.split("{token}")[1])["input_ids"]

    for start in range(0, len(tokens), batch):
        chunk = tokens.token_ids[start : start + batch]
        ids = torch.tensor([[bos, t, *suffix] for t in chunk], device=device)

        with _capture(model, layers) as caught:
            model(ids)

        for i, a in caught.items():
            out[i].append(a[:, SUBJECT_POS].float().cpu())

    acts = {i: torch.cat(v) for i, v in out.items()}
    if mean:
        acts = {i: a - mean[i].cpu() for i, a in acts.items()}
    if scale:
        acts = {i: a * scale[i] for i, a in acts.items()}
    return acts


@dataclass
class LetterProbe:
    direction: torch.Tensor  # in raw activation space
    test_idx: np.ndarray
    predicted: np.ndarray  # on the test split
    y: np.ndarray  # on the test split
    metrics: dict[str, float]


def fit_letter_probes(
    acts: torch.Tensor,
    letters: list[str],
    seed: int = 0,
    min_positives: int = 20,
) -> dict[str, LetterProbe]:
    """One binary probe per letter, kept so absorption can reuse the direction."""
    raw = acts.numpy()
    scaler = StandardScaler().fit(raw)
    x = scaler.transform(raw)
    index = np.arange(len(letters))
    out: dict[str, LetterProbe] = {}

    for letter in LETTERS:
        y = np.array([c == letter for c in letters], dtype=int)
        if y.sum() < min_positives:
            continue
        itr, ite = train_test_split(
            index, test_size=0.2, random_state=seed, stratify=y
        )
        # each letter is ~4% of the set; unbalanced, the probe just predicts 0
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(
            x[itr], y[itr]
        )
        pred = clf.predict(x[ite])
        tp = int(((pred == 1) & (y[ite] == 1)).sum())
        fp = int(((pred == 1) & (y[ite] == 0)).sum())
        fn = int(((pred == 0) & (y[ite] == 1)).sum())

        # undo standardization so the direction lives in activation space
        w = torch.tensor(clf.coef_[0] / scaler.scale_, dtype=acts.dtype)
        out[letter] = LetterProbe(
            direction=w,
            test_idx=ite,
            predicted=pred,
            y=y[ite],
            metrics={
                "acc": float((pred == y[ite]).mean()),
                "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "base_rate": float(y[ite].mean()),
                "n_pos": int(y.sum()),
            },
        )
    return out


def letter_probes(
    acts: torch.Tensor, letters: list[str], seed: int = 0, min_positives: int = 20
) -> dict[str, dict[str, float]]:
    probes = fit_letter_probes(acts, letters, seed, min_positives)
    return {k: v.metrics for k, v in probes.items()}


def summarize(probes: dict[str, dict[str, float]]) -> dict[str, float]:
    if not probes:
        return {"mean_f1": 0.0, "mean_acc": 0.0, "n_letters": 0}
    return {
        "mean_f1": float(np.mean([p["f1"] for p in probes.values()])),
        "mean_acc": float(np.mean([p["acc"] for p in probes.values()])),
        "n_letters": len(probes),
    }

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoTokenizer

from do_bsfs_suck.arch import blocks as _model_blocks
from do_bsfs_suck.config import StreamConfig
from do_bsfs_suck.randomize import RandomizeSpec, control_hook, load_model

CALIBRATION_TOKENS = 100_000


@contextmanager
def _capture(model: nn.Module, layers: tuple[int, ...]):
    """Grab resid_post (block output) for each requested layer."""
    caught: dict[int, torch.Tensor] = {}
    handles = []

    def make(i: int):
        def hook(_m, _args, out):
            caught[i] = out[0] if isinstance(out, tuple) else out

        return hook

    for i in layers:
        handles.append(_model_blocks(model)[i].register_forward_hook(make(i)))
    try:
        yield caught
    finally:
        for h in handles:
            h.remove()


def token_batches(cfg: StreamConfig, tokenizer) -> Iterator[torch.Tensor]:
    """(batch_seqs, seq_len) int64, packed end to end from the corpus."""
    ds = load_dataset(cfg.dataset, split="train", streaming=True)
    ds = ds.shuffle(seed=cfg.data_seed, buffer_size=10_000)

    buf: list[int] = []
    need = cfg.seq_len * cfg.batch_seqs
    for row in ds:
        buf.extend(tokenizer(row["text"])["input_ids"])
        buf.append(tokenizer.eos_token_id)
        while len(buf) >= need:
            chunk, buf = buf[:need], buf[need:]
            yield torch.tensor(chunk, dtype=torch.long).view(cfg.batch_seqs, cfg.seq_len)


class ActivationStream:
    """One model pass feeding every featurizer, with hooks on all layers at once.

    Activations are scaled by a single global constant per layer so that
    E[||x||] = sqrt(d); relative norms across tokens survive.
    """

    def __init__(
        self,
        cfg: StreamConfig,
        spec: RandomizeSpec = RandomizeSpec(),
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        self.model = load_model(cfg.model, cfg.condition, spec, cfg.seed, dtype).to(device)
        self.d_model = self.model.config.hidden_size

        self._control = None
        if cfg.condition == "control":
            self._control = self.model.get_input_embeddings().register_forward_hook(
                control_hook(spec, cfg.seed)
            )

        self.scale: dict[int, float] = {}
        self.mean: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def _forward(self, ids: torch.Tensor) -> dict[int, torch.Tensor]:
        with _capture(self.model, self.cfg.layers) as caught:
            self.model(ids.to(self.device))
        d = self.cfg.drop_first
        return {i: a[:, d:].reshape(-1, self.d_model).float() for i, a in caught.items()}

    def _calibration_pass(self, fn) -> int:
        seen = 0
        for ids in token_batches(self.cfg, self.tokenizer):
            acts = self._forward(ids)
            for i, a in acts.items():
                fn(i, a)
            seen += next(iter(acts.values())).shape[0]
            if seen >= CALIBRATION_TOKENS:
                return seen
        return seen

    def calibrate(self) -> None:
        """Per layer: the mean to subtract, then the constant that sets E||x-mu||
        to sqrt(d).

        Random transformers put nearly all their activation mass in a constant
        offset -- mean norm ~48x the spread on pythia-14m -- and the featurizer
        has no decoder bias, so without centering it spends every block on that
        offset and reconstruction collapses.
        """
        sums = {i: torch.zeros(self.d_model, device=self.device) for i in self.cfg.layers}
        seen = self._calibration_pass(lambda i, a: sums[i].add_(a.sum(0)))
        self.mean = {i: s / seen for i, s in sums.items()}

        # the dataset order is seeded, so this pass sees the same tokens
        norms = {i: 0.0 for i in self.cfg.layers}

        def acc(i: int, a: torch.Tensor) -> None:
            norms[i] += (a - self.mean[i]).norm(dim=-1).sum().item()

        seen = self._calibration_pass(acc)
        root_d = math.sqrt(self.d_model)
        self.scale = {i: root_d / (n / seen) for i, n in norms.items()}

    def normalize(self, acts: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        return {i: (a - self.mean[i]) * self.scale[i] for i, a in acts.items()}

    def __iter__(self) -> Iterator[dict[int, torch.Tensor]]:
        if not self.scale:
            self.calibrate()
        seen = 0
        for ids in token_batches(self.cfg, self.tokenizer):
            out = self.normalize(self._forward(ids))
            seen += next(iter(out.values())).shape[0]
            yield out
            if seen >= self.cfg.n_tokens:
                return

    def close(self) -> None:
        if self._control is not None:
            self._control.remove()
            self._control = None

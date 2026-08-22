import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from tqdm import tqdm
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


def corpus_batches(cfg: StreamConfig, tokenizer) -> Iterator[torch.Tensor]:
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


def bank_paths(cfg: StreamConfig) -> dict[int, Path]:
    """One memmap per layer, keyed by everything that changes the activations."""
    slug = f"{cfg.model}__{cfg.dataset}__{cfg.condition}".replace("/", "--")
    stem = f"{slug}_s{cfg.seed}_d{cfg.data_seed}_n{cfg.n_tokens}_L{cfg.seq_len}"
    return {i: Path(cfg.bank_dir) / f"{stem}_l{i}.npy" for i in cfg.layers}


def bank_rows(cfg: StreamConfig) -> int:
    """Rows the bank holds; a whole number of yields, so a replay lines up."""
    per_yield = cfg.batch_seqs * (cfg.seq_len - cfg.drop_first)
    return -(-cfg.n_tokens // per_yield) * per_yield


def bank_bytes(cfg: StreamConfig, d_model: int) -> int:
    """fp16, so two bytes per dimension per token per layer."""
    return bank_rows(cfg) * d_model * len(cfg.layers) * 2


class ActivationStream:
    """One model pass feeding every featurizer, with hooks on all layers at once."""

    def __init__(
        self,
        cfg: StreamConfig,
        spec: RandomizeSpec = RandomizeSpec(),
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if cfg.bank_dir is None:
            raise ValueError(
                "bank_dir is required: activations are banked to disk once and "
                "replayed, rather than re-running the model for every pass"
            )
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
        for ids in corpus_batches(self.cfg, self.tokenizer):
            acts = self._forward(ids)
            for i, a in acts.items():
                fn(i, a)
            seen += next(iter(acts.values())).shape[0]
            if seen >= CALIBRATION_TOKENS:
                return seen
        return seen

    def calibrate(self) -> None:
        """Per layer: the mean to subtract, then the scale that sets E||x-mu|| to sqrt(d)."""
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

    def _model_batches(self) -> Iterator[dict[int, torch.Tensor]]:
        seen = 0
        for ids in corpus_batches(self.cfg, self.tokenizer):
            out = self.normalize(self._forward(ids))
            seen += next(iter(out.values())).shape[0]
            yield out
            if seen >= self.cfg.n_tokens:
                return

    def build_bank(self, paths: dict[int, Path]) -> None:
        """Write normalized activations once; every later pass reads them back."""
        rows = bank_rows(self.cfg)
        tmp = {i: p.with_suffix(".building.npy") for i, p in paths.items()}
        for p in paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        arr = {
            i: np.lib.format.open_memmap(
                t, mode="w+", dtype=np.float16, shape=(rows, self.d_model)
            )
            for i, t in tmp.items()
        }

        filled = 0
        gb = bank_bytes(self.cfg, self.d_model) / 1e9
        bar = tqdm(
            total=rows, desc=f"banking {self.cfg.condition} ({gb:.0f}GB)",
            unit="tok", unit_scale=True,
        )
        for out in self._model_batches():
            take = min(next(iter(out.values())).shape[0], rows - filled)
            for i, a in out.items():
                arr[i][filled : filled + take] = a[:take].cpu().numpy().astype(np.float16)
            filled += take
            bar.update(take)
            if filled >= rows:
                break
        bar.close()
        for a in arr.values():
            a.flush()
        arr.clear()

        if filled < rows:
            for t in tmp.values():
                t.unlink(missing_ok=True)
            raise RuntimeError(f"stream ran dry at {filled} of {rows} rows")
        # rename last: a killed build leaves no bank, not a short one
        for i, t in tmp.items():
            t.rename(paths[i])

    def _bank_batches(self, paths: dict[int, Path]) -> Iterator[dict[int, torch.Tensor]]:
        mm = {i: np.load(p, mmap_mode="r") for i, p in paths.items()}
        per_yield = self.cfg.batch_seqs * (self.cfg.seq_len - self.cfg.drop_first)
        for start in range(0, bank_rows(self.cfg), per_yield):
            # np.array copies: a memmap slice is read-only, and torch would
            # hand back a tensor that lies about being writable
            yield {
                i: torch.from_numpy(np.array(m[start : start + per_yield]))
                .float()
                .to(self.device)
                for i, m in mm.items()
            }

    def __iter__(self) -> Iterator[dict[int, torch.Tensor]]:
        if not self.scale:
            self.calibrate()
        paths = bank_paths(self.cfg)
        if not all(p.exists() for p in paths.values()):
            # every rank would otherwise race to write the same files
            from accelerate import PartialState

            with PartialState().main_process_first():
                if not all(p.exists() for p in paths.values()):
                    self.build_bank(paths)
        yield from self._bank_batches(paths)

    def close(self) -> None:
        if self._control is not None:
            self._control.remove()
            self._control = None

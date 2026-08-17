import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from do_bsfs_suck.config import Condition, TrainConfig
from do_bsfs_suck.randomize import RandomizeSpec


@dataclass(frozen=True)
class GridSpec:
    kind: str = "main"  # main | smoke
    dict_dims: int = 16384


@dataclass(frozen=True)
class ShuffledSpec:
    enabled: bool = True
    block_dims: tuple[int, ...] = (4, 16)
    active_dims: int = 64


@dataclass(frozen=True)
class SweepSpec:
    model: str = "EleutherAI/pythia-160m"
    dataset: str = "monology/pile-uncopyrighted"
    layers: tuple[int, ...] = (3, 6, 9)
    conditions: tuple[Condition, ...] = ()
    out: Path = Path("results/sweep.json")
    seed: int = 0
    device: str = "auto"
    mixed_precision: str = "bf16"
    cache_dir: Path | None = None
    wandb_project: str | None = None
    ioi: bool = False
    n_tokens: int = 1_000_000_000
    eval_tokens: int = 2_000_000
    grid: GridSpec = field(default_factory=GridSpec)
    shuffled: ShuffledSpec = field(default_factory=ShuffledSpec)
    train: TrainConfig = field(default_factory=TrainConfig)
    randomize: RandomizeSpec = field(default_factory=RandomizeSpec)


_NESTED = {
    "grid": GridSpec,
    "shuffled": ShuffledSpec,
    "train": TrainConfig,
    "randomize": RandomizeSpec,
}
_TUPLES = {"layers", "conditions", "block_dims"}
_PATHS = {"out", "cache_dir"}


def _build(cls, data: dict[str, Any], where: str):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) in {where}: {sorted(unknown)}. Known: {sorted(known)}"
        )
    kw = {}
    for key, value in data.items():
        if key in _NESTED and isinstance(value, dict):
            value = _build(_NESTED[key], value, f"{where}.{key}")
        elif key in _TUPLES and isinstance(value, list):
            value = tuple(value)
        elif key in _PATHS and value is not None:
            value = Path(value)
        kw[key] = value
    return cls(**kw)


def load_spec(path: Path) -> SweepSpec:
    """Parse a sweep config. Unknown keys are an error, not a shrug: a typo'd
    key that silently keeps the default is the kind of thing you only notice
    after the cluster time is spent."""
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping at the top level")
    spec = _build(SweepSpec, data, str(path))
    if spec.grid.kind not in ("main", "smoke"):
        raise ValueError(f"grid.kind must be 'main' or 'smoke', got {spec.grid.kind!r}")
    if spec.mixed_precision not in ("no", "bf16", "fp16", "fp8"):
        raise ValueError(
            f"mixed_precision must be no|bf16|fp16|fp8, got {spec.mixed_precision!r}"
        )
    return spec


def dump_spec(spec: SweepSpec) -> str:
    def norm(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, tuple):
            return list(v)
        if dataclasses.is_dataclass(v):
            return {k: norm(x) for k, x in dataclasses.asdict(v).items()}
        return v

    return yaml.safe_dump(
        {f.name: norm(getattr(spec, f.name)) for f in fields(spec)}, sort_keys=False
    )

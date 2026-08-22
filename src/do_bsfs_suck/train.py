import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from accelerate import Accelerator
from tqdm import tqdm

from do_bsfs_suck.config import FeaturizerConfig, TrainConfig
from do_bsfs_suck.featurizers import Featurizer, build
from do_bsfs_suck.stream import ActivationStream
from do_bsfs_suck.tracking import Tracker


@dataclass
class Run:
    """One featurizer plus the optimizer state that belongs to it."""

    cfg: FeaturizerConfig
    layer: int
    model: Featurizer
    opt: torch.optim.Optimizer
    seen: int = 0
    ema_fvu: float = float("nan")

    @property
    def key(self) -> str:
        return f"L{self.layer}/{self.cfg.name}"

    def release_optimizer(self) -> None:
        """Drop Adam's moments once this run is trained.

        Nothing downstream reads them, and they are two extra copies of every
        weight -- held to the end of the sweep they cost more than the models.
        Only safe because groups partition the runs: each is trained once.
        """
        self.opt.state.clear()


def plan(specs: dict[int, list[FeaturizerConfig]]) -> list[tuple[int, FeaturizerConfig]]:
    """Deterministic (layer, cfg) order -- every process must shard the same list."""
    return [(layer, cfg) for layer in sorted(specs) for cfg in specs[layer]]


def make_runs(
    pairs: list[tuple[int, FeaturizerConfig]],
    lr: float,
    device,
    directions: dict[int, torch.Tensor] | None = None,
) -> list[Run]:
    runs = []
    for layer, cfg in pairs:
        d = directions.get(layer) if directions else None
        model = build(cfg, d).to(device)
        runs.append(Run(cfg, layer, model, torch.optim.Adam(model.parameters(), lr=lr)))
    return runs


@torch.no_grad()
def _metrics(runs: list[Run], lr: float, dead_after: float, prefix: str) -> dict[str, float]:
    """Per-featurizer training curves, namespaced so conditions do not collide."""
    out: dict[str, float] = {"tokens": runs[0].seen, "lr": lr}
    for r in runs:
        dead = (r.model.tokens_since_fired > dead_after).float().mean().item()
        out[f"{prefix}/{r.key}/fvu"] = r.ema_fvu
        out[f"{prefix}/{r.key}/dead_frac"] = dead
    return out


def _lr_scale(step: int, total: int, warmup_frac: float) -> float:
    warm = max(int(total * warmup_frac), 1)
    if step < warm:
        return step / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))


def cotrain(
    stream: ActivationStream,
    specs: dict[int, list[FeaturizerConfig]],
    tcfg: TrainConfig,
    device: str | None = None,
    directions: dict[int, torch.Tensor] | None = None,
    tracker: Tracker | None = None,
    accelerator: Accelerator | None = None,
) -> list[Run]:
    """Train every featurizer for every layer, keeping all of them.

    Convenience over `train_groups` for callers small enough to hold the whole
    grid; `run_sweep` consumes the groups instead, so nothing but the group in
    hand stays resident.
    """
    return [
        run
        for group in train_groups(
            stream, specs, tcfg, device, directions, tracker, accelerator
        )
        for run in group
    ]


def train_groups(
    stream: ActivationStream,
    specs: dict[int, list[FeaturizerConfig]],
    tcfg: TrainConfig,
    device: str | None = None,
    directions: dict[int, torch.Tensor] | None = None,
    tracker: Tracker | None = None,
    accelerator: Accelerator | None = None,
) -> Iterator[list[Run]]:
    """Train the local shard in groups of tcfg.parallel, yielding each finished.

    A group is built only when its turn comes and is dropped as soon as the
    caller lets go, so `parallel` bounds weights and Adam state alike. Adam's
    moments go back before the yield -- nothing downstream reads them.
    """
    acc = accelerator or Accelerator()
    # shard runs across processes; each re-runs the source forward
    mine = plan(specs)[acc.process_index :: acc.num_processes]
    size = max(tcfg.parallel, 1)
    chunks = [mine[i : i + size] for i in range(0, len(mine), size)]

    for n, chunk in enumerate(chunks, 1):
        group = make_runs(chunk, tcfg.lr, device or acc.device, directions)
        _train_group(
            stream, group, tcfg, acc,
            f"{stream.cfg.condition} {n}/{len(chunks)}", tracker or Tracker(),
        )
        for run in group:
            run.release_optimizer()
        yield group
        # the caller is done with it; do not pin it while the next one builds
        del group


def _train_group(
    stream: ActivationStream,
    runs: list[Run],
    tcfg: TrainConfig,
    acc: Accelerator,
    desc: str,
    tracker: Tracker,
) -> None:
    # one forward captures every layer, so a group spanning layers is free
    layers = sorted({r.layer for r in runs})
    prefix = stream.cfg.condition
    total_steps = max(stream.cfg.n_tokens // tcfg.batch_tokens, 1)
    step = 0
    bar = tqdm(total=total_steps, desc=desc, disable=not acc.is_main_process)

    # yields need not be a multiple of batch_tokens, so buffer across them
    buf: dict[int, list[torch.Tensor]] = {i: [] for i in layers}
    held = 0

    for acts in stream:
        for i in layers:
            buf[i].append(acts[i])
        held += next(iter(acts.values())).shape[0]
        if held < tcfg.batch_tokens:
            continue

        pooled = {i: torch.cat(buf[i]) for i in layers}

        n = next(iter(pooled.values())).shape[0]
        perm = torch.randperm(n, device=acc.device)
        used = 0
        for start in range(0, n - tcfg.batch_tokens + 1, tcfg.batch_tokens):
            idx = perm[start : start + tcfg.batch_tokens]
            used = start + tcfg.batch_tokens
            scale = _lr_scale(step, total_steps, tcfg.warmup_frac)

            for run in runs:
                x = pooled[run.layer][idx].to(acc.device)
                for group in run.opt.param_groups:
                    group["lr"] = tcfg.lr * scale

                with acc.autocast():
                    pre = run.model.encode_pre(x)
                    z = run.model.sparsify(pre)
                    x_hat = run.model.decode(z)
                losses = run.model.loss(x, x_hat, z)
                losses["loss"] = losses["loss"] + tcfg.aux_coeff * run.model.aux_loss(
                    x, x_hat.detach(), pre, tcfg.dead_after_tokens, tcfg.aux_k
                )
                run.opt.zero_grad(set_to_none=True)
                acc.backward(losses["loss"])
                torch.nn.utils.clip_grad_norm_(run.model.parameters(), tcfg.grad_clip)
                run.opt.step()
                run.model.constrain()

                run.seen += x.shape[0]
                run.model.track_dead(z, x.shape[0])
                with torch.no_grad():
                    fvu = (
                        (x_hat.float() - x.float()).pow(2).sum() / x.float().pow(2).sum()
                    ).item()
                run.ema_fvu = fvu if math.isnan(run.ema_fvu) else 0.99 * run.ema_fvu + 0.01 * fvu

            if step % tcfg.log_every == 0:
                tracker.log(_metrics(runs, tcfg.lr * scale, tcfg.dead_after_tokens, prefix))
            step += 1
            bar.update(1)
            if step >= total_steps:
                bar.close()
                tracker.log(_metrics(runs, tcfg.lr * scale, tcfg.dead_after_tokens, prefix))
                return

        # carry the unconsumed tail forward
        rest = perm[used:]
        buf = {i: [pooled[i][rest]] for i in layers}
        held = int(rest.numel())

    bar.close()
    if step == 0:
        raise RuntimeError(
            f"trained 0 steps: stream produced fewer than batch_tokens="
            f"{tcfg.batch_tokens} usable tokens"
        )

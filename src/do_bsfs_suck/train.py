import math
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
def _metrics(runs: list[Run], lr: float, dead_after: float) -> dict[str, float]:
    """Per-featurizer training curves. Dead fraction is the one to watch: it is
    what AuxK exists to hold down, and it climbs with b as k = A/b shrinks."""
    out: dict[str, float] = {"tokens": runs[0].seen, "lr": lr}
    for r in runs:
        dead = (r.model.tokens_since_fired > dead_after).float().mean().item()
        out[f"{r.key}/fvu"] = r.ema_fvu
        out[f"{r.key}/dead_frac"] = dead
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
    """Train every featurizer for every layer, in groups of tcfg.parallel.

    Each group gets its own pass of the stream, so the model forward is paid
    once per group rather than once per run. Groups are bounded because every
    resident run carries its own Adam state: all 162 runs of the main grid at
    once is ~32GB of optimizer state before a single activation.
    """
    acc = accelerator or Accelerator()
    # shard runs across processes rather than data-parallelising each one: the
    # models are independent and small, so replication would buy nothing. Each
    # process re-runs the source forward, ~4% of cost, to avoid moving activations.
    mine = plan(specs)[acc.process_index :: acc.num_processes]
    runs = make_runs(mine, tcfg.lr, device or acc.device, directions)
    if not runs:
        return runs
    size = max(tcfg.parallel, 1)
    groups = [runs[i : i + size] for i in range(0, len(runs), size)]
    for n, group in enumerate(groups, 1):
        _train_group(
            stream, group, tcfg, acc,
            f"{stream.cfg.condition} {n}/{len(groups)}", tracker or Tracker(),
        )
    return runs


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
    total_steps = max(stream.cfg.n_tokens // tcfg.batch_tokens, 1)
    step = 0
    bar = tqdm(total=total_steps, desc=desc, disable=not acc.is_main_process)

    # a stream yield is batch_seqs*(seq_len-drop_first) tokens, which need not be
    # a multiple of batch_tokens -- so buffer across yields rather than dropping
    # the remainder, which would silently train on nothing when the yield is
    # smaller than one batch
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

            if step % tcfg.log_every == 0 and acc.is_main_process:
                tracker.log(_metrics(runs, tcfg.lr * scale, tcfg.dead_after_tokens))
            step += 1
            bar.update(1)
            if step >= total_steps:
                bar.close()
                if acc.is_main_process:
                    tracker.log(_metrics(runs, tcfg.lr * scale, tcfg.dead_after_tokens))
                return

        # carry the unconsumed tail forward. A yield is batch_seqs*(seq_len-1)
        # tokens and need not divide batch_tokens: dropping the remainder cost
        # 50% of the corpus at 8176/4096, and ended the run at half the bar
        # without erroring.
        rest = perm[used:]
        buf = {i: [pooled[i][rest]] for i in layers}
        held = int(rest.numel())

    bar.close()
    if step == 0:
        raise RuntimeError(
            f"trained 0 steps: stream produced fewer than batch_tokens="
            f"{tcfg.batch_tokens} usable tokens"
        )

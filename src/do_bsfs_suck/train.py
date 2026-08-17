import math
from dataclasses import dataclass

import torch
from tqdm import tqdm

from do_bsfs_suck.config import FeaturizerConfig, TrainConfig
from do_bsfs_suck.featurizers import Featurizer, build
from do_bsfs_suck.stream import ActivationStream


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


def make_runs(
    specs: dict[int, list[FeaturizerConfig]],
    lr: float,
    device: str,
    directions: dict[int, torch.Tensor] | None = None,
) -> list[Run]:
    runs = []
    for layer, cfgs in specs.items():
        for cfg in cfgs:
            d = directions.get(layer) if directions else None
            model = build(cfg, d).to(device)
            runs.append(Run(cfg, layer, model, torch.optim.Adam(model.parameters(), lr=lr)))
    return runs


def _lr_scale(step: int, total: int, warmup_frac: float) -> float:
    warm = max(int(total * warmup_frac), 1)
    if step < warm:
        return step / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))


def cotrain(
    stream: ActivationStream,
    specs: dict[int, list[FeaturizerConfig]],
    tcfg: TrainConfig,
    device: str = "cpu",
    directions: dict[int, torch.Tensor] | None = None,
) -> list[Run]:
    """Train every featurizer for every layer against a single pass of the stream.

    The model forward dominates cost, so it is paid once per condition rather
    than once per run.
    """
    runs = make_runs(specs, tcfg.lr, device, directions)
    layers = sorted(specs)
    total_steps = max(stream.cfg.n_tokens // tcfg.batch_tokens, 1)
    step = 0
    bar = tqdm(total=total_steps, desc=stream.cfg.condition)

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
        buf = {i: [] for i in layers}
        held = 0

        n = next(iter(pooled.values())).shape[0]
        perm = torch.randperm(n, device=device)
        for start in range(0, n - tcfg.batch_tokens + 1, tcfg.batch_tokens):
            idx = perm[start : start + tcfg.batch_tokens]
            scale = _lr_scale(step, total_steps, tcfg.warmup_frac)

            for run in runs:
                x = pooled[run.layer][idx].to(device)
                for group in run.opt.param_groups:
                    group["lr"] = tcfg.lr * scale

                pre = run.model.encode_pre(x)
                z = run.model.sparsify(pre)
                x_hat = run.model.decode(z)
                losses = run.model.loss(x, x_hat, z)
                losses["loss"] = losses["loss"] + tcfg.aux_coeff * run.model.aux_loss(
                    x, x_hat.detach(), pre, tcfg.dead_after_tokens, tcfg.aux_k
                )
                run.opt.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(run.model.parameters(), tcfg.grad_clip)
                run.opt.step()
                run.model.constrain()

                run.seen += x.shape[0]
                run.model.track_dead(z, x.shape[0])
                with torch.no_grad():
                    fvu = ((x_hat - x).pow(2).sum() / x.pow(2).sum()).item()
                run.ema_fvu = fvu if math.isnan(run.ema_fvu) else 0.99 * run.ema_fvu + 0.01 * fvu

            step += 1
            bar.update(1)
            if step >= total_steps:
                bar.close()
                return runs

    bar.close()
    if step == 0:
        raise RuntimeError(
            f"trained 0 steps: stream produced fewer than batch_tokens="
            f"{tcfg.batch_tokens} usable tokens"
        )
    return runs

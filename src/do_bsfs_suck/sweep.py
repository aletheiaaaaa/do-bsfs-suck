import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer

from do_bsfs_suck.config import (
    ACTIVE_DIMS,
    BLOCK_DIMS,
    BSF_VARIANTS,
    MATCHED_A,
    MATCHED_K,
    Condition,
    FeaturizerConfig,
    StreamConfig,
    TrainConfig,
)
from do_bsfs_suck.evals.absorption import absorption, summarize_absorption
from do_bsfs_suck.evals.geometry import BlockGram, stable_ranks, summarize_ranks
from do_bsfs_suck.evals.ioi import (
    ioi_activations,
    ioi_directions,
    make_ioi,
    pos_split_counts,
    summarize_ioi,
)
from do_bsfs_suck.evals.mdl import MDLStats, mdl_curve
from do_bsfs_suck.evals.probes import (
    SpellingSet,
    fit_letter_probes,
    spelling_tokens,
    token_activations,
)
from do_bsfs_suck.evals.recon import ReconStats
from do_bsfs_suck.evals.splitting import (
    split_counts,
    summarize_splits,
    supervised_directions,
    vocab_sample,
)
from do_bsfs_suck.randomize import RandomizeSpec
from do_bsfs_suck.stream import ActivationStream
from do_bsfs_suck.tracking import Tracker
from do_bsfs_suck.train import train_groups

# arms where a ground-truth attribute is linearly decodable; the rest are nulls
COMPARISON: tuple[Condition, ...] = ("trained", "rand_excl_emb")
NULLS: tuple[Condition, ...] = ("control", "rand_incl_emb")

DELTAS = (0.1, 0.2, 0.4, 0.8)


def main_grid(
    d_in: int,
    dict_dims: int = 16384,
    block_dims: tuple[int, ...] = BLOCK_DIMS,
    active_dims: tuple[int, ...] = ACTIVE_DIMS,
    variants: tuple[str, ...] = BSF_VARIANTS,
    matched_k: int = MATCHED_K,
    seed: int = 0,
) -> list[FeaturizerConfig]:
    """BSF variants over b x A, plus the b=1 SAE baseline and the matched-k arm."""
    out = []
    for a in active_dims:
        for b in block_dims:
            if b > a:
                continue
            for variant in variants:
                out.append(
                    FeaturizerConfig(
                        d_in=d_in, variant=variant, dict_dims=dict_dims,
                        block_dim=b, active_dims=a, seed=seed,
                    )
                )
        out.append(
            FeaturizerConfig(
                d_in=d_in, variant="topk_sae", dict_dims=dict_dims,
                block_dim=1, active_dims=a, seed=seed,
            )
        )
    for b in block_dims:
        for variant in variants:
            out.append(
                FeaturizerConfig(
                    d_in=d_in, variant=variant, dict_dims=dict_dims,
                    block_dim=b, active_dims=matched_k * b, seed=seed,
                )
            )
    # the arms share their (b, A) corners; train each config once
    return list(dict.fromkeys(out))


def shuffled_grid(
    d_in: int, dict_dims: int = 16384, block_dims=(4, 16), active_dims: int = MATCHED_A, seed: int = 0
) -> list[FeaturizerConfig]:
    """The control needs a point on both arms."""
    out = [
        FeaturizerConfig(
            d_in=d_in, variant="shuffled", dict_dims=dict_dims,
            block_dim=b, active_dims=a, seed=seed,
        )
        for b in block_dims
        for a in (active_dims, MATCHED_K * b)
    ]
    return list(dict.fromkeys(out))


def _keep_directions(found: dict[int, torch.Tensor], runs) -> None:
    """Copy out the b=1 vanilla decoders before their group is dropped."""
    for run in runs:
        if run.cfg.variant == "vanilla" and run.cfg.block_dim == 1:
            found.setdefault(run.layer, run.model.W_dec.detach().squeeze(1).cpu().clone())


def _gather_directions(found: dict[int, torch.Tensor], layers, acc=None):
    """The b=1 vanilla decoder per layer, which the shuffled control regroups."""
    out = dict(found)
    if acc is not None and acc.num_processes > 1:
        from accelerate.utils import gather_object

        merged: dict[int, torch.Tensor] = {}
        for part in gather_object([out]):
            merged.update(part)
        out = merged

    missing = set(layers) - set(out)
    if missing:
        raise ValueError(f"no b=1 vanilla run to seed the shuffled control at {missing}")
    return out


EVAL_CHUNK = 1024


@torch.no_grad()
def _eval_pass(
    runs, stream: ActivationStream, layers: tuple[int, ...], chunk: int = EVAL_CHUNK
) -> dict[str, dict]:
    """One eval pass, chunked to bound code memory."""
    stats = {r.key: ReconStats() for r in runs}
    grams = {r.key: BlockGram(r.model.n_blocks, r.model.block_dim) for r in runs}
    mdls = {r.key: MDLStats(r.model.n_blocks) for r in runs}

    for acts in stream:
        for r in runs:
            for x in acts[r.layer].split(chunk):
                x_hat, z = r.model(x)
                stats[r.key].update(x, x_hat, z)
                grams[r.key].update(z)
                mdls[r.key].update(x, x_hat, z)

    return {
        r.key: {
            **stats[r.key].summary(),
            **summarize_ranks(stable_ranks(grams[r.key], r.model)),
            "mdl": mdl_curve(mdls[r.key], r.model, DELTAS),
        }
        for r in runs
    }


@dataclass(frozen=True)
class ConceptContext:
    """Per layer, everything absorption and splitting need that is not a run.

    Fitting the probes costs a forward pass over the vocabulary, so it is done
    once per condition rather than once per group of featurizers.
    """

    acts: dict[int, torch.Tensor]  # spelling activations, test split only
    letters: dict[int, list[str]]  # their labels, in the same order
    probes: dict[int, dict]  # letter -> (direction, prediction on the split)
    directions: dict[int, torch.Tensor]


def _concept_context(model, tokenizer, layers, scale, mean, seed: int) -> ConceptContext:
    spell = spelling_tokens(tokenizer, limit=4000, seed=seed)
    spell_acts = token_activations(model, tokenizer, spell, layers, scale=scale, mean=mean)

    ids, strings = vocab_sample(tokenizer, n=8000, seed=seed)
    vocab_set = SpellingSet(ids, strings, ["?"] * len(ids))
    vocab_acts = token_activations(model, tokenizer, vocab_set, layers, scale=scale, mean=mean)

    acts, letters, probes = {}, {}, {}
    for L in layers:
        fitted = fit_letter_probes(spell_acts[L], spell.letters, seed=seed)
        idx = next(iter(fitted.values())).test_idx
        acts[L] = spell_acts[L][idx]
        letters[L] = [spell.letters[i] for i in idx]
        probes[L] = {letter: (p.direction, p.predicted) for letter, p in fitted.items()}

    return ConceptContext(
        acts, letters, probes,
        {L: supervised_directions(vocab_acts[L], strings) for L in layers},
    )


def _concept_evals(runs, ctx: ConceptContext) -> dict[str, dict]:
    """Absorption and splitting, which need ground-truth token attributes."""
    out = {}
    for r in runs:
        L = r.layer
        per_letter = absorption(r.model, ctx.acts[L], ctx.letters[L], ctx.probes[L])
        out[r.key] = {
            **summarize_absorption(per_letter),
            **summarize_splits(split_counts(r.model, ctx.directions[L])),
        }
    return out


def _ioi_context(model, tokenizer, layers, scale, mean, seed: int) -> dict[int, torch.Tensor]:
    data = make_ioi(tokenizer, n=2048, seed=seed)
    acts = ioi_activations(model, tokenizer, data, layers, scale=scale, mean=mean)
    return {L: ioi_directions(acts[L], data) for L in layers}


def _ioi_evals(runs, dirs: dict[int, torch.Tensor]) -> dict[str, dict]:
    """Makelov et al.'s IOI oversplitting check."""
    return {
        r.key: summarize_ioi(pos_split_counts(r.model, dirs[r.layer])) for r in runs
    }


@dataclass
class Scoring:
    """Everything a group is scored against, fixed once per condition.

    Groups are evaluated as they finish training so their models can be freed,
    which means every metric has to be reducible to rows here and now.
    """

    condition: Condition
    model_name: str
    layers: tuple[int, ...]
    eval_stream: ActivationStream
    concept: ConceptContext
    ioi: dict[int, torch.Tensor] | None
    done: set[tuple]
    # b=1 vanilla decoders, kept back for the shuffled control
    found: dict[int, torch.Tensor] = field(default_factory=dict)

    def rows(self, group) -> list[dict]:
        _keep_directions(self.found, group)
        metrics = _eval_pass(group, self.eval_stream, self.layers)
        concept = _concept_evals(group, self.concept)
        if self.ioi is not None:
            extra = _ioi_evals(group, self.ioi)
            concept = {k: {**v, **extra.get(k, {})} for k, v in concept.items()}

        out = []
        for r in group:
            # a condition retrains whole; don't append finished configs twice
            if (self.condition, r.layer, r.cfg.name) in self.done:
                continue
            out.append(
                {
                    "condition": self.condition, "layer": r.layer, "name": r.cfg.name,
                    "variant": r.cfg.variant, "block_dim": r.cfg.block_dim,
                    "active_dims": r.cfg.active_dims, "n_blocks": r.cfg.n_blocks,
                    "k": r.cfg.k, "seed": r.cfg.seed, "model": self.model_name,
                    "is_null_arm": self.condition in NULLS,
                    **metrics[r.key], **concept[r.key],
                }
            )
        return out


def run_sweep(
    model_name: str,
    layers: tuple[int, ...],
    conditions: tuple[Condition, ...],
    grid,
    n_tokens: int,
    out_path: Path,
    shuffled=None,
    ioi: bool = False,
    dataset: str = "monology/pile-uncopyrighted",
    eval_tokens: int = 2_000_000,
    device: str = "cpu",
    seed: int = 0,
    spec: RandomizeSpec = RandomizeSpec(),
    train_cfg: TrainConfig = TrainConfig(),
    cache_dir: Path | None = None,
    wandb_project: str | None = None,
    mixed_precision: str = "no",
) -> list[dict]:
    acc = Accelerator(mixed_precision=mixed_precision)
    device = str(acc.device) if device == "auto" else device
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tracker = Tracker(
        wandb_project, name=f"{model_name.split('/')[-1]}_s{seed}",
        rank=acc.process_index, world_size=acc.num_processes,
        model=model_name, layers=list(layers), conditions=list(conditions),
        n_tokens=n_tokens, eval_tokens=eval_tokens, seed=seed,
        train=train_cfg, randomize=spec, mixed_precision=mixed_precision,
    )
    rows: list[dict] = []
    if out_path.exists():
        rows = json.loads(out_path.read_text())
    done = {(r["condition"], r["layer"], r["name"]) for r in rows}

    for condition in conditions:
        scfg = StreamConfig(
            model=model_name, condition=condition, layers=layers,
            dataset=dataset, n_tokens=n_tokens, seed=seed, data_seed=seed,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        stream = ActivationStream(scfg, spec, device=device)
        cfgs = grid(stream.d_model, seed=seed)
        specs = {L: cfgs for L in layers}

        if all((condition, L, c.name) in done for L in layers for c in cfgs):
            stream.close()
            continue

        # the eval arms reuse the training calibration, so fix it up front
        stream.calibrate()
        eval_stream = ActivationStream(
            dataclasses.replace(scfg, n_tokens=eval_tokens, data_seed=seed + 1000),
            spec, device=device,
        )
        eval_stream.scale, eval_stream.mean = stream.scale, stream.mean
        concept_ctx = _concept_context(
            stream.model, tokenizer, layers, stream.scale, stream.mean, seed
        )
        ioi_ctx = (
            _ioi_context(stream.model, tokenizer, layers, stream.scale, stream.mean, seed)
            if ioi
            else None
        )

        scoring = Scoring(
            condition=condition, model_name=model_name, layers=layers,
            eval_stream=eval_stream, concept=concept_ctx, ioi=ioi_ctx, done=done,
        )
        fresh: list[dict] = []

        for group in train_groups(
            stream, specs, train_cfg, device=device, tracker=tracker, accelerator=acc
        ):
            fresh.extend(scoring.rows(group))
            del group

        # the shuffled control regroups a *trained* b=1 dictionary
        if shuffled is not None:
            shuf_stream = ActivationStream(
                dataclasses.replace(scfg, data_seed=seed + 500), spec, device=device
            )
            shuf_stream.scale, shuf_stream.mean = stream.scale, stream.mean
            for group in train_groups(
                shuf_stream, {L: shuffled(stream.d_model, seed=seed) for L in layers},
                train_cfg, device=device,
                directions=_gather_directions(scoring.found, layers, acc),
                tracker=tracker, accelerator=acc,
            ):
                fresh.extend(scoring.rows(group))
                del group
            shuf_stream.close()

        eval_stream.close()
        stream.close()
        # each rank trained a different shard
        if acc.num_processes > 1:
            from accelerate.utils import gather_object

            fresh = [row for part in gather_object([fresh]) for row in part]
        rows.extend(fresh)
        done.update((r["condition"], r["layer"], r["name"]) for r in fresh)

        if acc.is_main_process:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(rows, indent=2))

    if acc.is_main_process:
        tracker.table("results", rows)
    tracker.finish()
    return rows

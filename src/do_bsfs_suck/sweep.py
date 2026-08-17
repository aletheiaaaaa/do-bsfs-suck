import dataclasses
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from do_bsfs_suck.config import Condition, FeaturizerConfig, StreamConfig, TrainConfig
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
from do_bsfs_suck.train import cotrain

# arms where a ground-truth attribute is linearly decodable, so absorption and
# splitting mean something; the rest are null calibration only
COMPARISON: tuple[Condition, ...] = ("trained", "rand_excl_emb", "step0_excl_emb")
NULLS: tuple[Condition, ...] = ("control", "step0", "rand_incl_emb")

DELTAS = (0.1, 0.2, 0.4, 0.8)
BLOCK_DIMS = (1, 2, 4, 8, 16)
ACTIVE_DIMS = (32, 64, 128)
BSF_VARIANTS = ("vanilla", "grassmann", "group_lasso")
# the matched-k arm: A = MATCHED_K * b, so every b fires the same number of
# blocks. Absorption is confounded across b at matched A -- raising b shrinks
# k = A/b, so fewer blocks fire and a silent parent gets likelier for reasons
# that have nothing to do with absorption. Only this arm isolates b itself.
MATCHED_K = 32
# the A the depth and shuffled grids hold, so the two arms cross there
MATCHED_A = 64


def main_grid(d_in: int, dict_dims: int = 16384, seed: int = 0) -> list[FeaturizerConfig]:
    """3 BSF variants over b x A, plus the b=1 SAE baseline at each A, plus the
    matched-k arm. Equal A means equal nonzeros; equal k means equal blocks."""
    out = []
    for a in ACTIVE_DIMS:
        for b in BLOCK_DIMS:
            if b > a:
                continue
            for variant in BSF_VARIANTS:
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
    for b in BLOCK_DIMS:
        for variant in BSF_VARIANTS:
            out.append(
                FeaturizerConfig(
                    d_in=d_in, variant=variant, dict_dims=dict_dims,
                    block_dim=b, active_dims=MATCHED_K * b, seed=seed,
                )
            )
    # the two arms share their (b, A) corners; train each config once
    return list(dict.fromkeys(out))


def shuffled_grid(
    d_in: int, dict_dims: int = 16384, block_dims=(4, 16), active_dims: int = MATCHED_A, seed: int = 0
) -> list[FeaturizerConfig]:
    """The control needs a point on *both* arms, or it drops out of whichever
    figure holds the other one fixed."""
    out = [
        FeaturizerConfig(
            d_in=d_in, variant="shuffled", dict_dims=dict_dims,
            block_dim=b, active_dims=a, seed=seed,
        )
        for b in block_dims
        for a in (active_dims, MATCHED_K * b)
    ]
    return list(dict.fromkeys(out))


def _trained_directions(runs, layers) -> dict[int, torch.Tensor]:
    """The b=1 vanilla decoder per layer, which the shuffled control regroups."""
    out = {}
    for run in runs:
        if run.cfg.variant == "vanilla" and run.cfg.block_dim == 1:
            out.setdefault(run.layer, run.model.W_dec.detach().squeeze(1).clone())
    missing = set(layers) - set(out)
    if missing:
        raise ValueError(f"no b=1 vanilla run to seed the shuffled control at {missing}")
    return out


def smoke_grid(d_in: int, seed: int = 0) -> list[FeaturizerConfig]:
    # (b=1, A=2) and (b=4, A=8) are the matched-k pair, both at k=2
    return [
        FeaturizerConfig(d_in=d_in, variant=v, dict_dims=512, block_dim=b, active_dims=a, seed=seed)
        for v, b, a in [
            ("topk_sae", 1, 8), ("vanilla", 1, 8), ("vanilla", 1, 2),
            ("vanilla", 4, 8), ("grassmann", 4, 8),
        ]
    ]


EVAL_CHUNK = 1024


@torch.no_grad()
def _eval_pass(
    runs, stream: ActivationStream, layers: tuple[int, ...], chunk: int = EVAL_CHUNK
) -> dict[str, dict]:
    """Chunked: a whole stream yield through one featurizer is an (N, G, b) code
    of ~500MB at G*b = 16384, which is enough to OOM a workstation."""
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


def _concept_evals(runs, model, tokenizer, layers, scale, mean, seed: int) -> dict[str, dict]:
    """Absorption and splitting, which need ground-truth token attributes."""
    spell = spelling_tokens(tokenizer, limit=4000, seed=seed)
    spell_acts = token_activations(model, tokenizer, spell, layers, scale=scale, mean=mean)

    ids, strings = vocab_sample(tokenizer, n=8000, seed=seed)
    vocab_set = SpellingSet(ids, strings, ["?"] * len(ids))
    vocab_acts = token_activations(model, tokenizer, vocab_set, layers, scale=scale, mean=mean)

    probes = {L: fit_letter_probes(spell_acts[L], spell.letters, seed=seed) for L in layers}
    directions = {
        L: supervised_directions(vocab_acts[L], strings) for L in layers
    }

    out = {}
    for r in runs:
        L = r.layer
        pr = {
            letter: (p.direction, p.predicted) for letter, p in probes[L].items()
        }
        idx = next(iter(probes[L].values())).test_idx
        letters_te = [spell.letters[i] for i in idx]
        per_letter = absorption(r.model, spell_acts[L][idx], letters_te, pr)
        out[r.key] = {
            **summarize_absorption(per_letter),
            **summarize_splits(split_counts(r.model, directions[L])),
        }
    return out


def _ioi_evals(runs, model, tokenizer, layers, scale, mean, seed: int) -> dict[str, dict]:
    """Makelov et al.'s IOI oversplitting check, as a cross-model validation of
    the portable split-counting proxy."""
    data = make_ioi(tokenizer, n=2048, seed=seed)
    acts = ioi_activations(model, tokenizer, data, layers, scale=scale, mean=mean)
    dirs = {L: ioi_directions(acts[L], data) for L in layers}
    return {
        r.key: summarize_ioi(pos_split_counts(r.model, dirs[r.layer])) for r in runs
    }


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
) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tracker = Tracker(
        wandb_project, name=f"{model_name.split('/')[-1]}_s{seed}",
        model=model_name, layers=list(layers), conditions=list(conditions),
        n_tokens=n_tokens, eval_tokens=eval_tokens, seed=seed,
        train=train_cfg, randomize=spec,
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

        runs = cotrain(stream, specs, train_cfg, device=device, tracker=tracker)

        # the shuffled control regroups a *trained* b=1 dictionary, so it needs a
        # second pass once that dictionary exists
        if shuffled is not None:
            directions = _trained_directions(runs, layers)
            shuf_cfgs = shuffled(stream.d_model, seed=seed)
            shuf_stream = ActivationStream(
                dataclasses.replace(scfg, data_seed=seed + 500), spec, device=device
            )
            shuf_stream.scale, shuf_stream.mean = stream.scale, stream.mean
            runs += cotrain(
                shuf_stream, {L: shuf_cfgs for L in layers}, train_cfg,
                device=device, directions=directions, tracker=tracker,
            )
            shuf_stream.close()

        eval_stream = ActivationStream(
            dataclasses.replace(scfg, n_tokens=eval_tokens, data_seed=seed + 1000), spec, device=device
        )
        eval_stream.scale = stream.scale
        eval_stream.mean = stream.mean
        metrics = _eval_pass(runs, eval_stream, layers)
        eval_stream.close()

        concept = _concept_evals(runs, stream.model, tokenizer, layers, stream.scale, stream.mean, seed)
        if ioi:
            extra = _ioi_evals(
                runs, stream.model, tokenizer, layers, stream.scale, stream.mean, seed
            )
            concept = {k: {**v, **extra.get(k, {})} for k, v in concept.items()}
        stream.close()

        for r in runs:
            # a condition is retrained whole when any of its configs is new, so
            # the finished ones must not be appended twice
            if (condition, r.layer, r.cfg.name) in done:
                continue
            rows.append(
                {
                    "condition": condition, "layer": r.layer, "name": r.cfg.name,
                    "variant": r.cfg.variant, "block_dim": r.cfg.block_dim,
                    "active_dims": r.cfg.active_dims, "n_blocks": r.cfg.n_blocks,
                    "k": r.cfg.k, "seed": r.cfg.seed, "model": model_name,
                    "is_null_arm": condition in NULLS,
                    **metrics[r.key], **concept[r.key],
                }
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2))

    tracker.table("results", rows)
    tracker.finish()
    return rows

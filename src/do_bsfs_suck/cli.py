import argparse
from pathlib import Path

from do_bsfs_suck.config import TrainConfig
from do_bsfs_suck.randomize import RandomizeSpec
from do_bsfs_suck.sweep import (
    COMPARISON,
    NULLS,
    main_grid,
    run_sweep,
    shuffled_grid,
    smoke_grid,
)

SMOKE_MODEL = "EleutherAI/pythia-14m"
SMOKE_DATASET = "NeelNanda/pile-10k"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="do-bsfs-suck",
        description="Evaluate block-sparse featurizers on trained vs randomly "
        "initialized transformers, for reconstruction, MDL, block geometry, "
        "feature absorption and oversplitting.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    sweep = commands.add_parser("sweep", help="train and evaluate a grid")
    sweep.add_argument("--model", default="EleutherAI/pythia-160m")
    # None, not the default string, so --smoke can tell "unset" from "asked for
    # the real corpus" -- the smoke dataset is uncompressed, so without this the
    # production .zst path is unreachable from any local command
    sweep.add_argument("--dataset", default=None)
    sweep.add_argument("--layers", type=int, nargs="+", default=[3, 6, 9])
    sweep.add_argument(
        "--conditions", nargs="+", default=list(COMPARISON + NULLS),
        help=f"comparison arms {COMPARISON}; nulls {NULLS}",
    )
    sweep.add_argument("--tokens", type=int, default=1_000_000_000)
    sweep.add_argument("--eval-tokens", type=int, default=2_000_000)
    sweep.add_argument("--dict-dims", type=int, default=16384)
    sweep.add_argument("--batch-tokens", type=int, default=4096)
    sweep.add_argument("--lr", type=float, default=3e-4)
    sweep.add_argument(
        "--parallel", type=int, default=8,
        help="featurizers trained per stream pass; each carries its own Adam state",
    )
    sweep.add_argument(
        "--cache-dir", type=Path, default=None,
        help="memmap tokenized ids here (~4GB per 1B tokens); omit to re-tokenize every pass",
    )
    sweep.add_argument("--device", default="cpu")
    sweep.add_argument("--seed", type=int, default=0)
    sweep.add_argument("--out", type=Path, default=Path("results/sweep.json"))
    sweep.add_argument(
        "--smoke", action="store_true",
        help=f"tiny end-to-end run on {SMOKE_MODEL}, for a CPU box",
    )
    sweep.add_argument(
        "--ioi", action="store_true",
        help="also run the gpt2 IOI oversplitting check (Makelov et al.)",
    )
    sweep.add_argument("--resample-layernorm", action="store_true")
    sweep.add_argument("--no-freeze-unembed", action="store_true")

    figures = commands.add_parser("figures", help="plot a finished sweep")
    figures.add_argument("--results", type=Path, default=Path("results/sweep.json"))
    figures.add_argument("--out", type=Path, default=Path("figures"))

    return parser


def _sweep(args: argparse.Namespace) -> None:
    spec = RandomizeSpec(
        resample_layernorm=args.resample_layernorm,
        freeze_unembed=not args.no_freeze_unembed,
    )
    if args.smoke:
        run_sweep(
            model_name=SMOKE_MODEL, layers=(2, 4), conditions=("trained", "rand_excl_emb"),
            grid=smoke_grid,
            shuffled=lambda d, seed=0: shuffled_grid(d, dict_dims=512, block_dims=(4,), active_dims=8, seed=seed),
            n_tokens=200_000, eval_tokens=40_000,
            dataset=args.dataset or SMOKE_DATASET,
            out_path=args.out, device=args.device, seed=args.seed,
            spec=spec, cache_dir=args.cache_dir,
            train_cfg=TrainConfig(batch_tokens=1024, lr=args.lr, parallel=args.parallel),
        )
        return

    def grid(d_in: int, seed: int = 0):
        return main_grid(d_in, dict_dims=args.dict_dims, seed=seed)

    def shuffled(d_in: int, seed: int = 0):
        return shuffled_grid(d_in, dict_dims=args.dict_dims, seed=seed)

    run_sweep(
        model_name=args.model, layers=tuple(args.layers),
        conditions=tuple(args.conditions), grid=grid, shuffled=shuffled, ioi=args.ioi,
        n_tokens=args.tokens,
        eval_tokens=args.eval_tokens,
        dataset=args.dataset or "monology/pile-uncopyrighted", out_path=args.out,
        device=args.device, seed=args.seed, spec=spec, cache_dir=args.cache_dir,
        train_cfg=TrainConfig(
            batch_tokens=args.batch_tokens, lr=args.lr, parallel=args.parallel
        ),
    )


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "sweep":
        _sweep(args)
    elif args.command == "figures":
        from do_bsfs_suck.figures import render

        render(args.results, args.out)

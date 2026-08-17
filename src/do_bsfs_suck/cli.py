import argparse
from pathlib import Path

from do_bsfs_suck.spec import SweepSpec, dump_spec, load_spec
from do_bsfs_suck.sweep import COMPARISON, NULLS, main_grid, run_sweep, shuffled_grid, smoke_grid


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="do-bsfs-suck",
        description="Evaluate block-sparse featurizers on trained vs randomly "
        "initialized transformers, for reconstruction, MDL, block geometry, "
        "feature absorption and oversplitting.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    sweep = commands.add_parser("sweep", help="train and evaluate a grid")
    sweep.add_argument("config", type=Path, help="sweep config.yaml; see configs/")

    figures = commands.add_parser("figures", help="plot a finished sweep")
    figures.add_argument("--results", type=Path, default=Path("results/sweep.json"))
    figures.add_argument("--out", type=Path, default=Path("figures"))

    show = commands.add_parser("config", help="print a config with defaults filled in")
    show.add_argument("config", type=Path, nargs="?")

    return parser


def _sweep(spec: SweepSpec) -> None:
    dims = spec.grid.dict_dims

    def grid(d_in: int, seed: int = 0):
        if spec.grid.kind == "smoke":
            return smoke_grid(d_in, seed=seed)
        return main_grid(d_in, dict_dims=dims, seed=seed)

    shuffled = None
    if spec.shuffled.enabled:

        def shuffled(d_in: int, seed: int = 0):
            return shuffled_grid(
                d_in, dict_dims=dims, block_dims=spec.shuffled.block_dims,
                active_dims=spec.shuffled.active_dims, seed=seed,
            )

    run_sweep(
        model_name=spec.model, layers=spec.layers,
        conditions=spec.conditions or (COMPARISON + NULLS),
        grid=grid, shuffled=shuffled, ioi=spec.ioi,
        n_tokens=spec.n_tokens, eval_tokens=spec.eval_tokens,
        dataset=spec.dataset, out_path=spec.out, device=spec.device,
        seed=spec.seed, spec=spec.randomize, train_cfg=spec.train,
        cache_dir=spec.cache_dir, wandb_project=spec.wandb_project,
        mixed_precision=spec.mixed_precision,
    )


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "sweep":
        _sweep(load_spec(args.config))
    elif args.command == "config":
        print(dump_spec(load_spec(args.config) if args.config else SweepSpec()), end="")
    elif args.command == "figures":
        from do_bsfs_suck.figures import render

        render(args.results, args.out)

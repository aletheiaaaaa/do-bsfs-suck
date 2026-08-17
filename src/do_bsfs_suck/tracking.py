import dataclasses
from typing import Any


class Tracker:
    """wandb wrapper that no-ops when no project is set or wandb is absent.

    Every metric is keyed by featurizer, and `tokens` is the x-axis rather than
    the step count: parallel groups each restart their step counter, so step
    would fold unrelated runs onto the same abscissa.
    """

    def __init__(self, project: str | None = None, name: str | None = None, **config: Any):
        self.run = None
        if not project:
            return
        try:
            import wandb
        except ImportError:
            return

        self.run = wandb.init(project=project, name=name, config=_flatten(config))
        wandb.define_metric("tokens")
        wandb.define_metric("*", step_metric="tokens")

    def log(self, data: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.log(data)

    def table(self, key: str, rows: list[dict]) -> None:
        if self.run is None or not rows:
            return
        import wandb

        cols = sorted({k for r in rows for k in r if not isinstance(r[k], (list, dict))})
        self.run.log(
            {key: wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows])}
        )

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None


def _flatten(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.items():
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for k, v in dataclasses.asdict(value).items():
                out[f"{key}.{k}"] = v
        else:
            out[key] = value
    return out

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from do_bsfs_suck.config import MATCHED_A, MATCHED_K  # noqa: E402
from do_bsfs_suck.sweep import COMPARISON  # noqa: E402

# comparison arms take the first three slots; nulls are recessive gray
COLORS = {
    "trained": "#2a78d6",
    "rand_excl_emb": "#eb6834",
    "step0_excl_emb": "#1baf7a",
}
NULL_COLOR = "#8a8a85"
# markers carry identity too
MARKERS = {"trained": "o", "rand_excl_emb": "s", "step0_excl_emb": "^"}
NULL_MARKER = "x"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def _series(ax, rows, x_key: str, y_key: str) -> None:
    """One line per condition, averaging rows that share an x."""
    for condition in sorted({r["condition"] for r in rows}):
        grouped: dict[float, list[float]] = {}
        for r in rows:
            if r["condition"] == condition and r.get(y_key) is not None:
                grouped.setdefault(r[x_key], []).append(r[y_key])
        if not grouped:
            continue

        xs = sorted(grouped)
        ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]
        lo = [min(grouped[x]) for x in xs]
        hi = [max(grouped[x]) for x in xs]

        is_null = condition not in COMPARISON
        color = NULL_COLOR if is_null else COLORS.get(condition, NULL_COLOR)
        if any(h > l for l, h in zip(lo, hi)):
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=1)
        ax.plot(
            xs, ys, color=color,
            marker=NULL_MARKER if is_null else MARKERS.get(condition, NULL_MARKER),
            markersize=5, linewidth=2 if not is_null else 1.2,
            linestyle="-" if not is_null else "--",
            label=condition, zorder=3 if not is_null else 2,
        )


def stable_rank(rows, out: Path) -> None:
    """The headline: do random-transformer blocks use their whole budget?"""
    variants = sorted({r["variant"] for r in rows if r["variant"] != "topk_sae"})
    fig, axes = plt.subplots(1, len(variants), figsize=(4.2 * len(variants), 3.6), squeeze=False)

    for ax, variant in zip(axes[0], variants):
        sub = [r for r in rows if r["variant"] == variant]
        bs = sorted({r["block_dim"] for r in sub})
        # isotropic codes saturate at b
        ax.plot(bs, bs, color=MUTED, linewidth=1, linestyle=":", label="isotropic (srank = b)")
        ax.axhspan(2, 4, color=GRID, alpha=0.5, zorder=1, label="paper's 2-4 band")
        _series(ax, sub, "block_dim", "srank_codes")
        _style(ax, "block dim b", "stable rank", variant)
        ax.set_xscale("log", base=2)

    axes[0][-1].legend(frameon=False, fontsize=8, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(out / "stable_rank_vs_b.png", dpi=160)
    plt.close(fig)


def reconstruction(rows, out: Path) -> None:
    variants = sorted({r["variant"] for r in rows})
    fig, axes = plt.subplots(1, len(variants), figsize=(3.6 * len(variants), 3.4), squeeze=False)
    for ax, variant in zip(axes[0], variants):
        _series(ax, [r for r in rows if r["variant"] == variant], "active_dims", "fvu")
        _style(ax, "active dims A = k*b", "FVU", variant)
        ax.set_xscale("log", base=2)
    axes[0][-1].legend(frameon=False, fontsize=8, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(out / "fvu_vs_active_dims.png", dpi=160)
    plt.close(fig)


ABSORPTION_KEYS = [
    ("absorption_rate", "absorption rate"),
    ("main_fire_rate", "main-block fire rate"),
    ("containment", "probe containment"),
]


def _held(rows, key: str, pinned: int) -> int | None:
    """The value of `key` to hold fixed."""
    span: dict[int, set] = {}
    for r in rows:
        span.setdefault(r[key], set()).add(r["block_dim"])
    if len(span.get(pinned, ())) > 1:
        return pinned
    return max(span, key=lambda v: len(span[v])) if span else None


def _absorption_panel(rows, path: Path, held: str) -> None:
    """Rows are metrics, columns variants."""
    variants = sorted({r["variant"] for r in rows})
    fig, axes = plt.subplots(
        3, len(variants), figsize=(3.5 * len(variants), 9.4), squeeze=False
    )
    for col, variant in enumerate(variants):
        sub = [r for r in rows if r["variant"] == variant]
        for row, (key, label) in enumerate(ABSORPTION_KEYS):
            ax = axes[row][col]
            _series(ax, sub, "block_dim", key)
            _style(ax, "block dim b", label, variant if row == 0 else "")
            ax.set_xscale("log", base=2)
    axes[0][-1].legend(frameon=False, fontsize=8, labelcolor=MUTED)
    fig.suptitle(held, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def absorption(rows, out: Path) -> None:
    """One panel per matching arm."""
    rows = [r for r in rows if "absorption_rate" in r]
    if not rows:
        return
    for key, pinned, stem, label in (
        ("active_dims", MATCHED_A, "matched_A", "active dims A = k*b"),
        ("k", MATCHED_K, "matched_k", "blocks per token k"),
    ):
        value = _held(rows, key, pinned)
        sub = [r for r in rows if r[key] == value]
        if len({r["block_dim"] for r in sub}) < 2:
            continue
        _absorption_panel(
            sub, out / f"absorption_{stem}.png", f"absorption vs b, holding {label} = {value}"
        )


def mdl(rows, out: Path) -> None:
    rows = [r for r in rows if r.get("mdl")]
    if not rows:
        return
    variants = sorted({r["variant"] for r in rows})
    fig, axes = plt.subplots(
        1, len(variants), figsize=(3.6 * len(variants), 3.4), squeeze=False
    )
    for ax, variant in zip(axes[0], variants):
        flat = [
            {"condition": r["condition"], "delta": m["delta"], "bits": m["bits_total"]}
            for r in rows
            if r["variant"] == variant
            for m in r["mdl"]
        ]
        _series(ax, flat, "delta", "bits")
        _style(ax, "distortion delta", "description length (bits/token)", variant)
        ax.set_xscale("log")
    axes[0][-1].legend(frameon=False, fontsize=8, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(out / "mdl_vs_delta.png", dpi=160)
    plt.close(fig)


def table(rows, out: Path) -> None:
    """The table view the contrast relief rule requires."""
    skip = {"mdl"}
    cols = sorted({k for r in rows for k in r if k not in skip})
    with (out / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def render(results: Path, out: Path) -> None:
    rows = json.loads(results.read_text())
    out.mkdir(parents=True, exist_ok=True)
    stable_rank(rows, out)
    reconstruction(rows, out)
    absorption(rows, out)
    mdl(rows, out)
    table(rows, out)

"""
Early exit on lm_eval benchmarks: one figure per JSON file (model x task)
produced by main_benchmarks.py. All metrics are expressed as percentages.

Panels:
  - one per strategy (dual axis):
      solid lines  (left axis)  : benchmark score(s) vs threshold, baseline = dashed line
      dotted line  (right axis) : skipped layers / total layers
  - trade-off: main score vs skipped layers, all strategies (baseline = star at 0%)

Usage: python plot_benchmarks.py results/benchmarks/<model>_<quant>_<task>_xxx.json [...]
Output: plots/benchmarks/<same name>.png
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # figures are only saved (also called during main_benchmarks runs)
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUT_DIR = Path(__file__).resolve().parent / "plots" / "benchmarks"
BASELINE = "NoExitStrategy"

METRIC_COLOR = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]
STRAT_COLOR = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
PAT_MARKER = ["o", "s", "^", "D"]
PCT = PercentFormatter(xmax=100, decimals=0)
LABEL_BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)


# --- metrics in % ------------------------------------------------------------
def score_metrics(results: dict) -> list:
    """Benchmark scores (e.g. 'acc,none', 'exact_match,strict-match'), without stderr."""
    return [k for k, v in results.items() if isinstance(v, float) and "stderr" not in k]


def metric_name(key: str) -> str:
    name, _, filt = key.partition(",")
    return name if filt in ("", "none") else f"{name} ({filt})"


score = lambda r, m: 100 * r["results"].get(m, math.nan)
skipped = lambda r: 100 * (1 - r["results"]["early_exit"]["mean_depth_ratio"])


def fmt(v):
    return f"{v:.0f}%" if abs(v) >= 10 else f"{v:.1f}%"


def annotate(ax, xs, ys, color, above=True):
    for x, y in zip(xs, ys):
        ax.annotate(fmt(y), (x, y), textcoords="offset points",
                    xytext=(0, 8 if above else -8), ha="center",
                    va="bottom" if above else "top", fontsize=7, color=color,
                    bbox=LABEL_BOX, zorder=10)


def style_x(ax, thresholds):
    ax.set_xlabel("threshold")
    ax.set_xticks(thresholds)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    pad = 0.1 * ((max(thresholds) - min(thresholds)) or 1)
    ax.set_xlim(min(thresholds) - pad, max(thresholds) + pad)
    ax.grid(alpha=0.3)


# --- panels ------------------------------------------------------------------
def strategy_panel(ax, sub, base, metrics, strat):
    ax2 = ax.twinx()
    patiences = sorted({r["strategy"].get("patience") for r in sub}, key=str)
    thresholds = sorted({r["strategy"]["threshold"] for r in sub})

    for i, m in enumerate(metrics):
        color = METRIC_COLOR[i % len(METRIC_COLOR)]
        ax.axhline(score(base, m), color=color, linestyle="--", linewidth=1.2, alpha=0.8)
        for j, pat in enumerate(patiences):
            pts = sorted((r for r in sub if r["strategy"].get("patience") == pat),
                         key=lambda r: r["strategy"]["threshold"])
            xs = [r["strategy"]["threshold"] for r in pts]
            ys = [score(r, m) for r in pts]
            label = metric_name(m) + (f" (patience={pat})" if len(patiences) > 1 else "")
            ax.plot(xs, ys, color=color, marker=PAT_MARKER[j % len(PAT_MARKER)],
                    linewidth=2, markersize=6, label=label)
            annotate(ax, xs, ys, color)

    for j, pat in enumerate(patiences):
        pts = sorted((r for r in sub if r["strategy"].get("patience") == pat),
                     key=lambda r: r["strategy"]["threshold"])
        ax2.plot([r["strategy"]["threshold"] for r in pts], [skipped(r) for r in pts],
                 color="gray", marker=PAT_MARKER[j % len(PAT_MARKER)], linestyle=":",
                 linewidth=2, markersize=6, markerfacecolor="white",
                 label="skipped layers" + (f" (patience={pat})" if len(patiences) > 1 else ""))

    ax.set_ylim(-5, 110)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Benchmark score, % (solid)\nbaseline = dashed")
    gmax = max((skipped(r) for r in sub), default=0) or 1.0
    ax2.set_ylim(-0.05 * gmax, 1.4 * gmax)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1 if gmax < 10 else 0))
    ax2.set_ylabel("Skipped layers, % of depth (dotted)\nhigher = more savings")
    style_x(ax, thresholds)
    ax.set_title(strat, fontsize=12, fontweight="bold")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8, framealpha=0.95)


def tradeoff_panel(ax, runs, base, metric, strategies):
    b = score(base, metric)
    ax.axhline(b, color="black", linestyle="--", linewidth=1.2)
    ax.scatter(0, b, marker="*", s=300, color="black", zorder=5, label=f"baseline ({fmt(b)})")
    for i, strat in enumerate(strategies):
        pts = sorted((r for r in runs if r["strategy"]["name"] == strat), key=skipped)
        ax.plot([skipped(r) for r in pts], [score(r, metric) for r in pts],
                color=STRAT_COLOR[i % len(STRAT_COLOR)], marker="o", linewidth=2,
                markersize=6, label=strat)
    ax.set_xlabel("Skipped layers, % of depth (higher = more savings)")
    ax.set_ylabel(f"{metric_name(metric)}, %")
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(-5, 110)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.95)
    ax.set_title("Score vs layer savings, all strategies\n(best: top right)",
                 fontsize=12, fontweight="bold")


# --- figure ------------------------------------------------------------------
def plot_benchmark(path) -> Path | None:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    ok = [r for r in data["runs"] if "results" in r]
    base = next((r for r in ok if r["strategy"]["name"] == BASELINE), None)
    runs = [r for r in ok if r is not base]
    if base is None or not runs:
        print(f"skipped {path.name}: baseline or strategies missing")
        return None

    metrics = score_metrics(base["results"])
    strategies = list(dict.fromkeys(r["strategy"]["name"] for r in runs))

    n_panels = len(strategies) + 1
    ncols = 2
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 5.5 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, strat in zip(axes, strategies):
        strategy_panel(ax, [r for r in runs if r["strategy"]["name"] == strat],
                       base, metrics, strat)
    tradeoff_panel(axes[len(strategies)], runs, base, metrics[0], strategies)
    for ax in axes[n_panels:]:
        ax.axis("off")

    limit = data["options"].get("limit")
    fig.suptitle(
        f"{data['model']} ({data['quant'] or 'bf16'}) | {data['task']} "
        f"({limit or 'all'} examples) | {data['n_layers']} layers | vs baseline (NoExitStrategy)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{path.stem}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_benchmarks.py <benchmark.json> [...]")
    for arg in sys.argv[1:]:
        # PowerShell ne développe pas les motifs (*.json) : on le fait ici
        paths = sorted(Path().glob(arg)) if any(c in arg for c in "*?[") else [Path(arg)]
        for p in paths:
            plot_benchmark(p)

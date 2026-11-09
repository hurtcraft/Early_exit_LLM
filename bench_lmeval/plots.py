"""
Early exit on lm_eval benchmarks: one figure per JSON file (model x task)
produced by main_benchmarks.py. Proportions (acc, exact_match…) are shown in %,
perplexities (WikiText) as raw values on a log scale (lower is better), cf.
bench_lmeval/metrics.py.

Panels:
  - one per strategy (dual axis):
      solid lines  (left axis)  : benchmark score(s) vs threshold, baseline = dashed line
      dotted line  (right axis) : skipped layers / total layers
  - trade-off: main score vs skipped layers, all strategies (baseline = star at 0%)
  - generation only (GSM8K…): main score vs relative compute (tokens generated x
    layers per token, baseline = 100%); right of 100% = costs MORE than the full model

Usage: python bench_lmeval/plots.py results/lmeval/<date>/<model>/<task>.json [...]
Output: plots/lmeval/<model>_<quant>_<task>[_patience<p>].png
        (one figure per patience value when the grid has several)
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # figures are only saved (also called during main_benchmarks runs)
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# lancé directement (python bench_lmeval/plots.py) : rendre le package importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench_lmeval.metrics import (  # noqa: E402  (score_metrics, metric_name : réexportés)
    higher_is_better, is_generative, is_rate, metric_name, primary_metric,
    relative_compute, score_metrics,
)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "lmeval"
BASELINE = "NoExitStrategy"

METRIC_COLOR = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]
STRAT_COLOR = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
PAT_MARKER = ["o", "s", "^", "D"]
PCT = PercentFormatter(xmax=100, decimals=0)
LABEL_BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)


# --- metrics -----------------------------------------------------------------
# proportions en %, perplexités en valeur brute
score = lambda r, m: (100 if is_rate(m) else 1) * r["results"].get(m, math.nan)
skipped = lambda r: 100 * (1 - r["results"]["early_exit"]["mean_depth_ratio"])


def fmt(v, metric=None):
    if metric is not None and not is_rate(metric):
        return f"{v:.3g}"
    return f"{v:.0f}%" if abs(v) >= 10 else f"{v:.1f}%"


def score_axis(ax, metric, label_suffix=""):
    """Axe Y d'un score : % de -5 à 110 pour une proportion, échelle log sinon."""
    if is_rate(metric):
        ax.set_ylim(-5, 110)
        ax.yaxis.set_major_formatter(PCT)
        ax.set_ylabel(f"Benchmark score, %{label_suffix}")
    else:
        ax.set_yscale("log")
        ax.set_ylabel(f"{metric_name(metric)}, log scale (lower is better){label_suffix}")


def annotate(ax, xs, ys, color, above=True, metric=None):
    for x, y in zip(xs, ys):
        ax.annotate(fmt(y, metric), (x, y), textcoords="offset points",
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
            annotate(ax, xs, ys, color, metric=m)

    for j, pat in enumerate(patiences):
        pts = sorted((r for r in sub if r["strategy"].get("patience") == pat),
                     key=lambda r: r["strategy"]["threshold"])
        ax2.plot([r["strategy"]["threshold"] for r in pts], [skipped(r) for r in pts],
                 color="gray", marker=PAT_MARKER[j % len(PAT_MARKER)], linestyle=":",
                 linewidth=2, markersize=6, markerfacecolor="white",
                 label="skipped layers" + (f" (patience={pat})" if len(patiences) > 1 else ""))

    score_axis(ax, metrics[0], " (solid)\nbaseline = dashed")
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
    ax.scatter(0, b, marker="*", s=300, color="black", zorder=5,
               label=f"baseline ({fmt(b, metric)})")
    for i, strat in enumerate(strategies):
        pts = sorted((r for r in runs if r["strategy"]["name"] == strat), key=skipped)
        ax.plot([skipped(r) for r in pts], [score(r, metric) for r in pts],
                color=STRAT_COLOR[i % len(STRAT_COLOR)], marker="o", linewidth=2,
                markersize=6, label=strat)
    ax.set_xlabel("Skipped layers, % of depth (higher = more savings)")
    ax.xaxis.set_major_formatter(PCT)
    score_axis(ax, metric)
    if is_rate(metric):
        ax.set_ylabel(f"{metric_name(metric)}, %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.95)
    best = "top right" if higher_is_better(metric) else "bottom right"
    ax.set_title(f"Score vs layer savings, all strategies\n(best: {best})",
                 fontsize=12, fontweight="bold")


def compute_panel(ax, runs, base, metric, strategies, n_layers):
    """Génération : score en fonction du calcul relatif (tokens x couches par token)."""
    be = base["results"]["early_exit"]
    b = score(base, metric)
    xmax = 100
    for i, strat in enumerate(strategies):
        pts = [r for r in runs if r["strategy"]["name"] == strat]
        xs = [relative_compute(r["results"]["early_exit"], be, n_layers) for r in pts]
        xmax = max([xmax] + xs)
        ax.scatter(xs, [score(r, metric) for r in pts], color=STRAT_COLOR[i % len(STRAT_COLOR)],
                   s=40, alpha=0.85, label=strat, zorder=3)
    xmax *= 1.08
    ax.axvspan(100, xmax, color="#f4cccc", alpha=0.4, linewidth=0,
               label="costs more than the full model")
    ax.axvline(100, color="black", linestyle="--", linewidth=1.2)
    ax.scatter(100, b, marker="*", s=300, color="black", zorder=5,
               label=f"baseline ({fmt(b, metric)})")
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("Relative compute vs baseline, % (tokens generated × layers per token)")
    score_axis(ax, metric)
    if is_rate(metric):
        ax.set_ylabel(f"{metric_name(metric)}, %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.95)
    ax.set_title("Score vs total compute (generation)\nright of 100% = costs MORE than the full model",
                 fontsize=12, fontweight="bold")


# --- figure ------------------------------------------------------------------
def _plot_figure(data, runs, base, out: Path, subtitle: str = "") -> Path:
    # métrique principale en tête ; les autres seulement si ce sont aussi des
    # proportions (même axe en %). Perplexités : seule la principale (échelles différentes).
    primary = primary_metric(data["task"], base["results"])
    if is_rate(primary):
        metrics = [primary] + [m for m in score_metrics(base["results"])
                               if m != primary and is_rate(m)]
    else:
        metrics = [primary]
    strategies = list(dict.fromkeys(r["strategy"]["name"] for r in runs))
    generative = is_generative([r["results"]["early_exit"] for r in runs],
                               base["results"]["early_exit"])

    n_panels = len(strategies) + 1 + generative
    ncols = 2
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 5.5 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, strat in zip(axes, strategies):
        strategy_panel(ax, [r for r in runs if r["strategy"]["name"] == strat],
                       base, metrics, strat)
    tradeoff_panel(axes[len(strategies)], runs, base, metrics[0], strategies)
    if generative:
        compute_panel(axes[len(strategies) + 1], runs, base, metrics[0], strategies,
                      data["n_layers"])
    for ax in axes[n_panels:]:
        ax.axis("off")

    limit = data["options"].get("limit")
    fig.suptitle(
        f"{data['model']} ({data['quant'] or 'bf16'}) | {data['task']} "
        f"({limit or 'all'} examples) | {data['n_layers']} layers{subtitle} "
        f"| vs baseline (NoExitStrategy)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)
    return out


def plot_benchmark(path, out_dir=None) -> list[Path]:
    """Une figure par valeur de patience (une seule si la grille n'en a qu'une)."""
    path = Path(path)
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    data = json.loads(path.read_text(encoding="utf-8"))

    ok = [r for r in data["runs"] if "results" in r]
    base = next((r for r in ok if r["strategy"]["name"] == BASELINE), None)
    runs = [r for r in ok if r is not base]
    if base is None or not runs:
        print(f"skipped {path.name}: baseline or strategies missing")
        return []

    stem = f"{data['model']}_{data['quant'] or 'bf16'}_{data['task']}"
    patiences = sorted({r["strategy"].get("patience") for r in runs}, key=str)
    if len(patiences) == 1:
        return [_plot_figure(data, runs, base, out_dir / f"{stem}.png")]

    return [
        _plot_figure(data, [r for r in runs if r["strategy"].get("patience") == pat], base,
                     out_dir / f"{stem}_patience{pat}.png", f" | patience={pat}")
        for pat in patiences
    ]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python bench_lmeval/plots.py <task.json> [...]")
    for arg in sys.argv[1:]:
        # PowerShell ne développe pas les motifs (*.json) : on le fait ici
        paths = sorted(Path().glob(arg)) if any(c in arg for c in "*?[") else [Path(arg)]
        for p in paths:
            plot_benchmark(p)

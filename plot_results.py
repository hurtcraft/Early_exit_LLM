"""
Early exit: one figure per strategy, compared to the baseline (NoExitStrategy).
All metrics are expressed as percentages.

Main panel (dual axis): quality vs layer savings
  - solid lines  (left axis)  : prefix match vs baseline            (baseline = 100%)
  - dotted lines (right axis) : skipped layers / total layers       (baseline = 0%)
Secondary panels:
  - exact match                                    (baseline = 100%)
  - mean exit layer as % of model depth, +/- std   (baseline = 100%)
  - early-exit rate                                (baseline = 0%)
  - generated tokens as % of baseline              (baseline = 100%)

The total number of layers is read from the baseline (mean_exit_layer).

Usage: python plot_results.py results/<model>_GRID_SUMMARY_xxx.json [out_dir]
Output: <out_dir or plots>/<model>_early_exit_<Strategy>.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # figures are only saved (also called at the end of main.py)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "plots"

PAT_COLOR = {2: "#ff7f0e", 3: "#1f77b4"}
PAT_MARKER = {2: "o", 3: "s"}
PCT = PercentFormatter(xmax=100, decimals=0)


def fmt(v):
    return f"{v:.0f}%" if abs(v) >= 10 else f"{v:.1f}%"


def series(sub, pat):
    pts = sorted((r for r in sub if r["params"]["patience"] == pat),
                 key=lambda r: r["params"]["threshold"])
    return pts, [r["params"]["threshold"] for r in pts]


LABEL_BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)


def annotate(ax, xs, ys, color, above, on=None):
    """Label each point above (or below) it, on a white box drawn over the markers.
    `on`: axes to draw the text on (the top one of a twinx pair), so it isn't hidden."""
    on = on or ax
    for x, y in zip(xs, ys):
        on.annotate(fmt(y), (x, y), xycoords=ax.transData, textcoords="offset points",
                    xytext=(0, 10 if above else -10), ha="center",
                    va="bottom" if above else "top", fontsize=8, color=color,
                    bbox=LABEL_BOX, zorder=10)


def style_x(ax, thresholds):
    ax.set_xlabel("threshold")
    ax.set_xticks(thresholds)
    # close thresholds (e.g. 0.08 / 0.10 / 0.12) overlap: tilt the labels
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    pad = 0.25 * ((max(thresholds) - min(thresholds)) or 1)
    ax.set_xlim(min(thresholds) - pad, max(thresholds) + pad)
    ax.grid(alpha=0.3)


def simple_panel(ax, sub, patiences, thresholds, title, get, bl, ylabel,
                 ylim=None, std=None):
    ax.axhline(bl, color="black", linestyle="--", linewidth=1.2)
    for pat in patiences:
        pts, xs = series(sub, pat)
        ys = [get(r) for r in pts]
        kw = dict(color=PAT_COLOR[pat], marker=PAT_MARKER[pat], linewidth=2, markersize=7)
        if std:
            ax.errorbar(xs, ys, yerr=[std(r) for r in pts], capsize=4, **kw)
        else:
            ax.plot(xs, ys, **kw)
        annotate(ax, xs, ys, PAT_COLOR[pat], above=pat == patiences[0])
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(PCT)
    style_x(ax, thresholds)
    if ylim:
        ax.set_ylim(*ylim)


def plot_grid_summary(path, out_dir=None) -> list:
    """One figure per strategy from a GRID_SUMMARY file. Returns the saved paths."""
    path = Path(path)
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # model name = filename prefix, e.g. "qwen3.5-2b_GRID_SUMMARY_xxx.json" -> "qwen3.5-2b"
    model = path.stem.split("_GRID_SUMMARY")[0]

    base = next(d for d in data if d["strategy"] == "NoExitStrategy")["summary"]
    runs = [d for d in data if "comparison" in d]
    strategies = list(dict.fromkeys(r["strategy"] for r in runs))

    n_layers = base["mean_exit_layer"]   # total depth
    n_tok_base = base["n_tokens"]

    # --- metrics in % --------------------------------------------------------
    prefix = lambda r: 100 * r["comparison"]["mean_prefix_match_ratio"]
    exact = lambda r: 100 * r["comparison"]["exact_match_rate"]
    skipped = lambda r: 100 * r["comparison"]["mean_skipped_layers"] / n_layers
    exit_depth = lambda r: 100 * r["summary"]["mean_exit_layer"] / n_layers
    exit_depth_std = lambda r: 100 * r["summary"]["std_exit_layer"] / n_layers
    early_rate = lambda r: 100 * r["summary"]["early_exit_rate"]
    tokens_pct = lambda r: 100 * r["summary"]["n_tokens"] / n_tok_base

    saved = []
    for strat in strategies:
        sub = [r for r in runs if r["strategy"] == strat]
        thresholds = sorted({r["params"]["threshold"] for r in sub})
        patiences = sorted({r["params"]["patience"] for r in sub})

        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(2, 3, figure=fig)

        # ---------- Main panel: quality (left) + savings (right) ----------
        ax = fig.add_subplot(gs[0, :2])
        ax2 = ax.twinx()
        gmax = max(skipped(r) for r in sub) or 1.0

        ax.axhline(100, color="black", linestyle="--", linewidth=1.2)
        for pat in patiences:
            pts, xs = series(sub, pat)
            c, m = PAT_COLOR[pat], PAT_MARKER[pat]
            q = [prefix(r) for r in pts]
            ax.plot(xs, q, color=c, marker=m, linewidth=2.2, markersize=8)
            g = [skipped(r) for r in pts]
            ax2.plot(xs, g, color=c, marker=m, linewidth=2, markersize=8,
                     linestyle=":", markerfacecolor="white", markeredgewidth=1.8)
            # labels on ax2 (drawn on top of ax): quality above, savings below
            annotate(ax, xs, q, c, above=True, on=ax2)
            annotate(ax2, xs, g, c, above=False)

        # aligned axes: 100% quality <-> max savings; zeros aligned
        ax.set_ylim(-10, 140)
        ax.set_yticks(range(0, 101, 20))
        ax2.set_ylim(-0.1 * gmax, 1.4 * gmax)
        ax.yaxis.set_major_formatter(PCT)
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1 if gmax < 10 else 0))
        ax.set_ylabel("Prefix match vs baseline, % (solid)\nhigher is better")
        ax2.set_ylabel("Skipped layers per token, % of depth (dotted)\nhigher = more savings")
        style_x(ax, thresholds)
        ax.set_title("Quality vs layer savings\n(quality: higher is better | savings: higher = more savings)",
                     fontsize=12, fontweight="bold")
        handles = [Line2D([], [], color=PAT_COLOR[p], marker=PAT_MARKER[p], linewidth=2,
                          label=f"patience = {p}") for p in patiences]
        handles += [Line2D([], [], color="gray", linewidth=2, label="prefix match (left axis)"),
                    Line2D([], [], color="gray", linewidth=2, linestyle=":",
                           label="% skipped layers (right axis)"),
                    Line2D([], [], color="black", linestyle="--", label="baseline (quality = 100%)")]
        ax.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, framealpha=0.95)

        # ---------- Secondary panels ----------
        simple_panel(fig.add_subplot(gs[0, 2]), sub, patiences, thresholds,
                     "Exact match vs baseline\n(higher is better)", exact, 100, "% identical outputs",
                     ylim=(-10, 115))
        simple_panel(fig.add_subplot(gs[1, 0]), sub, patiences, thresholds,
                     "Mean exit layer, % of depth (± std)\n(lower = more savings)", exit_depth,
                     100, "% of total depth", std=exit_depth_std)
        simple_panel(fig.add_subplot(gs[1, 1]), sub, patiences, thresholds,
                     "Early-exit rate\n(higher = more savings, if quality holds)", early_rate, 0, "% of tokens exiting early",
                     ylim=(-10, 115))
        simple_panel(fig.add_subplot(gs[1, 2]), sub, patiences, thresholds,
                     "Generated tokens, % of baseline\n(closer to 100% is better)", tokens_pct, 100,
                     "% of baseline n_tokens",
                     ylim=(0, max(100, *(tokens_pct(r) for r in sub)) * 1.2))

        fig.suptitle(f"{model} | {strat} vs baseline (NoExitStrategy, {int(n_layers)} layers)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        out = out_dir / f"{model}_early_exit_{strat}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print("saved", out)
        plt.close(fig)
        saved.append(out)
    return saved


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_results.py <GRID_SUMMARY.json> [out_dir]")
    plot_grid_summary(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
# """"
# Visualisation d'un fichier GRID_SUMMARY produit par ExperimentRunner.
# Chaque stratégie est traitée SÉPARÉMENT (ses seuils n'ont pas la même
# signification d'une stratégie à l'autre) : une figure + un tableau CSV par stratégie.
# La stratégie NoExitStrategy sert de BASELINE et est superposée à chaque figure.

# Usage :
#     python plot_grid_summary.py results/qwen3.5-2b_GRID_SUMMARY_xxx.json
# """
# import json
# import sys
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# BASELINE_NAME = "NoExitStrategy"
# BASE_COLOR = "tab:red"

# SUMMARY_METRICS = [
#     "n_questions", "n_tokens", "mean_exit_layer",
#     "std_exit_layer", "mean_depth_ratio", "early_exit_rate",
# ]
# COMPARISON_METRICS = [
#     "exact_match_rate", "mean_prefix_match_ratio", "mean_token_agreement",
#     "mean_similarity", "mean_first_divergence_idx", "mean_skipped_layers",
#     "mean_skipped_ratio", "mean_quality_adjusted_saving",
#     "divergence_per_skipped_ratio",
# ]
# METRICS = SUMMARY_METRICS + COMPARISON_METRICS
# # --------------------------------------------------------------------------- #
# # Chargement
# # --------------------------------------------------------------------------- #

# def load(path: Path) -> pd.DataFrame:
#     data = json.loads(path.read_text(encoding="utf-8"))
#     rows = [
#         {"strategy": d["strategy"], **d["params"], **d["summary"],
#          **d.get("comparison", {})}          # <-- ajout
#         for d in data
#         if "summary" in d
#     ]
#     return pd.DataFrame(rows)

# def get_baseline(df: pd.DataFrame):
#     """Métriques de la baseline (moyenne si plusieurs runs NoExit), ou None."""
#     sub = df[df["strategy"] == BASELINE_NAME]
#     if sub.empty:
#         return None
#     return sub[["mean_exit_layer", "std_exit_layer", "mean_depth_ratio",
#                 "early_exit_rate", "n_tokens"]].mean().to_dict()

# QUALITY_METRICS = [
#     ("exact_match_rate", "Exact match (sortie identique)"),
#     ("mean_token_agreement", "Accord token par token"),
#     ("mean_similarity", "Similarité moyenne"),
# ]


# def _quality_vs_skip(ax, g, var, metric, title, x="mean_skipped_layers",
#                      label_points=True):
#     """Qualité (y) en fonction du nombre de couches skippées (x).
#     La baseline est le point (0 couche skippée, qualité = 1.0) par définition."""
#     g = g.sort_values(x)
#     ax.plot(g[x], g[metric], marker="o", color="tab:blue", label="Stratégie")
#     if label_points:
#         for _, r in g.iterrows():
#             ax.annotate(make_label(r, var), (r[x], r[metric]),
#                         textcoords="offset points", xytext=(5, 5), fontsize=8)
#     # Baseline : aucune couche skippée, sortie identique à elle-même
#     ax.scatter(0, 1.0, marker="*", s=300, color=BASE_COLOR,
#                edgecolor="black", zorder=5, label="Baseline NoExit")
#     ax.set_xlabel("Couches skippées en moyenne" if x == "mean_skipped_layers"
#                   else "Ratio de couches skippées")
#     ax.set_ylabel(title)
#     ax.set_title(title)
#     ax.set_ylim(top=1.02)
#     ax.grid(alpha=0.3)
#     ax.legend(fontsize=8)


# def plot_quality(strat: str, sub: pd.DataFrame, out: Path):
#     if "mean_skipped_layers" not in sub.columns:
#         return
#     var = varying_params(sub)
#     fig, axes = plt.subplots(1, len(QUALITY_METRICS), figsize=(6 * len(QUALITY_METRICS), 5))
#     for ax, (m, t) in zip(axes, QUALITY_METRICS):
#         _quality_vs_skip(ax, sub, var, m, t)
#     fig.suptitle(f"{strat} : qualité vs couches skippées", fontsize=14)
#     fig.tight_layout()
#     fig.savefig(out / f"{strat}_quality.png", dpi=150)


# def plot_quality_global(df: pd.DataFrame, out: Path):
#     """Toutes les stratégies superposées : l'axe X (couches skippées) est
#     comparable entre stratégies, contrairement aux seuils."""
#     d = df[df["strategy"] != BASELINE_NAME].dropna(subset=["mean_skipped_layers"])
#     if d.empty:
#         return
#     fig, axes = plt.subplots(1, len(QUALITY_METRICS), figsize=(6 * len(QUALITY_METRICS), 5))
#     for ax, (m, t) in zip(axes, QUALITY_METRICS):
#         for strat, g in d.groupby("strategy"):
#             g = g.sort_values("mean_skipped_layers")
#             ax.plot(g["mean_skipped_layers"], g[m], marker="o", label=strat)
#         ax.scatter(0, 1.0, marker="*", s=300, color=BASE_COLOR,
#                    edgecolor="black", zorder=5, label="Baseline NoExit")
#         ax.set_xlabel("Couches skippées en moyenne")
#         ax.set_ylabel(t)
#         ax.set_title(t)
#         ax.grid(alpha=0.3)
#         ax.legend(fontsize=8)
#     fig.suptitle("Qualité vs couches skippées (toutes stratégies)", fontsize=14)
#     fig.tight_layout()
#     fig.savefig(out / "ALL_quality_vs_skipped.png", dpi=150)

# def strategy_params(sub: pd.DataFrame) -> list:
#     """Colonnes de paramètres réellement utilisées par cette stratégie."""
#     cols = [c for c in sub.columns if c not in METRICS + ["strategy"]]
#     return [c for c in cols if sub[c].notna().any()]


# def varying_params(sub: pd.DataFrame) -> list:
#     """Paramètres qui changent réellement entre les runs de cette stratégie."""
#     return [c for c in strategy_params(sub) if sub[c].nunique(dropna=True) > 1]


# def make_label(row, varying: list) -> str:
#     return ", ".join(f"{k}={row[k]}" for k in varying)


# # --------------------------------------------------------------------------- #
# # Panneaux
# # --------------------------------------------------------------------------- #
# def _heatmap(ax, g, a, b, metric, title, base):
#     pv = g.pivot_table(index=a, columns=b, values=metric)
#     im = ax.imshow(pv.values, cmap="viridis", aspect="auto")
#     ax.set_xticks(range(len(pv.columns)), labels=pv.columns)
#     ax.set_yticks(range(len(pv.index)), labels=pv.index)
#     ax.set_xlabel(b)
#     ax.set_ylabel(a)
#     if base:
#         title += f"\n(baseline NoExit : {base[metric]:.3f})"
#     ax.set_title(title)
#     for i in range(pv.shape[0]):
#         for j in range(pv.shape[1]):
#             v = pv.values[i, j]
#             if not np.isnan(v):
#                 ax.text(j, i, f"{v:.3f}", ha="center", va="center",
#                         color="white", fontsize=9)
#     plt.colorbar(im, ax=ax)


# def _effects(axes, g, var, base):
#     metrics = [("mean_depth_ratio", "Profondeur moyenne (ratio)"),
#                ("early_exit_rate", "Taux d'early exit")]
#     if len(var) >= 2:
#         for ax, (m, t) in zip(axes, metrics):
#             _heatmap(ax, g, var[0], var[1], m, t, base)
#     elif len(var) == 1:
#         s = g.sort_values(var[0])
#         for ax, (m, t) in zip(axes, metrics):
#             ax.plot(s[var[0]], s[m], marker="o", label="Stratégie")
#             if base:
#                 ax.axhline(base[m], color=BASE_COLOR, ls="--",
#                            label=f"Baseline NoExit ({base[m]:.3f})")
#             ax.set_xlabel(var[0])
#             ax.set_title(t)
#             ax.grid(alpha=0.3)
#             ax.legend(fontsize=8)
#     else:
#         for ax in axes:
#             ax.text(0.5, 0.5, "Un seul run : rien à comparer",
#                     ha="center", va="center")
#             ax.axis("off")


# def _tradeoff(ax, g, var, base):
#     ax.scatter(g["mean_depth_ratio"], g["early_exit_rate"],
#                s=80, color="tab:blue", edgecolor="black", label="Stratégie")
#     for _, r in g.iterrows():
#         ax.annotate(make_label(r, var),
#                     (r["mean_depth_ratio"], r["early_exit_rate"]),
#                     textcoords="offset points", xytext=(5, 5), fontsize=8)
#     if base:
#         ax.scatter(base["mean_depth_ratio"], base["early_exit_rate"],
#                    marker="*", s=300, color=BASE_COLOR, edgecolor="black",
#                    label="Baseline NoExit", zorder=5)
#         ax.legend(fontsize=8)
#     ax.set_xlabel("Profondeur moyenne (ratio, plus bas = plus rapide)")
#     ax.set_ylabel("Taux d'early exit (plus haut = plus rapide)")
#     ax.set_title("Compromis (meilleur : en haut à gauche)")
#     ax.grid(alpha=0.3)


# def _exit_layers(ax, g, var, base):
#     rows = [
#         {"label": make_label(r, var) or "run unique",
#          "mean": r["mean_exit_layer"], "std": r["std_exit_layer"],
#          "color": "tab:orange"}
#         for _, r in g.iterrows()
#     ]
#     if base:
#         rows.append({"label": "BASELINE (NoExit)", "mean": base["mean_exit_layer"],
#                      "std": base["std_exit_layer"], "color": BASE_COLOR})
#     d = pd.DataFrame(rows).sort_values("mean")

#     ax.barh(d["label"], d["mean"], xerr=d["std"], color=d["color"], capsize=3)

#     # Valeurs numériques directement sur le graphique : moyenne ± écart-type
#     max_x = (d["mean"] + d["std"]).max()
#     for i, (_, r) in enumerate(d.iterrows()):
#         ax.text(r["mean"] + r["std"] + max_x * 0.01, i,
#                 f"{r['mean']:.2f} ± {r['std']:.2f}",
#                 va="center", ha="left", fontsize=8)
#     ax.set_xlim(0, max_x * 1.25)  # marge pour que le texte ne soit pas coupé

#     ax.set_xlabel("Couche de sortie moyenne (± écart-type)")
#     ax.set_title("Couche de sortie par configuration")
#     ax.grid(axis="x", alpha=0.3)


# # --------------------------------------------------------------------------- #
# # Une figure par stratégie
# # --------------------------------------------------------------------------- #
# def plot_strategy(strat: str, sub: pd.DataFrame, out: Path, base):
#     var_all = varying_params(sub)
#     extra = var_all[2:]  # au-delà de 2 paramètres : une figure par combinaison

#     if extra:
#         groups = [
#             (dict(zip(extra, key if isinstance(key, tuple) else (key,))), g)
#             for key, g in sub.groupby(extra)
#         ]
#     else:
#         groups = [({}, sub)]

#     for fixed, g in groups:
#         var = varying_params(g)
#         fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#         _effects([axes[0, 0], axes[0, 1]], g, var, base)
#         _tradeoff(axes[1, 0], g, var, base)
#         _exit_layers(axes[1, 1], g, var, base)

#         suffix = "_".join(f"{k}{v}" for k, v in fixed.items())
#         title = strat + (f"  ({', '.join(f'{k}={v}' for k, v in fixed.items())})"
#                          if fixed else "")
#         fig.suptitle(title, fontsize=14)
#         fig.tight_layout()
#         fig.savefig(out / f"{strat}{'_' + suffix if suffix else ''}.png", dpi=150)


# def save_table(strat: str, sub: pd.DataFrame, out: Path, base):
#     # cols = strategy_params(sub) + [
#     #     "mean_exit_layer", "std_exit_layer", "mean_depth_ratio",
#     #     "early_exit_rate", "n_tokens",
#     # ]
#     cols = strategy_params(sub) + [
#         c for c in SUMMARY_METRICS + COMPARISON_METRICS
#         if c in sub.columns and c != "n_questions"
#     ]
#     table = sub[cols].sort_values("mean_depth_ratio").reset_index(drop=True)

#     if base:
#         # Gain par rapport à la baseline
#         table["couches_economisees"] = base["mean_exit_layer"] - table["mean_exit_layer"]
#         table["gain_profondeur_pct"] = (
#             1 - table["mean_depth_ratio"] / base["mean_depth_ratio"]
#         ) * 100

#     table.to_csv(out / f"{strat}.csv", index=False)
#     print(f"\n===== {strat} =====")
#     print(table.round(4).to_string(index=False))


# # --------------------------------------------------------------------------- #
# def main():
#     if len(sys.argv) < 2:
#         sys.exit("Usage : python plot_grid_summary.py <GRID_SUMMARY.json>")

#     path = Path(sys.argv[1])
#     out = path.parent / "plots"
#     out.mkdir(exist_ok=True)

#     df = load(path)
#     base = get_baseline(df)
#     if base:
#         print(f"Baseline {BASELINE_NAME} : couche moyenne = {base['mean_exit_layer']:.2f}, "
#               f"profondeur = {base['mean_depth_ratio']:.3f}")
#     else:
#         print(f"Attention : aucune {BASELINE_NAME} dans le fichier, pas de baseline.")

#     for strat, sub in df.groupby("strategy"):
#         if strat == BASELINE_NAME:
#             continue  # la baseline est superposée aux autres, pas tracée seule
#         sub = sub.dropna(axis=1, how="all")  # retire les colonnes d'autres stratégies
#         save_table(strat, sub, out, base)
#         plot_strategy(strat, sub, out, base)
#         plot_quality(strat, sub, out)          # <-- ajout

#     plot_quality_global(df, out)   
#     print(f"\nFigures et tableaux sauvegardés dans : {out}")
#     plt.show()


# main()
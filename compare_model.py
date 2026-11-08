"""
Compare les résultats lm_eval d'early exit entre plusieurs modèles, pour voir
comment le nombre de couches influe sur la qualité de l'early exit.

Attention : dans nos runs, nombre de couches et taille du modèle sont confondus
(2B = 24 couches, 4B et 9B = 32 couches). Chaque modèle est donc étiqueté avec
son nombre de couches, ex. "qwen3.5-9b (32 L)".

Les scores sont comparés en écart à la baseline de chaque modèle (Δ), car les
baselines diffèrent beaucoup d'un modèle à l'autre. Δ est toujours « plus haut =
mieux, 0 = baseline » :
  - proportions (acc, exact_match…) : écart en points de % ;
  - perplexités (WikiText, plus bas = mieux) : variation relative en %, signe inversé
    (perplexité +10 % -> Δ = -10).
Métrique principale par tâche : cf. bench_lmeval/metrics.py (acc_norm pour
HellaSwag, word_perplexity pour WikiText).

Figures produites, par tâche, dans <out>/<task>/ :
  <task>_savings_curve.png       « si je veux sauter au moins X % des couches, quel
                                 meilleur score je garde ? » : une ligne par modèle,
                                 meilleure config toutes stratégies confondues
                                 (le tableau correspondant est affiché en console)
  <task>_<Strategy>_by_threshold.png
                                 ligne 1 : Δscore vs seuil ; ligne 2 : profondeur de
                                 sortie moyenne vs seuil ; une colonne par patience

Et une figure commune à toutes les tâches, dans <out>/ :
  size_effect.png                « plus un modèle est gros, meilleur est l'early exit ? » :
                                 score réel, baseline de chaque modèle en premier groupe,
                                 puis tranches disjointes de couches sautées (10–20 %,
                                 20–30 %, 30–40 %), une barre par
                                 modèle du plus petit au plus gros (paramètres lus dans le
                                 nom, ex. '-9b'), un panneau par tâche

Usage :
  python compare_model.py [dossiers ou fichiers json ...] [--ymin -30] [--out DIR]
  Sans chemin : lit  "final results/final_results/lmeval"  (sur le Bureau)
                écrit "final results/final_plots/compare/<task>/"

Si un même couple (modèle, tâche) apparaît plusieurs fois, on garde le run qui a
le plus de configurations, puis le plus récent (chemin le plus grand : les
dossiers sont horodatés).
"""
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # figures uniquement sauvegardées
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from bench_lmeval.metrics import is_rate, metric_name, primary_metric  # noqa: E402
from bench_lmeval.plots import BASELINE  # noqa: E402

FINAL_DIR = ROOT.parent / "final results"   # dossier voisin du projet, sur le Bureau
DEFAULT_IN_DIR = FINAL_DIR / "final_results" / "lmeval"
DEFAULT_OUT_DIR = FINAL_DIR / "final_plots" / "compare"
MODEL_COLOR = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
PCT = PercentFormatter(xmax=100, decimals=0)
CURVE_MAX_TARGET = 50   # courbe d'économie : objectifs 0 %, 5 %, ..., 50 %
CURVE_STEP = 5
CURVE_YMIN = -20        # borne basse par défaut de la courbe (au-delà, résultat inutilisable)
SIZE_BINS = ((10, 20), (20, 30), (30, 40))   # figure taille : tranches d'économie [lo, hi[ en %


# --- chargement --------------------------------------------------------------
def collect_files(args_paths) -> list[Path]:
    files = []
    for arg in args_paths:
        p = Path(arg)
        if p.is_dir():
            files += sorted(p.rglob("*.json"))
        elif any(c in arg for c in "*?["):   # PowerShell ne développe pas les motifs
            files += sorted(Path().glob(arg))
        else:
            files.append(p)
    return files


def quality_delta(rate: bool, base: float, value: float) -> float:
    """Écart à la baseline, toujours « plus haut = mieux » :
    proportion : écart en points de % ; perplexité : variation relative en %,
    signe inversé (perplexité +10 % -> -10)."""
    return value - base if rate else 100 * (base - value) / base


def make_entry(path: Path, data: dict) -> dict | None:
    """Un couple (modèle, tâche) : baseline + un point par configuration."""
    ok = [r for r in data.get("runs", []) if "results" in r]
    base = next((r for r in ok if r["strategy"]["name"] == BASELINE), None)
    runs = [r for r in ok if r is not base]
    if base is None or not runs:
        return None

    metric = primary_metric(data["task"], base["results"])
    rate = is_rate(metric)
    scale = 100 if rate else 1          # proportions en %, perplexités brutes
    base_score = scale * base["results"][metric]
    n = data["n_layers"]

    points = []
    for r in runs:
        res = r["results"]
        ee = res.get("early_exit")
        if ee is None or metric not in res:
            continue
        s = r["strategy"]
        points.append(dict(
            strategy=s["name"],
            threshold=s.get("threshold"),
            patience=s.get("patience"),
            delta=quality_delta(rate, base_score, scale * res[metric]),
            # mean_depth_ratio = couche de sortie moyenne / (n - 1), couches indexées depuis 0
            skip_pct=100 * (1 - ee["mean_depth_ratio"]),
            skip_layers=(n - 1) - ee["mean_exit_layer"],
            exit_depth=100 * ee["mean_depth_ratio"],
        ))
    points = [p for p in points if not math.isnan(p["delta"])]
    if not points:
        return None

    return dict(
        model=data["model"], n_layers=n, metric=metric, rate=rate, base_score=base_score,
        label=f"{data['model']} ({n} L)", points=points, path=path, n_runs=len(runs),
    )


def load(files: list[Path]) -> dict[str, list[dict]]:
    selected = {}
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            print(f"ignoré {f} : {exc}")
            continue
        if not isinstance(data, dict) or not {"runs", "task", "model", "n_layers"} <= data.keys():
            continue
        entry = make_entry(f, data)
        if entry is None:
            print(f"ignoré {f} : baseline ou stratégies manquantes")
            continue

        key = (data["task"], data["model"], data.get("quant"))
        rank = (entry["n_runs"], str(f))
        if key not in selected or rank > selected[key][0]:
            selected[key] = (rank, entry)

    by_task = defaultdict(list)
    for (task, _, _), (_, entry) in selected.items():
        by_task[task].append(entry)
    for entries in by_task.values():
        entries.sort(key=lambda e: (e["n_layers"], e["model"]))
        for i, e in enumerate(entries):
            e["color"] = MODEL_COLOR[i % len(MODEL_COLOR)]
    return dict(by_task)


# --- utilitaires -------------------------------------------------------------
def strategies_of(entries) -> list[str]:
    return list(dict.fromkeys(p["strategy"] for e in entries for p in e["points"]))


def clip(y, ymin):
    return y if ymin is None else max(y, ymin)


def model_params(model: str) -> float | None:
    """Nombre de paramètres (en milliards) lu dans le nom : 'qwen3.5-9b' -> 9.0."""
    m = re.search(r"-(\d+(?:\.\d+)?)b(?![a-z0-9])", model.lower())
    return float(m.group(1)) if m else None


def cfg(p) -> str:
    parts = []
    if p["threshold"] is not None:
        parts.append(f"t={p['threshold']:g}")
    if p["patience"] is not None:
        parts.append(f"p={p['patience']}")
    return " ".join(parts)


def fmt_score(e, v, short=False) -> str:
    """Score dans son unité : '59.4%' pour une proportion, '12.3' pour une perplexité."""
    if e["rate"]:
        return f"{v:.0f}%" if short else f"{v:.1f}%"
    return f"{v:.3g}"


def score_of(e, p) -> float:
    """Score réel de la config p (inverse de quality_delta)."""
    if e["rate"]:
        return e["base_score"] + p["delta"]
    return e["base_score"] * (1 - p["delta"] / 100)


def delta_label(metric) -> str:
    unit = "points" if is_rate(metric) else "% relative change, − = worse"
    return f"Δ {metric_name(metric)} vs baseline ({unit})"


def delta_axis(ax, metric, ymin):
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_ylabel(delta_label(metric))
    if ymin is not None:
        ax.set_ylim(bottom=ymin - 2)
    ax.grid(alpha=0.3)


def save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def clip_note(ymin) -> str:
    return "" if ymin is None else f" | Δ < {ymin:g} drawn at the bottom edge"


# --- figure 1 : courbe d'économie --------------------------------------------
def best_at_target(points, target):
    """Meilleure config (toutes stratégies) qui saute au moins `target` % des couches.
    À 0 %, la baseline elle-même (Δ = 0) est une option."""
    ok = [p for p in points if p["skip_pct"] >= target]
    best = max(ok, key=lambda p: p["delta"], default=None)
    if target == 0 and (best is None or best["delta"] < 0):
        return dict(delta=0.0, strategy="baseline", threshold=None, patience=None)
    return best


def plot_savings_curve(task, entries, out, ymin):
    """Si je veux économiser au moins X % des couches, quel meilleur score puis-je garder ?"""
    metric = entries[0]["metric"]
    targets = list(range(0, CURVE_MAX_TARGET + 1, CURVE_STEP))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    offsets = [8, -14, 20, -26, 32, -38]   # décale les étiquettes de chaque modèle

    print(f"\n--- {task} : meilleur Δ en économisant au moins X % des couches ---")
    print("  " + f"{'X':>4} | " + " | ".join(f"{e['label']:<30}" for e in entries))
    rows = {t: [] for t in targets}

    for i, e in enumerate(entries):
        xs, ys = [], []
        for t in targets:
            b = best_at_target(e["points"], t)
            rows[t].append("—" if b is None else
                           f"{b['delta']:+6.1f}  {b['strategy'].replace('Strategy', '')} {cfg(b)}")
            if b is not None:
                xs.append(t)
                ys.append(b["delta"])

        ax.plot(xs, [clip(y, ymin) for y in ys], color=e["color"], marker="o",
                linewidth=2.5, markersize=7,
                label=f"{e['label']}, baseline {fmt_score(e, e['base_score'])}")
        dy = offsets[i % len(offsets)]
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:+.1f}", (x, clip(y, ymin)), textcoords="offset points",
                        xytext=(0, dy), ha="center", va="center", fontsize=8,
                        color=e["color"], fontweight="bold")

    for t in targets:
        print("  " + f"{t:>3}% | " + " | ".join(f"{r:<30}" for r in rows[t]))

    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.text(targets[-1], 0, " baseline", va="bottom", ha="right", fontsize=9)
    if ymin is not None:
        ax.set_ylim(bottom=ymin - 3)
    ax.set_xticks(targets)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("Layer saving target: skip AT LEAST this % of the layers")
    ax.set_ylabel(f"Best {delta_label(metric)}\nany strategy, any config")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.95, loc="lower left")
    note = "" if ymin is None else f"\nvalues below {ymin:g} are drawn at the bottom edge (label = true value)"
    ax.set_title(f"{task} | best score kept for a given layer saving{note}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, out)


# --- figure 2 : effet de la taille du modèle (toutes tâches) -----------------
def size_key(e):
    """Tri du plus petit au plus gros modèle (paramètres lus dans le nom, sinon couches)."""
    p = model_params(e["model"])
    return (p if p is not None else math.inf, e["n_layers"], e["model"])


def size_label(e) -> str:
    p = model_params(e["model"])
    return f"{p:g}B ({e['n_layers']} L)" if p is not None else e["label"]


def best_in_bin(points, lo, hi):
    """Meilleure config (toutes stratégies) qui saute entre lo % (inclus) et hi % (exclu)
    des couches. Tranches disjointes : une config n'apparaît que dans une seule."""
    ok = [p for p in points if lo <= p["skip_pct"] < hi]
    return max(ok, key=lambda p: p["delta"], default=None)


def kept_pct(e, b) -> float:
    """Part de la qualité de la baseline conservée par la config b : 100 % = aussi bon
    que la baseline, moins = moins bien (perplexité : baseline / score)."""
    s = score_of(e, b)
    return 100 * s / e["base_score"] if e["rate"] else 100 * e["base_score"] / s


def plot_size_effect(by_task, out):
    """Un modèle plus gros supporte-t-il mieux l'early exit ?
    Un panneau par tâche ; premier groupe = baseline de chaque modèle, puis un groupe par
    tranche de couches sautées ; une barre par modèle, du plus petit au plus gros.
    Barre = score réel. La part de la baseline conservée est affichée en console."""
    tasks = sorted(by_task)
    fig, axes = plt.subplots(1, len(tasks), figsize=(8 * len(tasks), 6.8), squeeze=False)
    legend = {}   # label -> handle, communs à toutes les tâches

    print("\n--- score par tranche de couches sautées (meilleure config, toutes stratégies) ---")
    for ax, task in zip(axes[0], tasks):
        entries = sorted(by_task[task], key=size_key)
        width = 0.8 / len(entries)
        rate = entries[0]["rate"]
        values = []   # hauteurs des barres, pour régler l'axe
        print(f"  {task} | baselines : " +
              ", ".join(f"{size_label(e)} {fmt_score(e, e['base_score'])}" for e in entries))
        for lo, hi in SIZE_BINS:
            cells = []
            for e in entries:
                b = best_in_bin(e["points"], lo, hi)
                cells.append(f"{size_label(e)} " + ("aucune config" if b is None else
                             f"{fmt_score(e, score_of(e, b))} = {kept_pct(e, b):.0f}% de la baseline "
                             f"({b['strategy'].replace('Strategy', '')} {cfg(b)}, "
                             f"{b['skip_pct']:.1f}% sautées)"))
            print(f"    {lo}-{hi}% : " + " | ".join(cells))

        # groupe 0 = baseline (toutes les couches), groupes 1.. = tranches d'économie
        groups = [None] + list(SIZE_BINS)
        for i, e in enumerate(entries):
            base = e["base_score"]
            for k, grp in enumerate(groups):
                x = k - 0.4 + width * (i + 0.5)
                if grp is None:
                    # baseline : barre claire hachurée, bordure à la couleur du modèle
                    ax.bar(x, base, width * 0.9, facecolor=e["color"], alpha=0.35,
                           edgecolor="none")
                    ax.bar(x, base, width * 0.9, facecolor="none", edgecolor=e["color"],
                           hatch="//", linewidth=1.5)
                    legend.setdefault(size_label(e), Patch(facecolor=e["color"]))
                    values.append(base)
                    ax.annotate(fmt_score(e, base), (x, base), textcoords="offset points",
                                xytext=(0, 3), ha="center", va="bottom", fontsize=9,
                                fontweight="bold")
                    continue

                b = best_in_bin(e["points"], *grp)
                if b is None:   # aucune config dans cette tranche : tiret au pied de la barre
                    ax.annotate("–", (x, 0), xycoords=("data", "axes fraction"),
                                textcoords="offset points", xytext=(0, 3),
                                ha="center", va="bottom", fontsize=10, color="gray")
                    continue
                score = score_of(e, b)
                values.append(score)
                ax.bar(x, score, width * 0.9, color=e["color"])
                ax.annotate(fmt_score(e, score, short=True), (x, score),
                            textcoords="offset points", xytext=(0, 3), ha="center",
                            va="bottom", fontsize=9)

        ax.axvspan(-0.5, 0.5, color="#eeeeee", zorder=0)   # fond gris derrière la baseline
        ax.axvline(0.5, color="gray", linewidth=1)          # sépare la baseline des tranches
        ax.set_xlim(-0.5, len(groups) - 0.5)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(["baseline\n(all layers)"] +
                           [f"{lo}–{hi}%\nskipped" for lo, hi in SIZE_BINS])
        ax.get_xticklabels()[0].set_fontweight("bold")
        name = metric_name(entries[0]["metric"])
        if rate:
            ax.set_ylim(0, max(values, default=100) * 1.12)
            ax.yaxis.set_major_formatter(PCT)
            ax.set_ylabel(f"{name}, %")
        else:
            # perplexité : échelle log (les configs cassées montent à des milliers)
            ax.set_yscale("log")
            ax.set_ylim(min(values) * 0.5, max(values) * 2.5)
            ax.set_ylabel(f"{name}, log scale")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(task if rate else f"{task} (lower is better)", fontsize=13, fontweight="bold")

    fig.legend(list(legend.values()), list(legend.keys()), loc="lower center",
               ncol=len(legend), fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Does a bigger model tolerate early exit better?", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.905, "best score for each range of skipped layers (any strategy)",
             ha="center", fontsize=10, color="dimgray")
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    save(fig, out)


# --- figure 3 : sensibilité au seuil, par stratégie --------------------------
def plot_by_threshold(task, entries, strategy, out, ymin):
    sel = lambda e, pat: sorted(
        (p for p in e["points"]
         if p["strategy"] == strategy and p["patience"] == pat and p["threshold"] is not None),
        key=lambda p: p["threshold"])
    patiences = sorted({p["patience"] for e in entries for p in e["points"]
                        if p["strategy"] == strategy and p["threshold"] is not None}, key=str)
    if not patiences:
        return

    ncols = len(patiences)
    fig, axes = plt.subplots(2, ncols, figsize=(6.5 * ncols, 9.5), squeeze=False,
                             sharex=True, sharey="row")
    for j, pat in enumerate(patiences):
        top, bottom = axes[0][j], axes[1][j]
        for e in entries:
            pts = sel(e, pat)
            if not pts:
                continue
            ths = [p["threshold"] for p in pts]
            top.plot(ths, [clip(p["delta"], ymin) for p in pts], color=e["color"],
                     marker="o", linewidth=2, markersize=5, label=e["label"])
            bottom.plot(ths, [p["exit_depth"] for p in pts], color=e["color"],
                        marker="o", linewidth=2, markersize=5, label=e["label"])

        top.set_title(f"patience={pat}", fontsize=12, fontweight="bold")
        delta_axis(top, entries[0]["metric"], ymin)
        bottom.set_ylim(0, 105)
        bottom.yaxis.set_major_formatter(PCT)
        bottom.set_ylabel("Mean exit depth, % of layers")
        bottom.set_xlabel("threshold")
        bottom.grid(alpha=0.3)
        if j > 0:
            top.set_ylabel("")
            bottom.set_ylabel("")
        top.legend(fontsize=8, framealpha=0.95, loc="lower left")

    fig.suptitle(f"{task} | {strategy} | top: Δ score vs threshold, "
                 f"bottom: where the model exits{clip_note(ymin)}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out)


# --- résumé console ----------------------------------------------------------
def print_summary(task, entries):
    print(f"\n=== {task} ({metric_name(entries[0]['metric'])}) ===")
    for e in entries:
        print(f"  {e['label']:<24} baseline {fmt_score(e, e['base_score']):>6} "
              f"| {len(e['points'])} configs | source : {e['path']}")


# --- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=[str(DEFAULT_IN_DIR)],
                    help=f"dossiers (recherche récursive) ou fichiers .json (défaut : {DEFAULT_IN_DIR})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="dossier de sortie")
    ap.add_argument("--ymin", type=float, default=None,
                    help=f"borne basse de l'axe Δ ; les points en dessous sont dessinés au bord "
                         f"(défaut : {CURVE_YMIN} pour les courbes d'économie et de paramètres, "
                         f"aucune pour les figures par seuil)")
    args = ap.parse_args()

    by_task = load(collect_files(args.paths))
    if not by_task:
        sys.exit(f"Aucun résultat lm_eval exploitable trouvé dans : {', '.join(args.paths)}")

    for task, entries in sorted(by_task.items()):
        out = args.out / task
        print_summary(task, entries)
        if len(entries) < 2:
            print(f"  (un seul modèle pour {task} : figures générées quand même)")
        curve_ymin = args.ymin if args.ymin is not None else CURVE_YMIN
        plot_savings_curve(task, entries, out / f"{task}_savings_curve.png", curve_ymin)
        for strat in strategies_of(entries):
            plot_by_threshold(task, entries, strat, out / f"{task}_{strat}_by_threshold.png",
                              args.ymin)

    plot_size_effect(by_task, args.out / "size_effect.png")



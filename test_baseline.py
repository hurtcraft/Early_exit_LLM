"""Tests de validité de l'évaluation lm_eval (MMLU par défaut).

1) Baseline SANS le mécanisme d'early exit : HFLM standard de lm_eval
   (forward HuggingFace normal, avec KV cache).
2) NoExitStrategy passée par notre boucle couche par couche : doit donner la même
   précision que 1) à quelques questions près (bruit bf16). Sinon, la boucle de
   early_exit/layers.py ne reproduit pas fidèlement le modèle.
3) Précision couche par couche : AlwaysExitStrategy(layer=k) force la sortie à la
   couche k. Montre à partir de quelle couche la réponse est déjà correcte, et donc
   si une précision inchangée malgré beaucoup de couches sautées est plausible.

Sorties :
    results/tests/<date>_<modèle>_<tâche>_layers.json
    plots/tests/<date>_<modèle>_<tâche>_layers.png

Usage : python test_baseline.py
"""
import os
os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
os.environ["SYCL_PI_LEVEL_ZERO_USE_RELAXED_ALLOCATION_LIMITS"] = "1"

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from bench_lmeval.evaluate import evaluate_baseline, evaluate_early_exit
from early_exit.layers import CpuTraceCache
from early_exit.models.factory import ModelFactory
from early_exit.strategies.AlwaysExitStrategy import AlwaysExitStrategy
from early_exit.strategies.NoExitStrategy import NoExitStrategy
from early_exit.utils import free_memory, model_tag, timestamp

MODEL = ("qwen3.5-2b", None)
TASK = "mmlu"
LIMIT = 10                 # questions par matière (MMLU : 57 matières)

# format du prompt : identique à run_lmeval.py (chat template, 0-shot)
#FORMAT = {"apply_chat_template": True, "num_fewshot": None}
# protocole MMLU standard pour comparaison : sans chat template, 5-shot
FORMAT = {"apply_chat_template": False, "num_fewshot": 5}

RUN_HFLM = True            # 1) baseline HuggingFace standard
RUN_LAYERS = True          # 3) précision couche par couche
LAYERS = None              # None = toutes les couches ; sinon ex. [8, 12, 16, 18, 20, 22]

CHANCE = 0.25              # QCM à 4 choix

STAMP = timestamp()
RESULTS_DIR = Path("results") / "tests"
PLOTS_DIR = Path("plots") / "tests"


def acc_of(results: dict) -> tuple[float, float]:
    return results.get("acc,none", float("nan")), results.get("acc_stderr,none", float("nan"))


def plot_layers(rows: list, hflm_acc, noexit_acc, n_layers: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    xs = [r["layer"] for r in rows]
    ys = [100 * r["acc"] for r in rows]
    errs = [100 * r["stderr"] for r in rows]

    ax.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2, capsize=3,
                color="#1f77b4", label="sortie forcée à la couche k (± stderr)")
    ax.axhline(100 * noexit_acc, color="black", linestyle="--", linewidth=1.2,
               label=f"NoExit, notre boucle ({100 * noexit_acc:.1f}%)")
    if hflm_acc is not None:
        ax.axhline(100 * hflm_acc, color="#2ca02c", linestyle=":", linewidth=1.5,
                   label=f"HFLM standard ({100 * hflm_acc:.1f}%)")
    ax.axhline(100 * CHANCE, color="#d62728", linestyle="-.", linewidth=1,
               label=f"hasard ({100 * CHANCE:.0f}%)")

    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7)

    ax.set_xlabel(f"couche de sortie (0 à {n_layers - 1})")
    ax.set_ylabel(f"précision {TASK}")
    ax.set_xticks(range(n_layers))
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"{model_tag(*MODEL)} | {TASK} (limit={LIMIT}, {FORMAT}) | "
                 f"précision selon la couche de sortie", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


model_name, quant = MODEL
adapter = ModelFactory.load_class(name=model_name, quant=quant)
n_layers = adapter.num_layers
device = adapter.model.get_input_embeddings().weight.device
summary = {"model": model_name, "quant": quant, "task": TASK, "limit": LIMIT,
           "format": FORMAT, "n_layers": n_layers}

# Ordre d'exécution : 2) et 3) d'abord, 1) en dernier. Lancé en premier, le HFLM
# laisse de la mémoire GPU que empty_cache() ne libère pas, et notre boucle
# (sans KV cache) sature ensuite le XPU sur les longs prompts 5-shot.

# ---------- 2) NoExit via notre boucle (remplit le cache de traces) ----------
trace_cache = CpuTraceCache(device)
noexit_acc, noexit_err = acc_of(evaluate_early_exit(
    adapter, NoExitStrategy(), task=TASK, limit=LIMIT, trace_cache=trace_cache, **FORMAT))
summary["noexit"] = {"acc": noexit_acc, "stderr": noexit_err}
free_memory()

# ---------- 3) Précision couche par couche (rejoue les traces en cache) ----------
rows = []
if RUN_LAYERS:
    for layer in (LAYERS if LAYERS is not None else range(n_layers - 1)):
        res = evaluate_early_exit(adapter, AlwaysExitStrategy(layer=layer), task=TASK,
                                  limit=LIMIT, trace_cache=trace_cache, **FORMAT)
        acc, err = acc_of(res)
        rows.append({"layer": layer, "acc": acc, "stderr": err})
        free_memory()
    # la dernière couche = NoExit
    rows.append({"layer": n_layers - 1, "acc": noexit_acc, "stderr": noexit_err})
    summary["layers"] = rows

# ---------- 1) Baseline HFLM standard (en dernier, cf. ci-dessus) ----------
hflm_acc = None
if RUN_HFLM:
    del trace_cache   # plus utile : libère la RAM avant le HFLM
    free_memory()
    hflm_acc, hflm_err = acc_of(evaluate_baseline(adapter, task=TASK, limit=LIMIT, **FORMAT))
    summary["hflm"] = {"acc": hflm_acc, "stderr": hflm_err}

# ---------- Résumé ----------
print("\n" + "=" * 60)
print(f"{model_tag(model_name, quant)} | {TASK} | limit={LIMIT} | {FORMAT}")
if hflm_acc is not None:
    print(f"1) HFLM standard (sans early exit) : acc = {hflm_acc:.4f} (± {hflm_err:.4f})")
print(f"2) NoExit, notre boucle            : acc = {noexit_acc:.4f} (± {noexit_err:.4f})")
if hflm_acc is not None:
    print(f"   écart HFLM - NoExit             : {100 * (hflm_acc - noexit_acc):+.2f} pts")
if rows:
    print("3) précision en sortant à la couche k :")
    for r in rows:
        bar = "#" * round(40 * r["acc"])
        print(f"   L{r['layer']:02d} | {100 * r['acc']:5.1f}% "
              f"({100 * (r['acc'] - noexit_acc):+5.1f} vs NoExit) | {bar}")
print("=" * 60)

name = f"{STAMP}_{model_tag(model_name, quant)}_{TASK}_layers"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(RESULTS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
print(f"Résultats : {RESULTS_DIR / f'{name}.json'}")

if rows:
    plot_layers(rows, hflm_acc, noexit_acc, n_layers, PLOTS_DIR / f"{name}.png")
    print(f"Plot      : {PLOTS_DIR / f'{name}.png'}")

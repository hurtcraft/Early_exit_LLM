"""Benchmark lm_eval (MMLU, GSM8K…) des stratégies d'early exit.

Pour chaque modèle, chaque benchmark, chaque stratégie : score du benchmark
+ statistiques d'early exit. NoExitStrategy passe en premier (baseline) et remplit
le cache de traces que les autres stratégies réutilisent.

Sorties d'un lancement :
    results/lmeval/<date>/<modèle>/<tâche>.json
    plots/lmeval/<date>/<modèle>_<tâche>.png
"""
import os
os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
os.environ["SYCL_PI_LEVEL_ZERO_USE_RELAXED_ALLOCATION_LIMITS"] = "1"

import json
import traceback
from pathlib import Path

from tqdm import tqdm

from bench_lmeval.evaluate import evaluate_early_exit
from bench_lmeval.plots import plot_benchmark
from config import MODELS, STRATEGY_GRID
from early_exit.layers import CpuTraceCache
from early_exit.models.factory import ModelFactory
from early_exit.strategies.NoExitStrategy import NoExitStrategy
from early_exit.utils import expand_grid, free_memory, model_tag, strategy_params, timestamp

# --------------------------------------------------
# Configuration
# --------------------------------------------------
# tâche lm_eval -> options (limit = nb d'exemples par tâche, None = complet)
# GSM8K sans KV cache est très lent : garder une petite limite.
BENCHMARKS = {
    #"mmlu": {"limit": 5, "num_fewshot": 5, "apply_chat_template": False},
    "gsm8k": {"limit": 10},
}

# Cache des forwards partagé entre stratégies (en RAM). Gain important,
# mais la mémoire grossit avec le nombre de tokens traités (surtout GSM8K).
USE_TRACE_CACHE = True

# un dossier par lancement
STAMP = timestamp()
RESULTS_DIR = Path("results") / "lmeval" / STAMP
PLOTS_DIR = Path("plots") / "lmeval" / STAMP


# --------------------------------------------------
# Utilitaires
# --------------------------------------------------
def build_strategies():
    """NoExitStrategy en tête, puis toutes les combinaisons de la grille."""
    strategies = [NoExitStrategy()]
    for cls, grid, fixed in STRATEGY_GRID:
        for combo in expand_grid(grid):
            strategies.append(cls(**fixed, **combo))
    return strategies


def label(strategy) -> str:
    s = strategy_params(strategy)
    params = ", ".join(f"{k}={v}" for k, v in s.items() if k not in ("name", "verbose"))
    return f"{s['name']}({params})"


def save(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def print_summary(model_name: str, task: str, runs: list) -> None:
    lines = [f"\n=== {model_name} | {task} ==="]
    for run in runs:
        if "error" in run:
            lines.append(f"  {run['label']:<70} | ERREUR : {run['error']}")
            continue
        scores = ", ".join(
            f"{k}={v:.3f}" for k, v in run["results"].items()
            if isinstance(v, float) and "stderr" not in k
        )
        ee = run["results"]["early_exit"]
        lines.append(f"  {run['label']:<70} | {scores} | profondeur {ee['mean_depth_ratio']:.1%}")
    tqdm.write("\n".join(lines))


# --------------------------------------------------
# Boucle principale : modèles -> benchmarks -> stratégies
# --------------------------------------------------
runs_per_model = len(BENCHMARKS) * len(build_strategies())
# barre globale (en haut) : un pas = une stratégie évaluée sur un benchmark.
# Les barres de lm_eval (questions) et des tokens générés s'affichent en dessous.
bar = tqdm(total=len(MODELS) * runs_per_model, desc="Total", unit="run", position=0)

for model_name, quant in MODELS:
    tqdm.write(f"\n######## {model_name} ({quant or 'bf16'}) ########")
    done = 0
    adapter = ModelFactory.load_class(name=model_name, quant=quant)
    device = adapter.model.get_input_embeddings().weight.device
    model_dir = RESULTS_DIR / model_tag(model_name, quant)
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        for task, options in BENCHMARKS.items():
            path = model_dir / f"{task}.json"
            output = {"model": model_name, "quant": quant, "task": task,
                      "options": options, "n_layers": adapter.num_layers, "runs": []}

            trace_cache = CpuTraceCache(device) if USE_TRACE_CACHE else None

            for strategy in build_strategies():
                run = {"label": label(strategy), "strategy": strategy_params(strategy)}
                bar.set_postfix_str(f"{model_name} | {task} | {run['label']}")
                try:
                    run["results"] = evaluate_early_exit(
                        adapter, strategy, task=task,
                        trace_cache=trace_cache, **options,
                    )
                except Exception as e:
                    # la baseline est indispensable : sans elle, on abandonne ce modèle
                    if isinstance(strategy, NoExitStrategy):
                        raise
                    traceback.print_exc()
                    run["error"] = str(e)

                output["runs"].append(run)
                save(path, output)   # sauvegarde après chaque stratégie
                bar.update(1)
                done += 1

            del trace_cache
            print_summary(model_name, task, output["runs"])
            tqdm.write(f"Résultats : {path}")
            try:
                plot_benchmark(path, PLOTS_DIR)
            except Exception:
                # un graphique raté ne doit pas interrompre les évaluations
                traceback.print_exc()

    except Exception as e:
        # un modèle qui plante n'empêche pas de tester les suivants
        traceback.print_exc()
        tqdm.write(f"!! {model_name} a échoué : {e}")
        bar.update(runs_per_model - done)   # runs de ce modèle sautés
    finally:
        del adapter
        free_memory()

bar.close()

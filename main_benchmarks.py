"""Évaluation des stratégies d'early exit sur des benchmarks lm_eval (MMLU, GSM8K…).

Pour chaque modèle, chaque benchmark, chaque stratégie : précision du benchmark
+ statistiques d'early exit. NoExitStrategy passe en premier (baseline) et remplit
le cache de traces que les autres stratégies réutilisent.
"""
import os
os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
os.environ["SYCL_PI_LEVEL_ZERO_USE_RELAXED_ALLOCATION_LIMITS"] = "1"

import gc
import json
import traceback
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from evaluation.ExperimentRunner import ExperimentRunner
from evaluation.benchmarks import evaluate_early_exit
from inference.EarlyExitGenerate import CpuTraceCache, _strategy_params
from modelsClass.ModelFactory import ModelFactory
from plot_benchmarks import plot_benchmark
from strategy.CosineExitStrategy import CosineExitStrategy
from strategy.EntropyExitStrategy import EntropyStrategy
from strategy.NoExitStrategy import NoExitStrategy
from strategy.StableTokenStrategy import StableTokenStrategy

# --------------------------------------------------
# Configuration
# --------------------------------------------------
RESULTS_DIR = Path("results") / "benchmarks"

MODELS = [
    # (nom du dossier dans models/, quantification : None, "8bit" ou "4bit")
    ("qwen3.5-2b", None),
    ("qwen3.5-4b", None),
    ("qwen3.5-9b", None),
]

# tâche lm_eval -> options (limit = nb d'exemples par tâche, None = complet)
# GSM8K sans KV cache est très lent : garder une petite limite.
BENCHMARKS = {
    "mmlu": {"limit": 10},
    "gsm8k": {"limit": 20},
}

# (stratégie, paramètres à faire varier, paramètres fixes) — NoExit est ajoutée automatiquement
STRATEGIES = [
    (EntropyStrategy,
     {"threshold": [0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05, 0.02], "patience": [3]},
     {"min_layer": 10}),
    (CosineExitStrategy,
     {"threshold": [0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99], "patience": [3]},
     {"min_layer": 10}),
    (StableTokenStrategy,
     {"threshold": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], "patience": [3]},
     {"min_layer": 10}),
]

# Cache des forwards partagé entre stratégies (en RAM). Gain important,
# mais la mémoire grossit avec le nombre de tokens traités (surtout GSM8K).
USE_TRACE_CACHE = True


# --------------------------------------------------
# Utilitaires
# --------------------------------------------------
def build_strategies():
    """NoExitStrategy en tête, puis toutes les combinaisons de la grille."""
    strategies = [NoExitStrategy()]
    for cls, grid, fixed in STRATEGIES:
        for combo in ExperimentRunner._expand_grid(grid):
            strategies.append(cls(**fixed, **combo))
    return strategies


def label(strategy) -> str:
    s = _strategy_params(strategy)
    params = ", ".join(f"{k}={v}" for k, v in s.items() if k not in ("name", "verbose"))
    return f"{s['name']}({params})"


def free_memory():
    """Libère la VRAM du modèle précédent avant de charger le suivant."""
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

runs_per_model = len(BENCHMARKS) * len(build_strategies())
# barre globale (en haut) : un pas = une stratégie évaluée sur un benchmark.
# Les barres de lm_eval (questions) et des tokens générés s'affichent en dessous.
bar = tqdm(total=len(MODELS) * runs_per_model, desc="Total", unit="run", position=0)

for model_name, quant in MODELS:
    tqdm.write(f"\n######## {model_name} ({quant or 'bf16'}) ########")
    done = 0
    adapter = ModelFactory.load_class(name=model_name, quant=quant)
    device = adapter.model.get_input_embeddings().weight.device

    try:
        for task, options in BENCHMARKS.items():
            stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            path = RESULTS_DIR / f"{model_name}_{quant or 'bf16'}_{task}_{stamp}.json"
            output = {"model": model_name, "quant": quant, "task": task,
                      "options": options, "n_layers": adapter.num_layers, "runs": []}

            trace_cache = CpuTraceCache(device) if USE_TRACE_CACHE else None

            for strategy in build_strategies():
                run = {"label": label(strategy), "strategy": _strategy_params(strategy)}
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
                plot_benchmark(path)
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

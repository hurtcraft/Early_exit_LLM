"""Benchmark lm_eval (MMLU, GSM8K, HellaSwag, WikiText…) des stratégies d'early exit.

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


# Options lues par ce script (retirées avant l'appel à lm_eval) :
#   cache_traces : partage les forwards entre stratégies (défaut USE_TRACE_CACHE).
#                  À désactiver quand une trace garde beaucoup de positions (perplexité,
#                  réponses longues) : le cache ne tiendrait pas en RAM.
# Options passées à evaluate_early_exit :
#   max_length   : longueur max d'une entrée ; pour WikiText = taille des fenêtres.
BENCHMARKS = {
    #"mmlu": {"limit": 5, "num_fewshot": 5, "apply_chat_template": False},
    #"gsm8k": {"limit": 10,"num_fewshot": 5, "apply_chat_template": False},
    # perplexité : limit = nombre d'articles (62 au total, ~4 000 tokens chacun)
    # cache RAM ≈ 9 Go (2B) / 15 Go (4B) / 24 Go (9B) pour 20 articles ; ×3 pour les 62
    "wikitext": {"limit": 20, "num_fewshot": None, "apply_chat_template": False,
                 "max_length": 2048, "cache_traces": True},
    # QCM à réponses longues (4 fins de phrase) : limit = nombre de questions
    # cache RAM ≈ 5,5 Go (2B) / 9 Go (4B) / 15 Go (9B) pour 1000 questions
    "hellaswag": {"limit": 1000, "num_fewshot": 0, "apply_chat_template": False,
                  "cache_traces": True},
}


USE_TRACE_CACHE = True

STAMP = timestamp()
RESULTS_DIR = Path("results") / "lmeval" / STAMP
PLOTS_DIR = Path("plots") / "lmeval" / STAMP



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



runs_per_model = len(BENCHMARKS) * len(build_strategies())

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

            eval_options = dict(options)
            use_cache = eval_options.pop("cache_traces", USE_TRACE_CACHE)
            trace_cache = CpuTraceCache(device) if use_cache else None

            for strategy in build_strategies():
                run = {"label": label(strategy), "strategy": strategy_params(strategy)}
                bar.set_postfix_str(f"{model_name} | {task} | {run['label']}")
                try:
                    run["results"] = evaluate_early_exit(
                        adapter, strategy, task=task,
                        trace_cache=trace_cache, **eval_options,
                    )
                except Exception as e:
                    if isinstance(strategy, NoExitStrategy):
                        raise
                    traceback.print_exc()
                    run["error"] = str(e)

                output["runs"].append(run)
                save(path, output)
                bar.update(1)
                done += 1

            del trace_cache
            print_summary(model_name, task, output["runs"])
            tqdm.write(f"Résultats : {path}")
            try:
                plot_benchmark(path, PLOTS_DIR)
            except Exception:
                traceback.print_exc()

    except Exception as e:
        traceback.print_exc()
        tqdm.write(f"!! {model_name} a échoué : {e}")
        bar.update(runs_per_model - done)
    finally:
        del adapter
        free_memory()

bar.close()

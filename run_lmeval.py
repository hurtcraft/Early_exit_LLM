"""Benchmark lm_eval (MMLU, GSM8K, HellaSwag, WikiText…) des stratégies d'early exit.

Pour chaque modèle, chaque benchmark, chaque stratégie : score du benchmark
+ statistiques d'early exit. NoExitStrategy passe en premier (baseline) et remplit
le cache de traces que les autres stratégies réutilisent.

Usage :
    python run_lmeval.py                                  # tous les modèles, toutes les tâches
    python run_lmeval.py --models qwen3.5-2b --tasks gsm8k
    python run_lmeval.py --stamp 20261109_120000          # reprend / complète ce lancement

--stamp : nom du dossier de résultats. Plusieurs processus lancés avec le même
stamp écrivent dans le même dossier (un fichier par modèle x tâche). Si le JSON
d'une tâche existe déjà, les configurations déjà terminées sont sautées (reprise
après une coupure) ; celles en erreur sont relancées.

Sorties d'un lancement :
    results/lmeval/<stamp>/<modèle>/<tâche>.json
    plots/lmeval/<stamp>/<modèle>_<tâche>.png
"""
import os
os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
os.environ["SYCL_PI_LEVEL_ZERO_USE_RELAXED_ALLOCATION_LIMITS"] = "1"

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from tqdm import tqdm

from bench_lmeval.evaluate import evaluate_early_exit
from bench_lmeval.plots import plot_benchmark
from config import MODELS, strategy_grid
from early_exit.layers import CpuTraceCache
from early_exit.models.factory import ModelFactory
from early_exit.strategies.NoExitStrategy import NoExitStrategy
from early_exit.utils import expand_grid, free_memory, model_tag, strategy_params, timestamp


# Options lues par ce script (retirées avant l'appel à lm_eval) :
#   cache_traces : partage les forwards entre stratégies (défaut USE_TRACE_CACHE).
#                  À désactiver quand une trace garde beaucoup de positions (perplexité,
#                  réponses longues) : le cache ne tiendrait pas en RAM.
#   grid         : "full" (défaut) ou "strict" (cf. config.strategy_grid).
# Options passées à evaluate_early_exit :
#   max_length   : longueur max d'une entrée ; pour WikiText = taille des fenêtres.
BENCHMARKS = {
    "wikitext": {"limit": 20, "num_fewshot": None, "apply_chat_template": False,"max_length": 2048, "cache_traces": False},
    "hellaswag": {"limit": 1000, "num_fewshot": 0, "apply_chat_template": False,"cache_traces": False},
    "mmlu": {"limit": 50, "num_fewshot": 5, "apply_chat_template": False},
    "gsm8k": {"limit": 100, "num_fewshot": 5, "apply_chat_template": False},
}

USE_TRACE_CACHE = True

# Fichier d'avancement lu par l'orchestrateur de launch_parallel.sh (variable
# d'environnement posée par ce script). Lancé à la main : rien n'est écrit.
PROGRESS_FILE = os.environ.get("LMEVAL_PROGRESS")


class Progress:
    """Avancement du processus : une ligne "fait|total|sautés|début|statut".
    sautés = configurations reprises d'un lancement précédent (ignorées pour
    estimer le temps restant)."""

    def __init__(self, bar):
        self.bar = bar
        self.start = int(time.time())
        self.skipped = 0

    def write(self, status: str) -> None:
        if not PROGRESS_FILE:
            return
        tmp = f"{PROGRESS_FILE}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(f"{self.bar.n}|{self.bar.total}|{self.skipped}|{self.start}|{status}\n")
        os.replace(tmp, PROGRESS_FILE)   # remplacement atomique : jamais lu à moitié écrit


def build_strategies(n_layers: int, grid: str = "full"):
    """NoExitStrategy en tête, puis toutes les combinaisons de la grille du modèle."""
    strategies = [NoExitStrategy()]
    for cls, params, fixed in strategy_grid(n_layers, grid):
        for combo in expand_grid(params):
            strategies.append(cls(**fixed, **combo))
    return strategies


def n_strategies(grid: str) -> int:
    # le nombre de stratégies ne dépend pas du modèle (seul min_layer change)
    return 1 + sum(len(expand_grid(p)) for _, p, _ in strategy_grid(0, grid))


def label(strategy) -> str:
    s = strategy_params(strategy)
    params = ", ".join(f"{k}={v}" for k, v in s.items() if k not in ("name", "verbose"))
    return f"{s['name']}({params})"


def save(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_previous(path: Path, options: dict) -> list:
    """Configurations déjà terminées d'un lancement précédent (même stamp).
    Les runs en erreur ne sont pas gardés : ils seront relancés."""
    if not path.exists():
        return []
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous.get("options") != options:
        sys.exit(f"{path} existe avec d'autres options ({previous.get('options')}) que "
                 f"celles de BENCHMARKS ({options}) : utiliser un autre --stamp.")
    return [r for r in previous.get("runs", []) if "results" in r]


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


def parse_args():
    model_names = [m for m, _ in MODELS]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=model_names, choices=model_names,
                    help="modèles à évaluer (défaut : tous ceux de config.MODELS)")
    ap.add_argument("--tasks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS),
                    help="tâches à évaluer (défaut : toutes celles de BENCHMARKS)")
    ap.add_argument("--stamp", default=None,
                    help="dossier de résultats (défaut : date et heure) ; existant = reprise")
    return ap.parse_args()


def run_model(model_name, quant, tasks, results_dir, plots_dir, progress):
    bar = progress.bar
    tqdm.write(f"\n######## {model_name} ({quant or 'bf16'}) ########")
    progress.write(f"{model_name} | chargement du modèle")
    adapter = ModelFactory.load_class(name=model_name, quant=quant)
    device = adapter.model.get_input_embeddings().weight.device
    model_dir = results_dir / model_tag(model_name, quant)
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        for task in tasks:
            options = BENCHMARKS[task]
            path = model_dir / f"{task}.json"
            runs = load_previous(path, options)
            done = {r["label"] for r in runs}
            if done:
                tqdm.write(f"{model_name} | {task} : reprise, {len(done)} configurations déjà faites")
            output = {"model": model_name, "quant": quant, "task": task,
                      "options": options, "n_layers": adapter.num_layers, "runs": runs}

            eval_options = dict(options)
            use_cache = eval_options.pop("cache_traces", USE_TRACE_CACHE)
            grid = eval_options.pop("grid", "full")
            trace_cache = CpuTraceCache(device) if use_cache else None

            for strategy in build_strategies(adapter.num_layers, grid):
                run = {"label": label(strategy), "strategy": strategy_params(strategy)}
                if run["label"] in done:
                    bar.update(1)
                    progress.skipped += 1
                    continue
                bar.set_postfix_str(f"{model_name} | {task} | {run['label']}")
                progress.write(f"{model_name} | {task} | {run['label']}")
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

            del trace_cache
            print_summary(model_name, task, output["runs"])
            tqdm.write(f"Résultats : {path}")
            try:
                plot_benchmark(path, plots_dir)
            except Exception:
                traceback.print_exc()
    finally:
        del adapter
        free_memory()


def main():
    args = parse_args()
    stamp = args.stamp or timestamp()
    results_dir = Path("results") / "lmeval" / stamp
    plots_dir = Path("plots") / "lmeval" / stamp
    tqdm.write(f"Résultats dans {results_dir}")

    models = [(m, q) for m, q in MODELS if m in args.models]
    runs_per_model = sum(n_strategies(BENCHMARKS[t].get("grid", "full")) for t in args.tasks)
    bar = tqdm(total=len(models) * runs_per_model, desc="Total", unit="run", position=0)
    progress = Progress(bar)
    failed = []

    for model_name, quant in models:
        start = bar.n
        try:
            run_model(model_name, quant, args.tasks, results_dir, plots_dir, progress)
        except Exception as e:
            traceback.print_exc()
            tqdm.write(f"!! {model_name} a échoué : {e}")
            failed.append(model_name)
            bar.update(runs_per_model - (bar.n - start))

    progress.write("terminé" if not failed else f"terminé, ÉCHEC : {', '.join(failed)}")
    bar.close()


if __name__ == "__main__":
    main()

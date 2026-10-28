"""Benchmark sur prompts custom : chaque stratégie est comparée à la baseline
(NoExitStrategy) sur les réponses générées.

Sorties d'un lancement :
    results/prompts/<date>/<modèle>/GRID_SUMMARY.json + runs/*.json
    plots/prompts/<date>/<modèle>_<Stratégie>.png
"""
import os
os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
os.environ["SYCL_PI_LEVEL_ZERO_USE_RELAXED_ALLOCATION_LIMITS"] = "1"

from pathlib import Path

from bench_prompts.plots import plot_grid_summary
from bench_prompts.runner import PromptGridRunner
from config import MODELS, STRATEGY_GRID
from early_exit.models.factory import ModelFactory
from early_exit.utils import free_memory, model_tag, timestamp

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "bench_prompts" / "data"
QUESTIONS_FILE = DATA_DIR / "easy" / "simple_questions.txt"
# QUESTIONS_FILE = DATA_DIR / "medium" / "medium_questions.txt"
# QUESTIONS_FILE = DATA_DIR / "hard" / "complex_questions.txt"
# QUESTIONS_FILE = DATA_DIR / "easy" / "small_set.txt"

MAX_NEW_TOKENS = 20
VERBOSE = True

STAMP = timestamp()
RESULTS_DIR = Path("results") / "prompts" / STAMP
PLOTS_DIR = Path("plots") / "prompts" / STAMP


def load_questions(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [q.strip() for q in f if q.strip()]


def build_runner(adapter, model_name, quant, questions):
    runner = PromptGridRunner(
        adapter=adapter,
        model_name=model_name,
        quant=quant,
        questions=questions,
        results_dir=RESULTS_DIR / model_tag(model_name, quant),
        max_new_tokens=MAX_NEW_TOKENS,
        verbose=VERBOSE,
    )
    for strategy_cls, param_grid, fixed_params in STRATEGY_GRID:
        runner.add_strategy(strategy_cls, param_grid=param_grid, fixed_params=fixed_params)
    return runner


def plot_all(summary_paths: list) -> None:
    """Génère les plots de chaque GRID_SUMMARY dans PLOTS_DIR."""
    print(f"\n######## Plots -> {PLOTS_DIR} ########")
    for path in summary_paths:
        try:
            plot_grid_summary(path, PLOTS_DIR)
        except Exception as e:
            # un plot raté n'empêche pas de tracer les autres modèles
            print(f"!! plot de {path} a échoué : {e}")


# --------------------------------------------------
# 1) Tests des prompts : un GRID_SUMMARY par modèle
# --------------------------------------------------
questions = load_questions(QUESTIONS_FILE)
summary_paths = []
for model_name, quant in MODELS:
    print(f"\n######## {model_name} ({quant or 'bf16'}) ########")
    adapter = ModelFactory.load_class(name=model_name, quant=quant)
    try:
        runner = build_runner(adapter, model_name, quant, questions)
        runner.run_all()
        summary_paths.append(runner.summary_path)
    except Exception as e:
        # un modèle qui plante n'empêche pas de tester les suivants
        print(f"!! {model_name} a échoué : {e}")
    finally:
        del adapter
        free_memory()

# --------------------------------------------------
# 2) Plots à partir des GRID_SUMMARY générés
# --------------------------------------------------
plot_all(summary_paths)

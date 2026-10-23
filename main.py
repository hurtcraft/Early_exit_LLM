import os
os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
os.environ["SYCL_PI_LEVEL_ZERO_USE_RELAXED_ALLOCATION_LIMITS"] = "1"
from evaluation.ExperimentRunner import ExperimentRunner
from evaluation.benchmarks import evaluate_baseline, evaluate_early_exit
from inference.EarlyExitGenerate import _strategy_params, early_exit_generate
from inference.EarlyExitHFLM import EarlyExitHFLM
from inference.EarlyExitRunner import EarlyExitRunner
from inference.MonitoringHiddentStates import MonitoringHiddenStates
from models_utils import build_run_config, load_model, show_probability_distribution,save_results, summarize_run
import json
from lm_eval.models.huggingface import HFLM
from model_evaluation import *
from modelsClass.ModelFactory import *
from strategy.CosineExitStrategy import CosineExitStrategy
from strategy.EntropyExitStrategy import *
from strategy.NoExitStrategy import NoExitStrategy
from strategy.StableTokenStrategy import *
from datetime import datetime
import gc
from plot_results import plot_grid_summary

# --------------------------------------------------
# Configuration
# --------------------------------------------------
PROMPTS_DIR = Path("prompts")
RESULTS_DIR = Path("results")
# un dossier de plots par lancement : plots/<date_heure>/
PLOTS_DIR = Path("plots") / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

QUANT = None
VERBOSE = False

MODELS = [
    ("qwen3.5-2b", None),
    ("qwen3.5-4b", None),
    ("qwen3.5-9b", None),
]


with open(r"C:\Users\Hurtcraft\Desktop\projet recherche\prompts\easy\simple_questions.txt", encoding="utf-8") as f:
    questions_easy = [q.strip() for q in f if q.strip()]
# with open(r"C:\Users\Hurtcraft\Desktop\projet recherche\prompts\medium\medium_questions.txt", encoding="utf-8") as f:
#     questions_medium = [q.strip() for q in f if q.strip()]
# with open(r"C:\Users\Hurtcraft\Desktop\projet recherche\prompts\hard\complex_questions.txt", encoding="utf-8") as f:
#     questions_hard = [q.strip() for q in f if q.strip()]
# with open(r"C:\Users\Hurtcraft\Desktop\projet recherche\prompts\easy\small_set.txt", encoding="utf-8") as f:
#     questions_easy = [q.strip() for q in f if q.strip()]


def build_runner(adapter, model_name, quant):
    runner = ExperimentRunner(
        adapter=adapter,
        model_name=model_name,
        quant=quant,
        questions=questions_easy,
        results_dir=RESULTS_DIR,
        max_new_tokens=20,
        verbose=VERBOSE,
    )
    runner.add_strategy(
        EntropyStrategy,
        param_grid={"threshold": [0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05, 0.02], "patience": [3]},
        fixed_params={"min_layer": 10},
    )
    runner.add_strategy(
        CosineExitStrategy,
        param_grid={"threshold": [0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99], "patience": [3]},
        fixed_params={"min_layer": 10},
    )
    runner.add_strategy(
        StableTokenStrategy,
        param_grid={"threshold": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], "patience": [3]},
        fixed_params={"min_layer": 10},
    )
    return runner


def free_memory():
    """Libère la VRAM du modèle précédent avant de charger le suivant."""
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def plot_all(summary_paths: list) -> None:
    """Génère les plots de chaque GRID_SUMMARY dans PLOTS_DIR."""
    print(f"\n######## Plots -> {PLOTS_DIR} ########")
    for path in summary_paths:
        try:
            plot_grid_summary(path, PLOTS_DIR)
        except Exception as e:
            # un plot raté n'empêche pas de tracer les autres modèles
            print(f"!! plot de {path.name} a échoué : {e}")


# --------------------------------------------------
# 1) Tests des prompts : un GRID_SUMMARY par modèle
# --------------------------------------------------
summary_paths = []
for model_name, quant in MODELS:
    print(f"\n######## {model_name} ({quant or 'bf16'}) ########")
    adapter = ModelFactory.load_class(name=model_name, quant=quant)
    try:
        runner = build_runner(adapter, model_name, quant)
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






















# monitor = MonitoringHiddenStates(adapter.model, adapter.tokenizer, top_k=5)
# monitor.show("quel est la capital de la chine", max_new_tokens=20, use_chat_template=False)
# early_exit_generate(adapter, stableTokenStrategy, "What is the capital of france ?", max_new_tokens=20)
# m = early_exit_generate(
#     adapter, stableTokenStrategy, "quel est la capital de la chine",
#     max_new_tokens=30, use_chat_template=True,
#     model_name=MODEL_NAME, quant=QUANT,
# )



# stableTokenStrategy = StableTokenStrategy(patience=3  , min_layer=10,threshold=0.5,tokenizer=adapter.tokenizer)
# entropyExitStrategy=EntropyStrategy(threshold=0.2,min_layer=10)
# cosineExitStrategy = CosineExitStrategy(threshold=0.99, min_layer=10, patience=1)




















# model, tokenizer = load_model(
#     MODEL_NAME,
#     quant=QUANT
# )
# model, tokenizer = ModelFactory.load_model(MODEL_NAME,QUANT)
# model = ModelFactory.load_class(
#     name=MODEL_NAME,
#     quant=QUANT,
# )
# print(f"Model: {MODEL_NAME}")
# print(f"Quantization: {QUANT}")

# def run_benchmark(lm, task: str, limit: int | None = None):
#     print(f"\nÉvaluation de {task} (limit={limit or 'tous les exemples'})")
#     evaluation = evaluator.simple_evaluate(
#         model=lm,
#         tasks=[task],
#         batch_size=1,
#         limit=limit,
#     )
#     return evaluation["results"].get(task, {})


# def evaluate_early_exit(adapter, strategy, limit=None):
#     lm = EarlyExitHFLM(adapter, strategy, batch_size=1)
#     results = run_benchmark(lm, "mmlu", limit)

#     exits = torch.tensor(lm.exit_layers, dtype=torch.float)
#     results["early_exit"] = {
#         "mean_exit_layer": exits.mean().item(),
#         "mean_depth_ratio": exits.mean().item() / (adapter.num_layers - 1),
#         "n_calls": len(lm.exit_layers),
#     }
#     return results



#results = evaluate_model(model=model,tokenizer=tokenizer)
# save_results(
#     model_name=MODEL_NAME,
#     results=result,
#     quant=QUANT,
# )



# show_probability_distribution(
#     model,
#     tokenizer,
#     "What is the capital of France?",
#     top_k=5,
# )
# process_all_prompts(
#     model,
#     tokenizer,
#     PROMPTS_DIR,
#     RESULTS_DIR,
#     MODEL_NAME,
#     QUANT
# )

























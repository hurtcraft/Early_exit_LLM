import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

from inference.EarlyExitHFLM import EarlyExitHFLM


def run_benchmark(
    lm,
    task: str,
    limit: int | None = None,
    num_fewshot: int | None = None,
    apply_chat_template: bool = True,
) -> dict:
    print(f"\nÉvaluation de {task} (limit={limit or 'tous les exemples'})")
    evaluation = evaluator.simple_evaluate(
        model=lm,
        tasks=[task],
        batch_size=1,              # pas de padding : requis par la boucle de couches
        limit=limit,
        num_fewshot=num_fewshot,   # None = valeur par défaut de la tâche
        apply_chat_template=apply_chat_template,
        log_samples=False,
    )
    # les scores agrégés d'un groupe (ex. mmlu) sont rangés dans "groups"
    return (
        evaluation.get("groups", {}).get(task)
        or evaluation["results"].get(task, {})
    )


def evaluate_baseline(adapter, task="mmlu", limit=None, **kwargs) -> dict:
    """Modèle HF complet via lm_eval (référence externe, avec KV cache)."""
    lm = HFLM(
        pretrained=adapter.model, tokenizer=adapter.tokenizer,
        batch_size=1, enable_thinking=False,
    )
    return run_benchmark(lm, task, limit, **kwargs)


def evaluate_early_exit(
    adapter, strategy, task="mmlu", limit=None, trace_cache=None, **kwargs
) -> dict:
    lm = EarlyExitHFLM(
        adapter, strategy, trace_cache=trace_cache,
        batch_size=1, enable_thinking=False,
    )
    results = run_benchmark(lm, task, limit, **kwargs)

    # une décision par question (loglikelihood) ou par token généré (generate_until)
    exits = torch.tensor(lm.exit_layers, dtype=torch.float)
    last = adapter.num_layers - 1
    results["early_exit"] = {
        "mean_exit_layer": exits.mean().item(),
        "std_exit_layer": exits.std(unbiased=False).item(),
        "mean_depth_ratio": exits.mean().item() / last,
        "early_exit_rate": (exits < last).float().mean().item(),
        "n_decisions": len(lm.exit_layers),
    }
    return results

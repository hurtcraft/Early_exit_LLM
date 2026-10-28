"""Métriques du benchmark sur prompts custom : résumé d'un run et comparaison à la baseline."""
from typing import Any, Dict, List, Optional, Sequence

import torch

from early_exit.utils import strategy_params


def build_run_config(adapter, strategy, model_name, quant, use_chat_template, max_new_tokens) -> dict:
    return {
        "model": model_name,
        "quant": quant,
        "n_layers": adapter.num_layers,
        "strategy": strategy_params(strategy),
        "use_chat_template": use_chat_template,
        "max_new_tokens": max_new_tokens,
    }


def summarize_run(results: list[dict], n_layers: int) -> dict:
    all_exits = torch.tensor([l for r in results for l in r["exit_layers"]], dtype=torch.float)
    last = n_layers - 1
    return {
        "n_questions": len(results),
        "n_tokens": len(all_exits),
        "mean_exit_layer": all_exits.mean().item(),
        "std_exit_layer": all_exits.std(unbiased=False).item(),
        "mean_depth_ratio": all_exits.mean().item() / last,
        "early_exit_rate": (all_exits < last).float().mean().item(),
    }


# =========================================================================== #
# Comparaison avec la baseline
# =========================================================================== #
def _get(result: Any, key: str) -> Any:
    """Lit une clé dans un résultat (dict ou objet), avec une erreur explicite."""
    if isinstance(result, dict):
        if key in result:
            return result[key]
        raise KeyError(
            f"Clé '{key}' absente du résultat de early_exit_generate. "
            f"Clés disponibles : {list(result.keys())}. "
            f"Adapte text_key / exit_layers_key dans PromptGridRunner."
        )
    if hasattr(result, key):
        return getattr(result, key)
    raise AttributeError(
        f"Attribut '{key}' absent du résultat ({type(result).__name__}). "
        f"Adapte text_key / exit_layers_key dans PromptGridRunner."
    )


def response_divergence(base_tokens: Sequence, tokens: Sequence) -> Dict[str, float]:
    """
    Compare baseline vs stratégie. Comme la génération est autorégressive,
    dès que la séquence diverge tout ce qui suit est différent : le préfixe
    commun est donc la métrique de fidélité clé.
    """
    base_tokens, tokens = list(base_tokens), list(tokens)
    longest = max(len(base_tokens), len(tokens))

    prefix = 0
    for a, b in zip(base_tokens, tokens):
        if a != b:
            break
        prefix += 1

    return {
        # 1.0 si la réponse est strictement identique à la baseline
        "exact_match": float(base_tokens == tokens),
        # fraction de la réponse reproduite avant la 1re divergence
        # (divisée par la plus longue des deux : pas de 1.0 si la stratégie rallonge)
        "prefix_match_ratio": prefix / longest if longest else 1.0,
    }


def layer_savings(exit_layers: Sequence[float], num_layers: int) -> Dict[str, float]:
    """Couches économisées par rapport au modèle complet (indexé 0..num_layers-1)."""
    last = num_layers - 1
    mean_exit = sum(exit_layers) / len(exit_layers) if exit_layers else float(last)
    return {"skipped_layers": last - mean_exit}


def compare_to_baseline(
    baseline_results: List[Any],
    results: List[Any],
    num_layers: int,
    text_key: str,
    exit_layers_key: str,
    tokens_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare chaque question à la baseline et agrège les métriques."""
    if len(baseline_results) != len(results):
        raise ValueError("La baseline et le run n'ont pas le même nombre de questions.")

    def has(r, key):
        return key in r if isinstance(r, dict) else hasattr(r, key)

    use_tokens = bool(tokens_key) and all(has(r, tokens_key) for r in results + baseline_results)
    if tokens_key and not use_tokens:
        print(f"    (!) clé '{tokens_key}' absente des résultats : "
              f"comparaison par mots à la place des tokens")

    def seq(r):
        if use_tokens:
            return list(_get(r, tokens_key))
        return str(_get(r, text_key)).split()

    per_question = []
    for i, (b, r) in enumerate(zip(baseline_results, results)):
        per_question.append({
            "question_idx": i,
            "baseline_text": _get(b, text_key),
            "text": _get(r, text_key),
            **response_divergence(seq(b), seq(r)),
            **layer_savings(list(_get(r, exit_layers_key)), num_layers),
        })

    def mean(key):
        return sum(q[key] for q in per_question) / len(per_question)

    summary = {
        "exact_match_rate": mean("exact_match"),
        "mean_prefix_match_ratio": mean("prefix_match_ratio"),
        "mean_skipped_layers": mean("skipped_layers"),
    }
    return {"summary": summary, "per_question": per_question}

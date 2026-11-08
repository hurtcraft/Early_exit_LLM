"""Description des métriques lm_eval, partagée par les scripts de graphes
(bench_lmeval/plots.py, compare_model.py).

Deux familles :
  - proportions (acc, acc_norm, exact_match…) : entre 0 et 1, plus haut = mieux,
    affichées en %.
  - valeurs brutes (word_perplexity, byte_perplexity, bits_per_byte de WikiText) :
    plus bas = mieux, affichées telles quelles (échelle log pour les perplexités).
"""

# sous-chaînes des métriques où plus bas = mieux
LOWER_IS_BETTER = ("perplexity", "bits_per_byte")

# métrique principale par tâche (sinon : première métrique trouvée)
PRIMARY_METRIC = {
    "hellaswag": "acc_norm,none",       # normalisée par la longueur de la réponse
    "wikitext": "word_perplexity,none",
}


def score_metrics(results: dict) -> list:
    """Scores du benchmark (ex. 'acc,none', 'exact_match,strict-match'), sans stderr."""
    return [k for k, v in results.items() if isinstance(v, float) and "stderr" not in k]


def primary_metric(task: str, results: dict) -> str:
    preferred = PRIMARY_METRIC.get(task)
    if preferred in results:
        return preferred
    return score_metrics(results)[0]


def metric_name(key: str) -> str:
    name, _, filt = key.partition(",")
    return name if filt in ("", "none") else f"{name} ({filt})"


def higher_is_better(key: str) -> bool:
    return not any(s in key for s in LOWER_IS_BETTER)


def is_rate(key: str) -> bool:
    """Proportion entre 0 et 1, affichée en % (toutes les métriques sauf perplexités)."""
    return higher_is_better(key)

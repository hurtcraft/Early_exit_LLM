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


# --- calcul relatif (génération) ---------------------------------------------
# Coût d'une génération ≈ tokens générés × couches traversées par token.
# Calcul relatif = ce coût / celui de la baseline (toutes les couches) :
#   100 % = autant que le modèle complet ; > 100 % = plus cher que le modèle complet
#   (l'early exit fait générer plus de tokens, ex. texte dégénéré jusqu'à la longueur max).
# Estimation : dans la simulation toutes les couches sont calculées, et le coût du
# prompt (identique pour toutes les configs) ainsi que celui des stratégies
# (projection sur le vocabulaire) ne sont pas comptés.

def relative_compute(early_exit: dict, base_early_exit: dict, n_layers: int) -> float:
    """Calcul relatif en % à partir des stats 'early_exit' d'un run et de la baseline."""
    cost = early_exit["n_decisions"] * (early_exit["mean_exit_layer"] + 1)
    base_cost = base_early_exit["n_decisions"] * n_layers
    return 100 * cost / base_cost


def is_generative(early_exits: list, base_early_exit: dict) -> bool:
    """Vrai si le nombre de décisions varie d'une config à l'autre (génération : le
    nombre de tokens générés dépend de la config). En loglikelihood (MMLU, HellaSwag,
    WikiText) il est fixé par le benchmark : le calcul relatif = la profondeur."""
    n = base_early_exit.get("n_decisions")
    return any(ee.get("n_decisions") != n for ee in early_exits)

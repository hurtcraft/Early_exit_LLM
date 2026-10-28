"""Configuration partagée par run_prompts.py et run_lmeval.py : modèles et grille de
stratégies. NoExitStrategy (baseline) est ajoutée automatiquement en tête."""
from early_exit.strategies.CosineExitStrategy import CosineExitStrategy
from early_exit.strategies.EntropyExitStrategy import EntropyStrategy
from early_exit.strategies.StableTokenStrategy import StableTokenStrategy

MODELS = [
    ("qwen3.5-2b", None),
    # ("qwen3.5-4b", None),
    # ("qwen3.5-9b", None),
]

# Seuils calibrés sur MMLU / qwen3.5-2b (résultats du 20261023_184507) :
# - Entropy     : <= 0.1 ~ baseline, gain quasi nul -> on monte vers 0.7 pour trouver
#                 le point où la précision décroche.
# - Cosine      : falaise entre 0.92 (précision au hasard) et 0.95 (jamais de sortie)
#                 -> grille fine dans cette zone, rien au-dessus de 0.95.
# - StableToken : précision insensible entre 0.5 et 0.9 -> on descend jusqu'à 0.2.
STRATEGY_GRID = [
    (EntropyStrategy,
     {"threshold": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8], "patience": [1,2,3]},
     {"min_layer": 1}),
    (CosineExitStrategy,
     {"threshold": [0.85,0.90, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95], "patience": [1,2,3]},
     {"min_layer": 1}),
    (StableTokenStrategy,
     {"threshold": [0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "patience": [1,2,3]},
     {"min_layer": 1}),
]

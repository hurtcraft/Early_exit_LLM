
from early_exit.strategies.CosineExitStrategy import CosineExitStrategy
from early_exit.strategies.EntropyExitStrategy import EntropyStrategy
from early_exit.strategies.StableTokenStrategy import StableTokenStrategy

MODELS = [
    ("qwen3.5-2b", None),
    ("qwen3.5-4b", None),
    ("qwen3.5-9b", None),
]


PATIENCES = [2, 3]


def strategy_grid(n_layers: int) -> list:
    """Grille des stratégies pour un modèle de n_layers couches.
    min_layer = moitié de la profondeur : même position relative quel que soit le
    modèle (2B : 24 couches -> 12 ; 4B / 9B : 32 couches -> 16)."""
    min_layer = n_layers // 2
    return [
        (EntropyStrategy,
         {"threshold": [0.05, 0.1, 0.2, 0.4, 0.6], "patience": PATIENCES},
         {"min_layer": min_layer}),
        (StableTokenStrategy,
         {"threshold": [0.95, 0.9, 0.7, 0.5, 0.3], "patience": PATIENCES},
         {"min_layer": min_layer}),
        (CosineExitStrategy,
         {"threshold": [0.95, 0.93, 0.9, 0.87, 0.85], "patience": PATIENCES},
         {"min_layer": min_layer}),
    ]

# Ancienne grille (résultats MMLU / GSM8K du 28/10) :
# STRATEGY_GRID = [
#     (EntropyStrategy,
#      {"threshold": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8], "patience": [1,2,3]},
#      {"min_layer": 1}),
#     (CosineExitStrategy,
#      {"threshold": [0.85,0.90, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95], "patience": [1,2,3]},
#      {"min_layer": 1}),
#     (StableTokenStrategy,
#      {"threshold": [0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "patience": [1,2,3]},
#      {"min_layer": 1}),
# ]

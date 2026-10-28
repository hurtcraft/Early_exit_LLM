import gc
import itertools
from datetime import datetime
from typing import Any, Dict, List

import torch


def strategy_params(strategy) -> dict:
    """Hyperparamètres de la stratégie (attributs simples, sans tokenizer)."""
    params = {
        k: v for k, v in vars(strategy).items()
        if isinstance(v, (int, float, str, bool)) and not k.startswith("_")
    }
    return {"name": type(strategy).__name__, **params}


def expand_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """{"threshold": [0.1, 0.2], "patience": [3]} -> une combinaison par dict."""
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(param_grid[k] for k in keys))
    ]


def timestamp() -> str:
    """Horodatage des dossiers de lancement : 20261023_143012."""
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def model_tag(model_name: str, quant: str | None) -> str:
    """Nom du modèle + quantification, ex. qwen3.5-2b_bf16."""
    return f"{model_name}_{quant or 'bf16'}"


def free_memory():
    """Libère la VRAM du modèle précédent avant de charger le suivant."""
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

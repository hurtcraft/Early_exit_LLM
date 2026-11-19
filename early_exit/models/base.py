from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseModel(ABC):
    """Interface commune aux modèles évalués."""

    # Tuned lens (early_exit/lens.py) appliqué par get_logits aux couches
    # intermédiaires. None = logit lens (comportement d'origine).
    lens = None

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Retourne le nombre de couches du modèle."""
        raise NotImplementedError

    @abstractmethod
    def prepare_inputs(self, prompt: str) -> dict[str, torch.Tensor]:
        """Tokenise et prépare le prompt."""
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Calcule les embeddings initiaux."""
        raise NotImplementedError

    @abstractmethod
    def forward_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Exécute une couche du modèle."""
        raise NotImplementedError

    @abstractmethod
    def get_logits(self, hidden_states: torch.Tensor, layer: int | None = None) -> torch.Tensor:
        """Calcule les logits à partir d'une représentation intermédiaire.
        layer : couche d'où vient hidden_states ; avec un tuned lens chargé, son
        translateur est appliqué avant la tête (None = pas de lens)."""
        raise NotImplementedError

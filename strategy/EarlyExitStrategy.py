from abc import ABC, abstractmethod
import torch


class EarlyExitStrategy(ABC):

    # False si la stratégie n'utilise pas les logits (évite la projection sur le vocabulaire)
    needs_logits: bool = True

    def first_layer(self, n_layers: int) -> int:
        """Première couche où should_exit a un effet : les couches précédentes ne
        sont pas soumises à la stratégie (ni projetées sur le vocabulaire)."""
        return 0

    @abstractmethod
    def should_exit(
        self,
        layer_idx: int,
        logits: torch.Tensor,
        state: dict,
    ) -> bool:
        """Détermine si le calcul peut être interrompu."""
        raise NotImplementedError

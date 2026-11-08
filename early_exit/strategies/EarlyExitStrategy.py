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

    def should_exit_rows(
        self,
        layer_idx: int,
        logits: torch.Tensor | None,
        hidden: torch.Tensor,
        state: dict,
    ) -> torch.Tensor:
        """Version par position de should_exit : chaque ligne est une position
        indépendante (réponses de plusieurs tokens, perplexité).

        logits : [P, V] (None si needs_logits est False) ; hidden : [P, H].
        state  : tenseurs [P] gardés d'une couche à l'autre.
        Retourne un tenseur booléen [P] : True = cette position sort à cette couche.
        Avec P = 1, même décision que should_exit avec un batch de 1.
        """
        raise NotImplementedError(
            f"{type(self).__name__} ne gère pas les réponses de plusieurs tokens "
            f"(should_exit_rows non implémentée)"
        )

import torch

from strategy.EarlyExitStrategy import EarlyExitStrategy


class NoExitStrategy(EarlyExitStrategy):
    """Baseline : ne sort jamais en avance, le modèle traverse toutes les couches."""

    needs_logits = False

    def first_layer(self, n_layers: int) -> int:
        return n_layers  # jamais appelée

    def should_exit(
        self,
        layer_idx: int,
        logits: torch.Tensor,
        state: dict,
    ) -> bool:
        return False
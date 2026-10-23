from __future__ import annotations

import time
from typing import Any

import torch

from modelsClass.BaseModel import BaseModel
from strategy.EarlyExitStrategy import EarlyExitStrategy


class EarlyExitRunner:
    """
    Exécute un modèle couche par couche et applique une stratégie
    d'early exit pour chaque token prédit.

    Cette première version prédit uniquement le prochain token
    à partir d'un prompt. Elle ne réalise pas encore de génération
    autorégressive complète avec cache KV.
    """

    def __init__(
        self,
        model: BaseModel,
        strategy: EarlyExitStrategy | None = None,
    ) -> None:
        self.model = model
        self.strategy = strategy

    @torch.inference_mode()
    def predict_next_token(
        self,
        prompt: str,
        early_exit: bool = True,
    ) -> dict[str, Any]:
        """
        Prédit le prochain token du modèle.

        Args:
            prompt: Texte d'entrée.
            early_exit: Active ou désactive la stratégie.

        Returns:
            Un dictionnaire contenant le token prédit,
            les métriques d'exécution et les logits finaux.
        """
        if early_exit and self.strategy is None:
            raise ValueError(
                "Une stratégie doit être fournie pour activer l'early exit."
            )

        inputs = self.model.prepare_inputs(prompt)

        hidden_states = self.model.get_embeddings(
            inputs["input_ids"]
        )

        state: dict[str, Any] = {}

        total_layers = self.model.num_layers
        executed_layers = 0
        exit_layer = None

        logits = None

        # Synchronise le périphérique avant la mesure si nécessaire.
        self._synchronize_device()

        start_time = time.perf_counter()

        for layer_idx in range(total_layers):
            hidden_states, _ = self.model.forward_layer(
                layer_idx,
                hidden_states,
            )

            executed_layers += 1

            # Projection de la représentation intermédiaire en logits.
            logits = self.model.get_logits(hidden_states)

            # Vérifie si l'exécution peut s'arrêter.
            if early_exit and self.strategy.should_exit(
                layer_idx=layer_idx,
                logits=logits,
                state=state,
            ):
                exit_layer = layer_idx
                break

        self._synchronize_device()

        latency_seconds = time.perf_counter() - start_time

        if logits is None:
            raise RuntimeError("Aucun logit n'a été calculé.")

        next_token = logits[:, -1, :].argmax(dim=-1)

        return {
            "token_id": next_token,
            "token": self.model.tokenizer.batch_decode(
                next_token.detach().cpu().tolist(),
                skip_special_tokens=False,
            ),
            "logits": logits[:, -1, :].detach(),
            "early_exit": exit_layer is not None,
            "exit_layer": exit_layer,
            "layers_executed": executed_layers,
            "total_layers": total_layers,
            "layer_reduction": (
                1.0 - executed_layers / total_layers
            ),
            "latency_seconds": latency_seconds,
        }

    def _synchronize_device(self) -> None:
        """
        Synchronise CUDA ou XPU lorsque ces périphériques sont utilisés.
        """
        device = self.model.model.get_input_embeddings().weight.device

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

        elif device.type == "xpu" and hasattr(torch, "xpu"):
            if torch.xpu.is_available():
                torch.xpu.synchronize(device)

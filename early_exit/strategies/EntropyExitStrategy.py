import math

import torch

from early_exit.strategies.EarlyExitStrategy import EarlyExitStrategy


class EntropyStrategy(EarlyExitStrategy):
    """Sort quand l'entropie de la distribution est inférieure à un seuil.

    threshold : 0 = certitude totale ;
    normalize : si True, l'entropie est divisée par ln(V) et le seuil est dans [0, 1],
                ce qui le rend indépendant de la taille du vocabulaire.
    min_layer : aucune sortie avant cette couche.
    patience  : nombre de couches consécutives sous le seuil exigé avant de sortir.
    verbose   : si True, affiche l'entropie à chaque couche (y compris avant min_layer).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_layer: int = 0,
        patience: int = 1,
        normalize: bool = True,
        verbose: bool = False,
    ):
        if threshold <= 0:
            raise ValueError("threshold doit être strictement positif.")
        if min_layer < 0:
            raise ValueError("min_layer doit être positif ou nul.")
        if patience < 1:
            raise ValueError("patience doit être supérieur ou égal à 1.")

        self.threshold = threshold
        self.min_layer = min_layer
        self.patience = patience
        self.normalize = normalize
        self.verbose = verbose

    def entropy(self, logits: torch.Tensor) -> float:
        # log_softmax est plus stable numériquement que softmax().log()
        log_probs = logits[:, -1, :].float().log_softmax(dim=-1)
        probs = log_probs.exp()
        h = -(probs * log_probs).sum(dim=-1)  # [B]

        if self.normalize:
            h = h / math.log(logits.shape[-1])

        return h.max().item()  # max sur le batch : prudent si B > 1

    def first_layer(self, n_layers: int) -> int:
        # avant min_layer, should_exit ne fait que remettre le compteur à zéro
        return self.min_layer

    def should_exit(self, layer_idx: int, logits: torch.Tensor, state: dict) -> bool:
        h = self.entropy(logits)
        state["last_entropy"] = h  # utile pour le debug

        if layer_idx < self.min_layer:
            state["low_entropy_count"] = 0
            if self.verbose:
                print(f"  L{layer_idx:02d} | H={h:7.4f} | (avant min_layer={self.min_layer})")
            return False

        if h <= self.threshold:
            state["low_entropy_count"] = state.get("low_entropy_count", 0) + 1
        else:
            state["low_entropy_count"] = 0

        if self.verbose:
            print(
                f"  L{layer_idx:02d} | H={h:7.4f} | seuil={self.threshold} "
                f"| count={state['low_entropy_count']}/{self.patience}"
            )

        return state["low_entropy_count"] >= self.patience
import torch
import torch.nn.functional as F

from early_exit.strategies.EarlyExitStrategy import EarlyExitStrategy


class CosineExitStrategy(EarlyExitStrategy):
    """Sort quand le hidden state de la dernière position cesse d'évoluer :
    cos(h_i, h_{i-1}) >= threshold pendant `patience` couches consécutives.

    Lit le hidden state dans state["hidden"] (à renseigner dans la boucle).
    Si absent, se rabat sur la similarité entre logits de couches consécutives.
    """

    def __init__(
        self,
        threshold: float = 0.99,
        min_layer: int = 0,
        patience: int = 1,
        verbose: bool = False,
    ):
        if not -1 < threshold <= 1:
            raise ValueError("threshold doit être dans ]-1, 1].")
        if min_layer < 0:
            raise ValueError("min_layer doit être positif ou nul.")
        if patience < 1:
            raise ValueError("patience doit être supérieur ou égal à 1.")

        self.threshold = threshold
        self.min_layer = min_layer
        self.patience = patience
        self.verbose = verbose

    # travaille sur state["hidden"] : les logits ne servent qu'en repli
    needs_logits = False

    def first_layer(self, n_layers: int) -> int:
        # la couche min_layer - 1 fournit le vecteur précédent
        return max(self.min_layer - 1, 0)

    def should_exit(self, layer_idx: int, logits: torch.Tensor, state: dict) -> bool:
        # Vecteur courant : hidden state si fourni, sinon logits
        current = state.get("hidden")
        if current is None:
            current = logits[:, -1, :]
        current = current.float()

        previous = state.get("previous_vector")
        state["previous_vector"] = current  # toujours mis à jour, même avant min_layer

        if previous is None or layer_idx < self.min_layer:
            state["stable_count"] = 0
            return False

        sim = F.cosine_similarity(current, previous, dim=-1).min().item()  # min sur le batch
        state["last_similarity"] = sim

        if sim >= self.threshold:
            state["stable_count"] = state.get("stable_count", 0) + 1
        else:
            state["stable_count"] = 0

        if self.verbose:
            print(
                f"  L{layer_idx:02d} | cos={sim:.4f} | seuil={self.threshold} "
                f"| count={state['stable_count']}/{self.patience}"
            )

        return state["stable_count"] >= self.patience
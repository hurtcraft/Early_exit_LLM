from strategy.EarlyExitStrategy import EarlyExitStrategy
import torch
import torch

from strategy.EarlyExitStrategy import EarlyExitStrategy
import torch

from strategy.EarlyExitStrategy import EarlyExitStrategy


class StableTokenStrategy(EarlyExitStrategy):
    """Sort si le token top-1 est stable ET confiant.

    patience  : nombre de couches consécutives stables et confiantes exigé.
    min_layer : aucune sortie avant cette couche.
    threshold : probabilité minimale du top-1 pour compter une couche comme stable.
    tokenizer : optionnel, permet d'afficher le token décodé en plus de son id.
    verbose   : si True, affiche le top-1 et son état à chaque couche.
    """

    def __init__(
        self,
        patience: int = 3,
        min_layer: int = 5,
        threshold: float = 0.7,
        tokenizer=None,
        verbose: bool = False,
    ):
        if patience < 1:
            raise ValueError("patience doit être supérieur ou égal à 1.")
        if min_layer < 0:
            raise ValueError("min_layer doit être positif ou nul.")
        if not 0 < threshold <= 1:
            raise ValueError("threshold doit être dans ]0, 1].")

        self.patience = patience
        self.min_layer = min_layer
        self.threshold = threshold
        self.tokenizer = tokenizer
        self.verbose = verbose

    def _label(self, token_id: int) -> str:
        if self.tokenizer is None:
            return str(token_id)
        return f"{self.tokenizer.decode([token_id])!r} ({token_id})"

    def first_layer(self, n_layers: int) -> int:
        # la couche min_layer - 1 fournit le token précédent
        return max(self.min_layer - 1, 0)

    def should_exit(self, layer_idx, logits, state) -> bool:
        probs = logits[:, -1, :].float().softmax(dim=-1)
        max_prob, current_token = probs.max(dim=-1)

        if layer_idx < self.min_layer:
            state["previous_token"] = current_token
            state["stable_count"] = 0
            if self.verbose:
                print(
                    f"  L{layer_idx:02d} | top={self._label(current_token.item())} "
                    f"p={max_prob.item():6.2%} | (avant min_layer={self.min_layer})"
                )
            return False

        previous_token = state.get("previous_token")
        state["previous_token"] = current_token

        # Stable ET confiant, en une seule synchronisation GPU
        stable = previous_token is not None and bool(
            ((current_token == previous_token) & (max_prob >= self.threshold)).all()
        )

        # sinon on remet le compteur à zéro
        if stable:
            state["stable_count"] = state.get("stable_count", 0) + 1
        else:
            state["stable_count"] = 0

        if self.verbose:
            same = previous_token is not None and torch.equal(current_token, previous_token)
            confident = max_prob.item() >= self.threshold
            print(
                f"  L{layer_idx:02d} | top={self._label(current_token.item())} "
                f"p={max_prob.item():6.2%} | stable={'oui' if same else 'non'} "
                f"confiant={'oui' if confident else 'non'} (seuil={self.threshold}) "
                f"| count={state['stable_count']}/{self.patience}"
            )

        return state["stable_count"] >= self.patience
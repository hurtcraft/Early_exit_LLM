from early_exit.strategies.EarlyExitStrategy import EarlyExitStrategy


class AlwaysExitStrategy(EarlyExitStrategy):
    needs_logits = False

    def __init__(self, layer: int):
        self.layer = layer

    def first_layer(self, n_layers):
        return self.layer

    def should_exit(self, layer_idx, logits, state):
        return layer_idx >= self.layer
import torch
from lm_eval.models.huggingface import HFLM
from tqdm import tqdm

from early_exit.generate import _eos_ids
from early_exit.layers import _layer_trace, _run_layers


class EarlyExitHFLM(HFLM):
    """HFLM dont le forward utilise un adaptateur + une stratégie d'early exit.

    - loglikelihood (MMLU…) : _model_call, décision d'early exit sur la dernière position.
    - generate_until (GSM8K…) : _model_generate, génération gloutonne token par token,
      une décision d'early exit par token.

    trace_cache : dict partagé entre plusieurs EarlyExitHFLM (une par stratégie) pour
    réutiliser les forwards d'une même séquence (cf. CpuTraceCache).
    """

    def __init__(self, adapter, strategy, trace_cache: dict | None = None, **kwargs):
        super().__init__(
            pretrained=adapter.model,
            tokenizer=adapter.tokenizer,
            **kwargs,
        )
        self.adapter = adapter
        self.strategy = strategy
        self.trace_cache = trace_cache
        self.exit_layers: list[int] = []

    @torch.inference_mode()
    def _model_call(self, inps, attn_mask=None, labels=None):
        trace = _layer_trace(self.adapter, inps, self.trace_cache)
        last_logits, exit_layer, _ = _run_layers(self.adapter, self.strategy, trace)
        self.exit_layers.append(exit_layer)

        # Seule la dernière position est calculée : valable pour les réponses
        # d'un seul token (MMLU : " A"), lm_eval ne lit alors que cette position.
        B, L = inps.shape
        logits = last_logits.new_zeros(B, L, last_logits.shape[-1])
        logits[:, -1, :] = last_logits[:, 0, :]
        return logits

    @torch.inference_mode()
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        """Génération gloutonne avec early exit. Retourne contexte + tokens générés ;
        lm_eval retire le contexte et coupe lui-même sur les séquences `stop`."""
        input_ids = context
        start = context.shape[1]
        eos = _eos_ids(self.adapter)
        tok = self.adapter.tokenizer

        # une question peut prendre plusieurs minutes : barre par token générée
        with tqdm(total=max_length - start, desc="tokens", unit="tok", leave=False) as bar:
            while input_ids.shape[1] < max_length:
                trace = _layer_trace(self.adapter, input_ids, self.trace_cache)
                logits, exit_layer, _ = _run_layers(self.adapter, self.strategy, trace)
                self.exit_layers.append(exit_layer)

                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_id], dim=-1)
                bar.update(1)

                if next_id.item() in eos:
                    break
                # arrêt dès qu'une séquence stop apparaît : inutile de générer la suite
                if any(s in tok.decode(input_ids[0, start:]) for s in stop):
                    break

        return input_ids

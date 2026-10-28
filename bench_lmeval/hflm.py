import torch
from lm_eval.models.huggingface import HFLM
from tqdm import tqdm

from early_exit.generate import _eos_ids
from early_exit.layers import LayerTracer, _layer_trace, _run_layers


class EarlyExitHFLM(HFLM):
    """HFLM dont le forward utilise un adaptateur + une stratégie d'early exit.

    - loglikelihood (MMLU…) : _model_call, décision d'early exit sur la dernière position.
    - generate_until (GSM8K…) : _model_generate, génération gloutonne token par token
      avec KV cache (LayerTracer), une décision d'early exit par token.

    trace_cache : dict partagé entre plusieurs EarlyExitHFLM (une par stratégie) pour
    réutiliser les forwards d'une même séquence (cf. CpuTraceCache). Utilisé
    seulement par loglikelihood : la génération a son propre KV cache.
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
        lm_eval retire le contexte et coupe lui-même sur les séquences `stop`.

        KV cache : le prompt passe une fois dans les couches (prefill), puis chaque
        token généré seul."""
        start = context.shape[1]
        eos = _eos_ids(self.adapter)
        tok = self.adapter.tokenizer
        # une séquence stop qui vient d'apparaître finit dans les derniers tokens :
        # inutile de décoder tout le texte généré à chaque pas
        tail = max((len(s) for s in stop), default=0) + 1

        tracer = LayerTracer(self.adapter)
        trace = tracer.step(context)
        generated = []

        # une question peut prendre plusieurs minutes : barre par token générée
        with tqdm(total=max_length - start, desc="tokens", unit="tok", leave=False) as bar:
            while start + len(generated) < max_length:
                logits, exit_layer, _ = _run_layers(self.adapter, self.strategy, trace)
                self.exit_layers.append(exit_layer)

                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(next_id)
                bar.update(1)

                if next_id.item() in eos:
                    break
                # arrêt dès qu'une séquence stop apparaît : inutile de générer la suite
                recent = tok.decode(torch.cat(generated[-tail:], dim=-1)[0])
                if any(s in recent for s in stop):
                    break

                trace = tracer.step(next_id)

        return torch.cat([context, *generated], dim=-1)

import torch
import torch.nn.functional as F
from lm_eval.models.huggingface import HFLM
from tqdm import tqdm

from early_exit.generate import _eos_ids
from early_exit.layers import LayerTracer, _layer_trace, _run_layers, _run_layers_rows


class EarlyExitHFLM(HFLM):
    """HFLM dont le forward utilise un adaptateur + une stratégie d'early exit.

    - loglikelihood et loglikelihood_rolling : _loglikelihood_tokens (remplace celui
      de HFLM). Réponse d'un seul token (MMLU : " A") : une décision sur la dernière
      position, chemin d'origine. Réponse de plusieurs tokens (HellaSwag, ARC…) ou
      perplexité (WikiText) : une décision indépendante par token de la réponse.
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
        # Plus utilisé par loglikelihood (cf. _loglikelihood_tokens) ; gardé au cas où
        # HFLM l'appellerait ailleurs.
        trace = _layer_trace(self.adapter, inps, self.trace_cache)
        last_logits, exit_layer, _ = _run_layers(self.adapter, self.strategy, trace)
        self.exit_layers.append(exit_layer)

        # Seule la dernière position est calculée : valable pour les réponses
        # d'un seul token (MMLU : " A"), lm_eval ne lit alors que cette position.
        B, L = inps.shape
        logits = last_logits.new_zeros(B, L, last_logits.shape[-1])
        logits[:, -1, :] = last_logits[:, 0, :]
        return logits

    # ---------- loglikelihood (QCM, perplexité) ----------

    @property
    def _embed_device(self):
        return self.adapter.model.get_input_embeddings().weight.device

    @torch.inference_mode()
    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, override_bs=None, **kwargs):
        """Remplace HFLM._loglikelihood_tokens (appelé par loglikelihood et
        loglikelihood_rolling). requests : [(request_str, context_enc, continuation_enc)].
        Retourne [(somme des log-probas de la continuation, continuation = top-1 partout)].

        Même découpage que HFLM : entrée = (contexte + continuation)[-(max_length+1):][:-1],
        les len(continuation) dernières positions prédisent la continuation.
        """
        res = []
        memo_key, memo_lp = None, None   # réponses d'un token partageant le même contexte

        for request_str, ctx_enc, cont_enc in tqdm(requests, disable=disable_tqdm,
                                                    desc="loglikelihood", leave=False):
            inp = (ctx_enc + cont_enc)[-(self.max_length + 1):][:-1]
            n = min(len(cont_enc), len(inp))
            cont = cont_enc[-n:]

            if n == 1:
                # Chemin d'origine (MMLU) : une décision sur la dernière position. Les choix
                # d'une même question ont la même entrée : une seule décision, comme quand
                # lm_eval regroupait ces requêtes en un seul appel à _model_call.
                key = tuple(inp)
                if key != memo_key:
                    memo_key, memo_lp = key, self._last_position_logprobs(inp)
                answer = (float(memo_lp[cont[0]]), bool(memo_lp.argmax() == cont[0]))
            else:
                answer = self._continuation_logprob(inp, cont)

            res.append(answer)
            if request_str is not None:
                self.cache_hook.add_partial("loglikelihood", request_str, answer)
        return res

    def _last_position_logprobs(self, inp: list[int]) -> torch.Tensor:
        """Log-probas [V] de la dernière position, une décision d'early exit."""
        ids = torch.tensor([inp], dtype=torch.long, device=self._embed_device)
        trace = _layer_trace(self.adapter, ids, self.trace_cache)
        logits, exit_layer, _ = _run_layers(self.adapter, self.strategy, trace)
        self.exit_layers.append(exit_layer)
        # log_softmax dans le dtype du modèle, comme HFLM : résultats MMLU inchangés
        return F.log_softmax(logits[0, -1], dim=-1)

    def _continuation_logprob(self, inp: list[int], cont: list[int]) -> tuple[float, bool]:
        """Une décision d'early exit indépendante par token de la continuation."""
        ids = torch.tensor([inp], dtype=torch.long, device=self._embed_device)
        trace = _layer_trace(self.adapter, ids, self.trace_cache, keep=len(cont))  # [n_layers, n, H]
        targets = torch.tensor(cont, dtype=torch.long, device=self._embed_device)
        logprob, greedy, exits = _run_layers_rows(self.adapter, self.strategy, trace, targets)
        self.exit_layers.extend(exits.tolist())
        return float(logprob.sum()), bool(greedy.all())

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

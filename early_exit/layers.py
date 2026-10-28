"""Moteur de l'early exit, commun aux deux benchmarks (prompts custom et lm_eval).

- _layer_trace : forward complet de la séquence, couche par couche, et garde le
  hidden state de la dernière position à chaque couche (partageable via un cache).
- LayerTracer  : même trace, mais avec KV cache, pour la génération : chaque appel
  ne calcule que les nouveaux tokens.
- _run_layers  : rejoue une stratégie couche par couche sur cette trace.
"""
import torch
from transformers import DynamicCache


def _is_linear_layer(text_model, i: int) -> bool:
    """Indique si la couche i est une couche d'attention linéaire (DeltaNet)."""
    layer = text_model.layers[i]

    # 1) Structure de la couche (le plus fiable)
    if hasattr(layer, "linear_attn"):
        return True
    if hasattr(layer, "self_attn"):
        return False

    # 2) Repli sur la config
    layer_types = getattr(text_model.config, "layer_types", None)
    if layer_types is not None:
        return layer_types[i] == "linear_attention"

    raise RuntimeError(f"Type de la couche {i} indéterminé")


def _build_layer_inputs(adapter, input_ids: torch.Tensor, offset: int = 0):
    """Embeddings, position_ids, position_embeddings (mRoPE) et masque causal.

    offset : nombre de tokens déjà dans le KV cache (positions offset .. offset+L-1).
    """
    tm = adapter.model.model
    B, L = input_ids.shape
    hidden = adapter.get_embeddings(input_ids)

    pos = torch.arange(offset, offset + L, device=input_ids.device).unsqueeze(0).expand(B, -1)
    pe = tm.rotary_emb(hidden, pos.unsqueeze(0).expand(3, -1, -1))

    # Un seul nouveau token voit tout le passé : pas de masque.
    # Sinon [L, offset+L] : le token i voit les clés j <= offset + i.
    mask = None
    if L > 1:
        mask = torch.full(
            (L, offset + L), torch.finfo(hidden.dtype).min,
            device=input_ids.device, dtype=hidden.dtype,
        ).triu(offset + 1)[None, None]

    return hidden, pos, pe, mask


def _forward_trace(adapter, input_ids: torch.Tensor, offset: int = 0, cache=None) -> torch.Tensor:
    """Exécute toutes les couches sur input_ids et garde le hidden state de la
    dernière position à chaque couche : [n_layers, B, H].
    cache : DynamicCache lu et mis à jour par les couches (None = sans cache)."""
    tm = adapter.model.model
    hidden, pos, pe, mask = _build_layer_inputs(adapter, input_ids, offset)

    lasts = []
    for i in range(adapter.num_layers):
        layer_mask = None if _is_linear_layer(tm, i) else mask
        hidden, _ = adapter.forward_layer(
            i, hidden,
            attention_mask=layer_mask,
            position_ids=pos,
            position_embeddings=pe,
            past_key_values=cache,
        )
        lasts.append(hidden[:, -1, :])

    return torch.stack(lasts)


# ---------- Forward avec early exit ----------

class CpuTraceCache(dict):
    """Cache de traces stocké en RAM (CPU) : pour les grands caches partagés entre
    stratégies sur un benchmark entier, qui ne tiendraient pas en VRAM."""

    def __init__(self, device):
        super().__init__()
        self.device = device

    def __setitem__(self, key, trace):
        super().__setitem__(key, trace.to("cpu"))

    def __getitem__(self, key):
        return super().__getitem__(key).to(self.device)


def _layer_trace(adapter, input_ids: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
    """Hidden state de la dernière position à chaque couche : [n_layers, B, H].

    Toutes les couches sont exécutées sur toute la séquence. Le résultat ne dépend
    que de input_ids (pas de la stratégie) : `cache` le partage entre stratégies.
    """
    key = tuple(input_ids[0].tolist())
    if cache is not None and key in cache:
        return cache[key]

    trace = _forward_trace(adapter, input_ids)
    if cache is not None:
        cache[key] = trace
    return trace


class LayerTracer:
    """Trace couche par couche avec KV cache, pour la génération token par token.

    Le premier appel (prefill) traite tout le prompt ; les suivants ne traitent que
    le token ajouté. Toutes les couches restent calculées pour chaque token : la
    trace est la même que _layer_trace sur la séquence complète (aux arrondis bf16
    près), seul le recalcul du passé disparaît.

    Le cache de Qwen3.5 est hybride : K/V pour les couches d'attention complète,
    états récurrent + convolution pour les couches DeltaNet. Les couches le
    mettent à jour elles-mêmes via past_key_values.
    """

    def __init__(self, adapter):
        self.adapter = adapter
        self.cache = DynamicCache(config=adapter.model.config)
        self.seen = 0   # tokens déjà dans le cache

    def step(self, new_ids: torch.Tensor) -> torch.Tensor:
        """new_ids : [B, n] nouveaux tokens. Retourne la trace [n_layers, B, H]
        de la dernière position."""
        trace = _forward_trace(self.adapter, new_ids, offset=self.seen, cache=self.cache)
        self.seen += new_ids.shape[1]
        return trace


def _run_layers(adapter, strategy, trace: torch.Tensor, verbose: bool = False):
    """Rejoue la stratégie couche par couche sur la trace.
    Retourne (logits [B, 1, V], couche de sortie, lignes d'affichage verbose)."""
    n_layers = trace.shape[0]

    # Avant `start`, should_exit n'a aucun effet : inutile de l'appeler.
    # En verbose, toutes les couches sont projetées pour être affichées.
    start = 0 if verbose else min(strategy.first_layer(n_layers), n_layers)
    project = verbose or strategy.needs_logits

    # Une seule projection pour toutes les couches : les poids du lm_head
    # (la plus grosse matrice du modèle) ne sont lus qu'une fois.
    all_logits = adapter.get_logits(trace[start:]) if project and start < n_layers else None

    def layer_logits(i):
        if all_logits is not None and i >= start:
            return all_logits[i - start][:, None, :]
        return adapter.get_logits(trace[i][:, None, :])

    state = {}  # remis à zéro à chaque token
    exit_layer = n_layers - 1
    lines = []

    for i in range(start, n_layers):
        logits = layer_logits(i) if project else None
        state["hidden"] = trace[i]
        exit_now = strategy.should_exit(i, logits, state)

        if verbose:
            lines.append(
                f"  L{i:02d} | {_fmt_top_k(adapter.tokenizer, logits)}"
            )
        if exit_now:
            exit_layer = i
            break

    if lines:
        lines[-1] += "  [SORTIE]"
    return layer_logits(exit_layer), exit_layer, lines


# ---------- Lignes du mode verbose ----------

def _short(text: str, width: int = 12) -> str:
    """repr du token, tronqué et aligné sur `width` caractères."""
    s = repr(text)
    if len(s) > width:
        s = s[:width - 2] + ".."
    return f"{s:<{width}}"


def _fmt_top_k(tok, logits: torch.Tensor, k: int = 3) -> str:
    probs = logits[0, -1].float().softmax(-1)
    top_p, top_i = probs.topk(k)
    return " | ".join(
        f"{_short(tok.decode([i]))} {p:6.1%}"
        for p, i in zip(top_p.tolist(), top_i.tolist())
    )

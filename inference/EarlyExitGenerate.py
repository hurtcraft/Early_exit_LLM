import torch
import time
from tqdm import tqdm

from inference.EarlyExitHFLM import _is_linear_layer




# ---------- Préparation ----------

def _encode_prompt(adapter, prompt: str, use_chat_template: bool) -> torch.Tensor:
    if use_chat_template:
        return adapter.prepare_inputs(prompt)["input_ids"]
    device = adapter.model.get_input_embeddings().weight.device
    return adapter.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)


def _eos_ids(adapter) -> set[int]:
    eos = adapter.model.generation_config.eos_token_id
    return set(eos) if isinstance(eos, (list, tuple)) else {eos}


def _build_layer_inputs(adapter, input_ids: torch.Tensor):
    """Embeddings, position_ids, position_embeddings (mRoPE) et masque causal."""
    tm = adapter.model.model
    B, L = input_ids.shape
    hidden = adapter.get_embeddings(input_ids)

    pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
    pe = tm.rotary_emb(hidden, pos.unsqueeze(0).expand(3, -1, -1))
    mask = torch.full(
        (L, L), torch.finfo(hidden.dtype).min,
        device=input_ids.device, dtype=hidden.dtype,
    ).triu(1)[None, None]

    return hidden, pos, pe, mask


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

    tm = adapter.model.model
    hidden, pos, pe, mask = _build_layer_inputs(adapter, input_ids)

    lasts = []
    for i in range(adapter.num_layers):
        layer_mask = None if _is_linear_layer(tm, i) else mask
        hidden, _ = adapter.forward_layer(
            i, hidden,
            attention_mask=layer_mask,
            position_ids=pos,
            position_embeddings=pe,
        )
        lasts.append(hidden[:, -1, :])

    trace = torch.stack(lasts)
    if cache is not None:
        cache[key] = trace
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


# ---------- Affichage verbose ----------

SEP = "-----------------------"   # entre deux tokens


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


def _print_token(step: int, max_new_tokens: int, token: str, prob: float,
                 exit_layer: int, n_layers: int, lines: list) -> None:
    tqdm.write("\n".join([
        f"TOKEN {step + 1}/{max_new_tokens} | {token!r} {prob:.1%} "
        f"| sortie couche {exit_layer}/{n_layers - 1}",
        *lines,
        SEP,
    ]))


def _pick_token(logits: torch.Tensor):
    """Token le plus probable et sa probabilité."""
    last = logits[:, -1, :]
    next_id = last.argmax(dim=-1, keepdim=True)
    prob = last.float().softmax(dim=-1).gather(-1, next_id).item()
    return next_id, prob


# ---------- Métriques ----------



def _strategy_params(strategy) -> dict:
    """Hyperparamètres de la stratégie (attributs simples, sans tokenizer)."""
    params = {
        k: v for k, v in vars(strategy).items()
        if isinstance(v, (int, float, str, bool)) and not k.startswith("_")
    }
    return {"name": type(strategy).__name__, **params}


def _compute_metrics(prompt, generated, token_ids, exit_layers, probs, n_layers, stopped_by_eos) -> dict:
    exits = torch.tensor(exit_layers, dtype=torch.float)
    last = n_layers - 1
    std = exits.std(unbiased=False).item()

    return {
        "prompt": prompt,
        "answer": "".join(generated),
        "tokens": token_ids,
        "n_tokens": len(generated),
        "stopped_by_eos": stopped_by_eos,
        "exit_layers": exit_layers,
        "mean_exit_layer": exits.mean().item(),
        "std_exit_layer": std,
        "mean_depth_ratio": exits.mean().item() / last,
        "early_exit_rate": (exits < last).float().mean().item(),
        "mean_prob": sum(probs) / len(probs),
        "min_prob": min(probs),
    }
def _print_metrics(m: dict) -> None:
    s = m["strategy"]
    params = ", ".join(f"{k}={v}" for k, v in s.items() if k != "name")

    print("\n" + "=" * 60)
    print(f"Prompt                : {m['prompt']}")
    print(f"Modèle                : {m['model']} ({m['quant']}) | {m['n_layers']} couches")
    print(f"Stratégie             : {s['name']}({params})")
    print(f"Chat template         : {m['use_chat_template']} | prompt = {m['prompt_tokens']} tokens")
    print("-" * 60)
    print(f"Réponse               : {m['answer']}")
    print(f"Tokens générés        : {m['n_tokens']} / {m['max_new_tokens']}"
          f"{' (EOS)' if m['stopped_by_eos'] else ''}")
    print(f"Couche moyenne        : {m['mean_exit_layer']:.2f} ± {m['std_exit_layer']:.2f}")
    print(f"Profondeur moyenne    : {m['mean_depth_ratio']:.1%} ± {m['std_depth_ratio']:.1%}")
    print(f"Min / Max             : {m['min_exit_layer']} / {m['max_exit_layer']}")
    print(f"Tokens sortis tôt     : {m['early_exit_rate']:.0%}")
    print(f"Confiance moy. / min  : {m['mean_prob']:.1%} / {m['min_prob']:.1%}")
    print(f"Durée                 : {m['duration_s']:.1f}s")

# ---------- Point d'entrée ----------

@torch.inference_mode()
def early_exit_generate(
    adapter,
    strategy,
    prompt: str,
    max_new_tokens: int = 20,
    use_chat_template: bool = False,
    model_name: str | None = None,
    quant: str | None = None,
    verbose: bool = False,
    trace_cache: dict | None = None,
) -> dict:
    """verbose     : affiche chaque token et le top-k de chaque couche (projette toutes les couches).
    trace_cache : dict partagé entre appels pour réutiliser les forwards d'un même préfixe."""
    start = time.perf_counter()
    tok = adapter.tokenizer
    n_layers = adapter.num_layers
    input_ids = _encode_prompt(adapter, prompt, use_chat_template)
    prompt_tokens = input_ids.shape[1]
    eos = _eos_ids(adapter)

    generated, token_ids, exit_layers, probs = [], [], [], []
    stopped_by_eos = False

    for step in range(max_new_tokens):
        trace = _layer_trace(adapter, input_ids, trace_cache)
        logits, exit_layer, lines = _run_layers(adapter, strategy, trace, verbose)
        next_id, prob = _pick_token(logits)
        token_id = next_id.item()

        token = tok.decode(next_id[0], clean_up_tokenization_spaces=False)
        generated.append(token)
        token_ids.append(token_id)
        exit_layers.append(exit_layer)
        probs.append(prob)

        if verbose:
            _print_token(step, max_new_tokens, token, prob, exit_layer, n_layers, lines)

        input_ids = torch.cat([input_ids, next_id], dim=-1)
        if token_id in eos:
            stopped_by_eos = True
            break


    metrics = _compute_metrics(prompt, generated, token_ids, exit_layers, probs, n_layers, stopped_by_eos)

    return metrics

def print_top_k(tok, logits, layer_idx, k=5):
    probs = logits[0, -1].float().softmax(-1)
    top_p, top_i = probs.topk(k)
    items = " | ".join(
        f"{tok.decode([i])!r}: {p * 100:6.2f}%"
        for p, i in zip(top_p.tolist(), top_i.tolist())
    )
    tqdm.write(f"Layer {layer_idx:02d} | {items}")






"""
@torch.no_grad()
def early_exit_generate(
    adapter,
    strategy,
    prompt: str,
    max_new_tokens: int = 20,
    use_chat_template: bool = False,
) -> dict:
    tok = adapter.tokenizer
    tm = adapter.model.model
    n_layers = adapter.num_layers

    if use_chat_template:
        input_ids = adapter.prepare_inputs(prompt)["input_ids"]
    else:
        device = adapter.model.get_input_embeddings().weight.device
        input_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)

    eos = adapter.model.generation_config.eos_token_id
    eos = set(eos) if isinstance(eos, (list, tuple)) else {eos}

    generated = []
    exit_layers = []   # <- couche de sortie de chaque token
    probs = []         # <- probabilité du token choisi

    for step in range(max_new_tokens):
        B, L = input_ids.shape
        hidden = adapter.get_embeddings(input_ids)

        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        pe = tm.rotary_emb(hidden, pos.unsqueeze(0).expand(3, -1, -1))
        mask = torch.full(
            (L, L), torch.finfo(hidden.dtype).min,
            device=input_ids.device, dtype=hidden.dtype,
        ).triu(1)[None, None]

        state = {}
        exit_layer = n_layers - 1

        for i in range(n_layers):
            layer_mask = None if _is_linear_layer(tm, i) else mask
            hidden, _ = adapter.forward_layer(
                i, hidden,
                attention_mask=layer_mask,
                position_ids=pos,
                position_embeddings=pe,
            )
            logits = adapter.get_logits(hidden[:, -1:, :])
            state["hidden"] = hidden[:, -1, :]

            if strategy.should_exit(i, logits, state):
                exit_layer = i
                break

        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        prob = logits[:, -1, :].float().softmax(dim=-1).gather(-1, next_id).item()

        token = tok.decode(next_id[0], clean_up_tokenization_spaces=False)
        generated.append(token)
        exit_layers.append(exit_layer)
        probs.append(prob)

        print(
            f"Token {step + 1:>2} | {token!r:<15} | "
            f"p={prob * 100:6.2f}% | sortie couche {exit_layer}/{n_layers - 1}"
        )

        input_ids = torch.cat([input_ids, next_id], dim=-1)
        if next_id.item() in eos:
            break

    # ---------- Métriques ----------
    exits = torch.tensor(exit_layers, dtype=torch.float)
    last = n_layers - 1
    # unbiased=False : écart type de la population (les tokens générés sont tout l'échantillon)
    std = exits.std(unbiased=False).item()

    metrics = {
        "answer": "".join(generated),
        "n_tokens": len(generated),
        "exit_layers": exit_layers,
        "mean_exit_layer": exits.mean().item(),
        "std_exit_layer": std,
        "min_exit_layer": int(exits.min().item()),
        "max_exit_layer": int(exits.max().item()),
        "mean_depth_ratio": exits.mean().item() / last,
        "std_depth_ratio": std / last,
        "early_exit_rate": (exits < last).float().mean().item(),
        "mean_prob": sum(probs) / len(probs),
    }

    print("\nRéponse :", metrics["answer"])
    print("-" * 60)
    print(f"Tokens générés        : {metrics['n_tokens']}")
    print(f"Couche moyenne        : {metrics['mean_exit_layer']:.2f} ± {metrics['std_exit_layer']:.2f}")
    print(f"Profondeur moyenne    : {metrics['mean_depth_ratio']:.1%} ± {metrics['std_depth_ratio']:.1%}")
    print(f"Min / Max             : {metrics['min_exit_layer']} / {metrics['max_exit_layer']}")
    print(f"Tokens sortis tôt     : {metrics['early_exit_rate']:.0%}")
    print(f"Confiance moyenne     : {metrics['mean_prob']:.1%}")
    return metrics
"""
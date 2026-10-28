"""Génération gloutonne token par token avec early exit (utilisée par bench_prompts)."""
import torch
from tqdm import tqdm

from early_exit.layers import _layer_trace, _run_layers


# ---------- Préparation ----------

def _encode_prompt(adapter, prompt: str, use_chat_template: bool) -> torch.Tensor:
    if use_chat_template:
        return adapter.prepare_inputs(prompt)["input_ids"]
    device = adapter.model.get_input_embeddings().weight.device
    return adapter.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)


def _eos_ids(adapter) -> set[int]:
    eos = adapter.model.generation_config.eos_token_id
    return set(eos) if isinstance(eos, (list, tuple)) else {eos}


def _pick_token(logits: torch.Tensor):
    """Token le plus probable et sa probabilité."""
    last = logits[:, -1, :]
    next_id = last.argmax(dim=-1, keepdim=True)
    prob = last.float().softmax(dim=-1).gather(-1, next_id).item()
    return next_id, prob


# ---------- Affichage verbose ----------

SEP = "-----------------------"   # entre deux tokens


def _print_token(step: int, max_new_tokens: int, token: str, prob: float,
                 exit_layer: int, n_layers: int, lines: list) -> None:
    tqdm.write("\n".join([
        f"TOKEN {step + 1}/{max_new_tokens} | {token!r} {prob:.1%} "
        f"| sortie couche {exit_layer}/{n_layers - 1}",
        *lines,
        SEP,
    ]))


# ---------- Métriques ----------

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


# ---------- Point d'entrée ----------

@torch.inference_mode()
def early_exit_generate(
    adapter,
    strategy,
    prompt: str,
    max_new_tokens: int = 20,
    use_chat_template: bool = False,
    verbose: bool = False,
    trace_cache: dict | None = None,
) -> dict:
    """verbose     : affiche chaque token et le top-k de chaque couche (projette toutes les couches).
    trace_cache : dict partagé entre appels pour réutiliser les forwards d'un même préfixe."""
    tok = adapter.tokenizer
    n_layers = adapter.num_layers
    input_ids = _encode_prompt(adapter, prompt, use_chat_template)
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

    return _compute_metrics(prompt, generated, token_ids, exit_layers, probs, n_layers, stopped_by_eos)

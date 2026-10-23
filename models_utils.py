from pathlib import Path
import torch
from inference.EarlyExitGenerate import _strategy_params
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
from datetime import datetime
from pathlib import Path

MODELS_DIR = Path("models")

def build_run_config(adapter, strategy, model_name, quant, use_chat_template, max_new_tokens) -> dict:
    return {
        "model": model_name,
        "quant": quant,
        "n_layers": adapter.num_layers,
        "strategy": _strategy_params(strategy),
        "use_chat_template": use_chat_template,
        "max_new_tokens": max_new_tokens,
    }
def summarize_run(results: list[dict], n_layers: int) -> dict:
    all_exits = torch.tensor([l for r in results for l in r["exit_layers"]], dtype=torch.float)
    last = n_layers - 1
    return {
        "n_questions": len(results),
        "n_tokens": len(all_exits),
        "mean_exit_layer": all_exits.mean().item(),
        "std_exit_layer": all_exits.std(unbiased=False).item(),
        "mean_depth_ratio": all_exits.mean().item() / last,
        "early_exit_rate": (all_exits < last).float().mean().item(),
    } 

def load_model(name: str, quant: str | None = None):
    """quant : None (fp16/bf16), '8bit' ou '4bit'."""
    path = MODELS_DIR / name
    tokenizer = AutoTokenizer.from_pretrained(path)

    kwargs = {"device_map": "auto"}
    if quant == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    elif quant is None:
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        raise ValueError(f"quant inconnu : {quant}")

    model = AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()
    return model, tokenizer



RESULTS_DIR = Path("results")


def save_results(
    model_name: str,
    results: dict,
    quant: str | None = None,
):
    """Enregistre les résultats des benchmarks dans un fichier JSON."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "model": model_name,
        "quantization": quant or "native",
        "date": datetime.now().isoformat(timespec="seconds"),
        "benchmarks": results,
    }

    safe_name = model_name.replace("/", "_").replace("\\", "_")
    quant_name = quant or "native"

    filename = f"{safe_name}_{quant_name}.json"
    output_path = RESULTS_DIR / filename

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    print(f"Résultats sauvegardés dans : {output_path}")

    return output_path

import torch
import torch.nn.functional as F
"""
def show_probability_distribution(
    model,
    tokenizer,
    question: str,
    top_k: int = 5,
):
    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states

    print()
    print("=" * 100)
    print("Question :", question)
    print("=" * 100)

    for layer_idx, hidden_state in enumerate(hidden_states):

        # Dernier token
        last_hidden = hidden_state[:, -1, :]

        # IMPORTANT :
        # appliquer la normalisation finale du modèle
        normalized_hidden = model.model.norm(last_hidden)

        # Projection vers le vocabulaire
        logits = model.lm_head(normalized_hidden)

        probabilities = torch.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(
            probabilities,
            k=top_k,
            dim=-1
        )

        tokens = []

        for prob, token_id in zip(
            top_probs[0],
            top_indices[0]
        ):

            token = tokenizer.decode(
                [token_id.item()]
            )

            tokens.append(
                f"{repr(token)}: {prob.item() * 100:7.3f}%"
            )

        print(
            f"Layer {layer_idx:02d} | "
            + " | ".join(tokens)
        )

    print("=" * 100)
"""

import torch


def show_probability_distribution(
    model,
    tokenizer,
    question: str,
    top_k: int = 5,
    max_new_tokens: int = 20,
):
    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    # ---------------------------------------------------------
    # Tokenisation initiale
    # ---------------------------------------------------------

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    input_ids = inputs["input_ids"].to(model.device)

    attention_mask = inputs["attention_mask"].to(model.device)

    print()
    print("=" * 110)
    print("Question :", question)
    print("=" * 110)

    generated_tokens = []

    # ---------------------------------------------------------
    # Génération token par token
    # ---------------------------------------------------------

    for step in range(max_new_tokens):

        with torch.no_grad():

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states

        print()
        print("-" * 110)
        print(
            f"TOKEN {step + 1}"
        )

        # -----------------------------------------------------
        # Distribution couche par couche
        # -----------------------------------------------------

        for layer_idx, hidden_state in enumerate(hidden_states):

            # Dernier token de la séquence
            last_hidden = hidden_state[:, -1, :]

            # Normalisation finale
            normalized_hidden = model.model.norm(
                last_hidden
            )

            # Projection vers le vocabulaire
            logits = model.lm_head(
                normalized_hidden
            )

            probabilities = torch.softmax(
                logits.float(),
                dim=-1
            )

            top_probs, top_indices = torch.topk(
                probabilities,
                k=top_k,
                dim=-1
            )

            tokens = []

            for prob, token_id in zip(
                top_probs[0],
                top_indices[0]
            ):

                token_id = token_id.item()

                token = tokenizer.decode(
                    [token_id],
                    clean_up_tokenization_spaces=False
                )

                tokens.append(
                    f"{repr(token)}: "
                    f"{prob.item() * 100:7.3f}%"
                )

            print(
                f"Layer {layer_idx:02d} | "
                + " | ".join(tokens)
            )

        # -----------------------------------------------------
        # Distribution officielle du modèle
        # -----------------------------------------------------

        next_token_logits = outputs.logits[:, -1, :]

        next_token_id = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True
        )

        next_token = tokenizer.decode(
            next_token_id[0],
            clean_up_tokenization_spaces=False
        )

        print()
        print(
            f" [Token généré] : {repr(next_token)}"
        )

        generated_tokens.append(next_token)

        # -----------------------------------------------------
        # Ajouter le nouveau token au contexte
        # -----------------------------------------------------

        input_ids = torch.cat(
            [input_ids, next_token_id],
            dim=-1
        )

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
            ],
            dim=-1
        )

        # -----------------------------------------------------
        # Stop si EOS
        # -----------------------------------------------------

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # ---------------------------------------------------------
    # Réponse complète
    # ---------------------------------------------------------

    print()
    print("=" * 110)
    print("Réponse générée :")
    print("".join(generated_tokens))
    print("=" * 110)

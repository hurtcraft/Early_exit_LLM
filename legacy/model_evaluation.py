
import json
import re
from pathlib import Path
import torch

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

import time
def load_questions(file_path: str | Path) -> list[str]:
    """
    Charge un fichier contenant des questions numérotées :

    1. What is the capital of France?
    2. What is 7 + 8?
    """

    questions = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Supprime "1. ", "2. ", etc.
            match = re.match(r"^\d+\.\s*(.*)$", line)

            if match:
                questions.append(match.group(1))

    return questions



def run_benchmark(lm, task: str, limit: int | None = None):
    """Exécute un benchmark et retourne ses résultats."""

    print(f"\nÉvaluation de {task} ({limit or 'tous les exemples'})")

    evaluation = evaluator.simple_evaluate(
        model=lm,
        tasks=[task],
        batch_size=1,
        limit=limit,
    )

    return evaluation["results"].get(task, {})


def evaluate_mmlu(lm, limit: int | None = 1):
    return run_benchmark(lm, "mmlu", limit)


def evaluate_gsm8k(lm, limit: int | None = 1):
    return run_benchmark(lm, "gsm8k", limit)


def evaluate_arc(lm, limit: int | None = 1):
    return run_benchmark(lm, "arc_challenge", limit)


def evaluate_hellaswag(lm, limit: int | None = 1):
    return run_benchmark(lm, "hellaswag", limit)


def evaluate_model(model, tokenizer, batch_size: int = 1):
    """Prépare le modèle et exécute les benchmarks sélectionnés."""

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    return {
        "mmlu": evaluate_mmlu(lm)
        # "gsm8k": evaluate_gsm8k(lm),
        # "arc_challenge": evaluate_arc(lm),
        # "hellaswag": evaluate_hellaswag(lm),
    }

def generate_answer(model, tokenizer, question: str) -> dict:

    messages = [
        {
            "role": "system",
            "content": (
                "Answer the question concisely and directly. "
                "Give only the final answer. "
                "Do not provide explanations, reasoning, or additional information."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # -----------------------------
    # Tokenisation
    # -----------------------------

    start_total = time.perf_counter()

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

    input_tokens = inputs["input_ids"].shape[1]

    # -----------------------------
    # Génération
    # -----------------------------

    # Important pour les GPU/XPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_generation = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Synchronisation pour avoir un temps fiable
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_generation = time.perf_counter()

    # -----------------------------
    # Décodage
    # -----------------------------

    generated_tokens = outputs[0][input_tokens:]

    output_tokens = generated_tokens.shape[0]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    if "</think>" in answer:
        answer = answer.split("</think>", 1)[1].strip()

    end_total = time.perf_counter()

    # -----------------------------
    # Métriques
    # -----------------------------

    generation_time = end_generation - start_generation
    total_time = end_total - start_total

    tokens_per_second = (
        output_tokens / generation_time
        if generation_time > 0
        else 0
    )

    return {
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "generation_time": generation_time,
        "total_time": total_time,
        "tokens_per_second": tokens_per_second,
    }
def answer_questions(
    model,
    tokenizer,
    questions_file: str | Path,
    output_file: str | Path
):
    """
    Pose toutes les questions au modèle et sauvegarde
    les réponses dans un fichier JSON.
    """

    questions = load_questions(questions_file)

    results = []

    for i, question in enumerate(questions, start=1):

        print(f"[{i}/{len(questions)}] {question}")

        answer = generate_answer(
            model,
            tokenizer,
            question
        )

        print(f"  -> {answer}")
        results.append({
            "question": question,
            **answer
        })
        # results.append({
        #     "question": question,
        #     "answer": answer
        # })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nRésultats sauvegardés dans : {output_file}")


def process_all_prompts(
    model,
    tokenizer,
    prompts_dir: Path,
    results_dir: Path,
    model_name: str,
    quant: str | None,
):
    output_dir = results_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_files = sorted(prompts_dir.glob("*.txt"))

    print(f"Nombre de fichiers : {len(prompt_files)}")

    for prompt_file in prompt_files:

        output_file = output_dir / (
            f"{prompt_file.stem}_{model_name}_{quant or 'fp16'}.json"
        )

        print("\n" + "=" * 60)
        print(f"Prompt file : {prompt_file}")
        print(f"Output file : {output_file}")
        print("=" * 60)

        answer_questions(
            model,
            tokenizer,
            prompt_file,
            output_file
        )
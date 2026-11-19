"""Entraîne un tuned lens par modèle (cf. early_exit/lens.py).

Données : texte générique (FineWeb, en streaming), découpé en séquences de
--seq-len tokens. Les --val premières séquences servent à l'évaluation (jamais vues
à l'entraînement), les suivantes à l'entraînement.

Pour chaque modèle :
  1. évaluation avant entraînement  = logit lens
  2. entraînement (un passage sur --tokens tokens)
  3. évaluation après entraînement  = tuned lens
  4. lenses/<modèle>.pt  (translateurs + les deux évaluations)

Usage :
    python train_lens.py                                   # les 3 modèles de config.MODELS
    python train_lens.py --models qwen3.5-2b --tokens 1e6  # essai rapide
    python train_lens.py --models qwen3.5-9b --groups 2    # si le 9B manque de VRAM

--groups N : entraîne les couches en N groupes successifs (moins de VRAM pour les
gradients et le momentum, mais N passages sur les données).
"""
import os
# GPU Intel (XPU) : autorise les allocations de plus de 4 Go (sinon OutOfMemoryError
# au chargement du modèle, même avec la mémoire libre). Avant l'import de torch.
os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
os.environ["SYCL_PI_LEVEL_ZERO_USE_RELAXED_ALLOCATION_LIMITS"] = "1"

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from config import MODELS
from early_exit.lens import TunedLens
from early_exit.models.factory import ModelFactory
from early_exit.utils import free_memory

LENS_DIR = Path("lenses")


def load_sequences(tokenizer, n_seqs: int, seq_len: int, dataset: str, subset: str | None,
                   text_field: str = "text") -> list:
    """n_seqs séquences [1, seq_len] : documents tokenisés, concaténés (séparés par
    EOS) puis découpés. Streaming : seule la partie utilisée est téléchargée."""
    from datasets import load_dataset

    ds = load_dataset(dataset, subset, split="train", streaming=True)
    eos = tokenizer.eos_token_id
    buf, seqs = [], []
    bar = tqdm(total=n_seqs, desc=f"données {dataset}", unit="séq")
    for doc in ds:
        buf.extend(tokenizer(doc[text_field], add_special_tokens=False)["input_ids"])
        if eos is not None:
            buf.append(eos)
        while len(buf) >= seq_len and len(seqs) < n_seqs:
            seqs.append(torch.tensor(buf[:seq_len], dtype=torch.long).unsqueeze(0))
            buf = buf[seq_len:]
            bar.update(1)
        if len(seqs) >= n_seqs:
            break
    bar.close()
    if len(seqs) < n_seqs:
        raise RuntimeError(f"jeu de données trop petit : {len(seqs)}/{n_seqs} séquences")
    return seqs


def print_comparison(before: dict, after: dict) -> None:
    tqdm.write(f"  {'couche':>6} | {'KL logit':>9} {'KL tuned':>9} | "
               f"{'top1 logit':>10} {'top1 tuned':>10} | {'ppl logit':>10} {'ppl tuned':>10}")
    for j, l in enumerate(before["layers"]):
        tqdm.write(f"  {l:>6} | {before['kl'][j]:>9.3f} {after['kl'][j]:>9.3f} | "
                   f"{before['top1'][j]:>10.1%} {after['top1'][j]:>10.1%} | "
                   f"{before['ppl'][j]:>10.1f} {after['ppl'][j]:>10.1f}")
    tqdm.write(f"  dernière couche : ppl {after['final_ppl']:.2f}")


def train_model(model_name: str, quant, args) -> Path:
    tqdm.write(f"\n######## {model_name} ########")
    adapter = ModelFactory.load_class(name=model_name, quant=quant)
    try:
        n_train = int(args.tokens) // args.seq_len
        seqs = load_sequences(adapter.tokenizer, args.val + n_train, args.seq_len,
                              args.dataset, args.subset)
        val, train = seqs[:args.val], seqs[args.val:]

        lens = TunedLens.for_adapter(adapter, model_name, args.min_layer)
        tqdm.write(f"couches {lens.layers[0]}..{lens.layers[-1]}, "
                   f"{sum(p.numel() for p in lens.parameters()) / 1e6:.0f} M paramètres, "
                   f"{len(train)} séquences d'entraînement")

        before = lens.evaluate(adapter, val, chunk=args.chunk)   # translateurs à zéro = logit lens

        history = []
        groups = [lens.layers[i::args.groups] for i in range(args.groups)]
        for g, layers in enumerate(groups, 1):
            desc = f"{model_name} groupe {g}/{len(groups)}" if len(groups) > 1 else model_name
            history += [[g, s, kl] for s, kl in lens.fit(
                adapter, train, layers=layers, lr=args.lr, seqs_per_step=args.seqs_per_step,
                chunk=args.chunk, desc=desc)]

        after = lens.evaluate(adapter, val, chunk=args.chunk)
        print_comparison(before, after)

        path = lens.save(
            LENS_DIR / f"{model_name}.pt",
            train={"dataset": args.dataset, "subset": args.subset, "tokens": len(train) * args.seq_len,
                   "seq_len": args.seq_len, "lr": args.lr, "seqs_per_step": args.seqs_per_step,
                   "groups": args.groups, "history": history},
            eval={"logit": before, "tuned": after},
        )
        tqdm.write(f"Lens : {path}")
        return path
    finally:
        del adapter
        free_memory()


def main():
    names = [m for m, _ in MODELS]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=names, choices=names)
    ap.add_argument("--tokens", type=float, default=5e6, help="tokens d'entraînement (défaut 5e6)")
    ap.add_argument("--val", type=int, default=16, help="séquences d'évaluation (défaut 16)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--seqs-per-step", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=256, help="tokens par bloc de logits")
    ap.add_argument("--groups", type=int, default=1)
    ap.add_argument("--min-layer", type=int, default=None, help="défaut : n_layers // 2")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    ap.add_argument("--subset", default="sample-10BT")
    args = ap.parse_args()

    for model_name, quant in MODELS:
        if model_name in args.models:
            train_model(model_name, quant, args)


if __name__ == "__main__":
    main()

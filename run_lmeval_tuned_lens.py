"""run_lmeval.py avec le tuned lens au lieu du logit lens.

Mêmes benchmarks, mêmes stratégies et mêmes options que run_lmeval.py ; seule la
lecture des couches intermédiaires change : chaque couche passe par son translateur
(lenses/<modèle>.pt, cf. train_lens.py) avant norm + lm_head. La dernière couche n'a
pas de translateur : la baseline NoExitStrategy est identique à celle du logit lens.

Usage (mêmes options que run_lmeval.py) :
    python run_lmeval_tuned_lens.py
    python run_lmeval_tuned_lens.py --models qwen3.5-2b --tasks wikitext
    python run_lmeval_tuned_lens.py --stamp 20261120_090000     # reprise

Sorties :
    results/lmeval_tuned/<stamp>/<modèle>/<tâche>.json   (options : "lens": "tuned")
    plots/lmeval_tuned/<stamp>/<modèle>_<tâche>.png
"""
from run_lmeval import main

if __name__ == "__main__":
    main(default_lens="tuned")

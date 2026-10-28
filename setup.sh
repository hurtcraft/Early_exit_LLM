#!/usr/bin/env bash
# Installation du projet sur un pod RunPod (image PyTorch + CUDA).
#
# Prérequis : dépôt cloné dans le Network Volume, par exemple :
#   cd /workspace && git clone git@github.com:<toi>/<repo>.git projet && cd projet
#
# Usage (depuis la racine du projet) :
#   bash setup.sh                  # installation + téléchargement des modèles
#   SKIP_MODELS=1 bash setup.sh    # sans télécharger les modèles
#
# Modèles gated (Gemma, Mistral…) : exporter HF_TOKEN avant de lancer le script.
set -euo pipefail

cd "$(dirname "$0")"

# --------------------------------------------------
# Caches Hugging Face (datasets lm_eval, hub) dans le volume persistant :
# sans ça, ils sont perdus à chaque arrêt du pod.
# --------------------------------------------------
if [ -d /workspace ]; then
    export HF_HOME=/workspace/.cache/huggingface
    if ! grep -q "HF_HOME=" ~/.bashrc 2>/dev/null; then
        echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
    fi
fi

echo "==> Outils système (tmux)"
if ! command -v tmux >/dev/null; then
    apt-get update -qq && apt-get install -y -qq tmux
fi

echo "==> Dépendances Python"
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q

echo "==> Vérification du GPU"
python - <<'EOF'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERREUR : CUDA indisponible, vérifier l'image du pod.")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i} : {p.name} ({p.total_memory / 1024**3:.0f} Go)")
print(f"  torch {torch.__version__} | CUDA {torch.version.cuda}")
EOF

if [ "${SKIP_MODELS:-0}" != "1" ]; then
    echo "==> Téléchargement des modèles (tools/models_to_test.json)"
    python tools/download_models.py
fi

mkdir -p results plots

echo
echo "Installation terminée. Lancer un run dans tmux pour survivre à une coupure SSH :"
echo "  tmux new -s run"
echo "  python run_lmeval.py"
echo "  (détacher : Ctrl+B puis D | revenir : tmux attach -t run)"

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

python -m pip install --upgrade pip -q

# --------------------------------------------------
# torch aligné sur le CUDA du pod (nvcc). Le torch de PyPI peut être compilé pour
# une autre version (ex. cu128 sur une image CUDA 13.0) : causal-conv1d refuse
# alors de compiler (« CUDA versions mismatch »). On réinstalle torch, torchvision
# et torchaudio ENSEMBLE depuis l'index PyTorch du bon CUDA, avant requirements.txt.
# --------------------------------------------------
[ -x /usr/local/cuda/bin/nvcc ] && export PATH="/usr/local/cuda/bin:$PATH"

cuda_tag() {  # "13.0" -> "130"
    sed -n 's/[^0-9]*\([0-9][0-9]*\)\.\([0-9]\).*/\1\2/p' | head -1
}

NVCC_CUDA=""
if command -v nvcc >/dev/null; then
    NVCC_CUDA=$(nvcc --version | grep -o 'release [0-9]*\.[0-9]*' | cuda_tag)
fi
DRIVER_CUDA=""
if command -v nvidia-smi >/dev/null; then
    DRIVER_CUDA=$(nvidia-smi | grep -o 'CUDA Version: [0-9]*\.[0-9]*' | cuda_tag)
fi
torch_cuda() {
    python -c "import torch; print((torch.version.cuda or '').replace('.', ''))" 2>/dev/null || true
}

CAN_BUILD_KERNELS=0
echo "==> torch / CUDA (nvcc : ${NVCC_CUDA:-absent} | driver : ${DRIVER_CUDA:-?} | torch : $(torch_cuda))"
if [ -z "$NVCC_CUDA" ]; then
    echo "  !! nvcc absent : kernels DeltaNet non compilables (image « devel » nécessaire)."
elif [ -n "$DRIVER_CUDA" ] && [ "$DRIVER_CUDA" -lt "$NVCC_CUDA" ]; then
    echo "  !! driver (CUDA $DRIVER_CUDA) plus ancien que nvcc (CUDA $NVCC_CUDA) : torch laissé tel quel."
else
    if [ "$(torch_cuda)" != "$NVCC_CUDA" ]; then
        echo "  torch réinstallé depuis l'index cu$NVCC_CUDA"
        python -m pip uninstall -y -q torch torchvision torchaudio || true
        python -m pip install -q "torch>=2.14" torchvision torchaudio \
            --index-url "https://download.pytorch.org/whl/cu$NVCC_CUDA"
    fi
    CAN_BUILD_KERNELS=1
fi

echo "==> Dépendances Python"
python -m pip install -r requirements.txt -q

# --------------------------------------------------
# Kernels optimisés des couches DeltaNet de Qwen3.5. Sans eux, transformers se rabat
# sur une implémentation PyTorch correcte mais beaucoup plus lente.
#   flash-linear-attention : Triton, rien à compiler.
#   causal-conv1d          : extension CUDA compilée contre le torch installé
#                            (~10 min). La wheel est gardée dans /workspace, rangée
#                            par version de torch : un binaire compilé pour un autre
#                            torch ne s'importe pas (« undefined symbol »).
# --------------------------------------------------
echo "==> Kernels DeltaNet"
python -m pip install -q flash-linear-attention || echo "  !! flash-linear-attention non installé"

if [ "$CAN_BUILD_KERNELS" = "1" ] && ! python -c "import causal_conv1d" 2>/dev/null; then
    python -m pip uninstall -y -q causal-conv1d 2>/dev/null || true
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    WHEEL_DIR="${WHEEL_CACHE:-/workspace/.cache/wheels}/causal_conv1d-torch$TORCH_VER"

    if ! ls "$WHEEL_DIR"/*.whl >/dev/null 2>&1; then
        echo "  compilation de causal-conv1d pour torch $TORCH_VER (~10 min)"
        python -m pip install -q ninja
        mkdir -p "$WHEEL_DIR"
        # MAX_JOBS borne la RAM de la compilation
        CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS:-8}" \
            python -m pip wheel causal-conv1d --no-build-isolation --no-deps --no-cache-dir \
            -w "$WHEEL_DIR" -q || rm -f "$WHEEL_DIR"/*.whl
    else
        echo "  causal-conv1d : wheel en cache ($WHEEL_DIR)"
    fi

    if ls "$WHEEL_DIR"/*.whl >/dev/null 2>&1; then
        python -m pip install -q --no-deps "$WHEEL_DIR"/*.whl
    fi
    # un module qui ne s'importe pas fait planter transformers : on le retire
    if ! python -c "import causal_conv1d" 2>/dev/null; then
        echo "  !! causal-conv1d inutilisable : désinstallé (implémentation PyTorch utilisée)"
        python -m pip uninstall -y -q causal-conv1d 2>/dev/null || true
        rm -rf "$WHEEL_DIR"
    fi
fi

echo "==> Vérification du GPU"
python - <<'EOF'
import importlib.util
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERREUR : CUDA indisponible, vérifier l'image du pod.")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i} : {p.name} ({p.total_memory / 1024**3:.0f} Go)")
print(f"  torch {torch.__version__} | CUDA {torch.version.cuda}")
for mod in ("fla", "causal_conv1d"):
    try:
        importlib.import_module(mod)
        status = "OK"
    except Exception as e:
        status = f"ABSENT ({type(e).__name__}) -> implémentation PyTorch, plus lente"
    print(f"  {mod:<14}: {status}")
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

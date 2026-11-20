#!/usr/bin/env bash
# Entraîne les tuned lens manquants, puis lance l'évaluation des benchmarks avec
# le tuned lens (LENS=tuned ./launch_parallel.sh).
#
#   1. pour chaque modèle : lenses/<modèle>.pt existe déjà -> rien à faire,
#      sinon -> python train_lens.py --models <modèle>
#      (en cas de manque de mémoire GPU, nouvel essai avec --groups 2)
#   2. si tous les lens sont là : LENS=tuned ./launch_parallel.sh <préréglages>
#      (l'orchestrateur attend que la mémoire GPU soit libérée avant chaque modèle)
#
# Usage :
#   ./train_and_eval_tuned.sh             # 2b, 4b, 9b
#   ./train_and_eval_tuned.sh 9b          # seulement ces préréglages
#   STAMP=20261121_090000 ./train_and_eval_tuned.sh   # reprend l'évaluation de ce lancement
#
# Le script se relance lui-même dans une session tmux "lens-pipeline" (survit à une
# coupure SSH) : tmux attach -t lens-pipeline   (détacher : Ctrl+B puis D)
# Journal : logs/lens/<date>_pipeline.log
#
# Variables optionnelles : PYTHON, STAMP (transmis à launch_parallel.sh),
# TRAIN_ARGS (options supplémentaires de train_lens.py, ex. TRAIN_ARGS="--tokens 2e6").
set -euo pipefail
cd "$(dirname "$0")"
SCRIPT="$(basename "$0")"
SESSION="lens-pipeline"
PYTHON="${PYTHON:-python}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
ALL_PRESETS=(2b 4b 9b)

model_of() {   # préréglage -> nom du modèle (mêmes préréglages que launch_parallel.sh)
    case "$1" in
        2b) echo "qwen3.5-2b" ;;
        4b) echo "qwen3.5-4b" ;;
        9b) echo "qwen3.5-9b" ;;
        *)  return 1 ;;
    esac
}

PRESETS=("$@")
[ ${#PRESETS[@]} -eq 0 ] && PRESETS=("${ALL_PRESETS[@]}")
for p in "${PRESETS[@]}"; do
    model_of "$p" >/dev/null || { echo "Préréglage inconnu : $p (attendu : ${ALL_PRESETS[*]})" >&2; exit 1; }
done

# --- hors tmux : se relance dans une session tmux et rend la main -------------
if [ -z "${TMUX:-}" ] && [ "${PIPELINE_INNER:-}" != "1" ]; then
    command -v tmux >/dev/null || { echo "tmux introuvable" >&2; exit 1; }
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        echo "Session '$SESSION' déjà ouverte : tmux attach -t $SESSION" >&2
        echo "(ou la fermer : tmux kill-session -t $SESSION)" >&2
        exit 1
    fi
    mkdir -p logs/lens
    log="logs/lens/$(date +%Y%m%d_%H%M%S)_pipeline.log"
    env_vars="PIPELINE_INNER=1 PYTHON=$(printf '%q' "$PYTHON") TRAIN_ARGS=$(printf '%q' "$TRAIN_ARGS")"
    [ -n "${STAMP:-}" ] && env_vars+=" STAMP=$(printf '%q' "$STAMP")"
    cmd="$env_vars bash $(printf '%q' "./$SCRIPT") ${PRESETS[*]} 2>&1 | tee -a $(printf '%q' "$log"); exec bash"
    tmux new-session -d -s "$SESSION" -c "$PWD" "bash -c $(printf '%q' "$cmd")"
    echo "Lancé dans tmux : tmux attach -t $SESSION"
    echo "Journal         : $log"
    exit 0
fi

# --- 1. tuned lens manquants ----------------------------------------------------
echo "=== Tuned lens : vérification $(date '+%F %T') ==="
mkdir -p lenses logs/lens
for p in "${PRESETS[@]}"; do
    model="$(model_of "$p")"
    lens="lenses/$model.pt"
    if [ -f "$lens" ]; then
        echo "  $lens : déjà présent"
        continue
    fi

    echo "  $lens absent : entraînement ($(date '+%F %T'))"
    train_log="logs/lens/$(date +%Y%m%d_%H%M%S)_$model.log"
    # shellcheck disable=SC2086  # TRAIN_ARGS : plusieurs options séparées par des espaces
    if ! $PYTHON train_lens.py --models "$model" $TRAIN_ARGS 2>&1 | tee "$train_log"; then
        if grep -q "OutOfMemoryError\|out of memory" "$train_log"; then
            echo "  manque de mémoire GPU : nouvel essai avec --groups 2"
            # shellcheck disable=SC2086
            $PYTHON train_lens.py --models "$model" --groups 2 $TRAIN_ARGS 2>&1 | tee -a "$train_log"
        else
            echo "!! entraînement de $model en échec (journal : $train_log) : évaluation non lancée" >&2
            exit 1
        fi
    fi

    [ -f "$lens" ] || { echo "!! $lens toujours absent après l'entraînement : arrêt" >&2; exit 1; }
    echo "  $lens : entraîné ($(date '+%F %T'))"
done

# --- 2. évaluation avec le tuned lens ---------------------------------------------
echo
echo "=== Évaluation des benchmarks avec le tuned lens : ${PRESETS[*]} ==="
LENS=tuned PYTHON="$PYTHON" ./launch_parallel.sh "${PRESETS[@]}"
echo
echo "Avancement : tmux attach -t lmeval-tuned"

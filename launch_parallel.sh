#!/usr/bin/env bash
# Lance les benchmarks de plusieurs modèles, un modèle après l'autre, avec pour
# chaque modèle plusieurs run_lmeval.py en parallèle (une session tmux chacun).
#
# Avec un batch de 1, un seul processus laisse le GPU en grande partie inactif :
# plusieurs processus sur le même GPU augmentent le débit total. Tous écrivent
# dans le même dossier results/lmeval/<STAMP>/ (un fichier par modèle x tâche).
#
# Usage :
#   ./launch_parallel.sh                  # 2b, puis 4b, puis 9b
#   ./launch_parallel.sh 4b 9b            # seulement ces préréglages, dans cet ordre
#   STAMP=20261109_120000 ./launch_parallel.sh   # reprend / complète ce lancement
#
# Déroulement : une session tmux "lmeval" (l'orchestrateur) lance les processus
# d'un préréglage, chacun dans sa propre session ("lmeval-2b-gsm8k"…), attend
# qu'ils soient TOUS terminés, attend que la mémoire GPU soit libérée (chaque
# processus ne charge qu'un modèle et se termine), puis passe au préréglage
# suivant. Le script rend la main tout de suite : tout tourne dans tmux et survit
# à une coupure SSH.
#
# Variables optionnelles : STAMP, SESSION (préfixe des sessions tmux), PYTHON.
#
# Suivre les calculs :
#   tmux ls                          liste des sessions
#   tmux attach -t lmeval            avancement global (orchestrateur)
#   tmux attach -t lmeval-2b-gsm8k   un processus précis
#   détacher sans arrêter : Ctrl+B puis D
#   tout arrêter : tmux ls -F '#S' | grep '^lmeval' | xargs -n1 tmux kill-session -t
#   journaux : logs/lmeval/<STAMP>/<session>.log
#
# Préréglages, dimensionnés pour une RTX A5000 (24 Go). Chaque processus charge
# sa propre copie du modèle :
#   2b (~5 Go par copie)  : 3 processus
#   4b (~9 Go par copie)  : 2 processus
#   9b (~19 Go)           : 1 seul processus (pas de place pour une 2e copie)
set -euo pipefail
cd "$(dirname "$0")"
SCRIPT="$(basename "$0")"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-python}"
SESSION="${SESSION:-lmeval}"
LOG_DIR="logs/lmeval/$STAMP"
POLL="${POLL:-10}"      # secondes entre deux actualisations de l'avancement
GPU_FREE_MB="${GPU_FREE_MB:-1000}"       # mémoire GPU considérée comme libérée en dessous
GPU_WAIT_MAX="${GPU_WAIT_MAX:-300}"       # attente max de la libération (secondes)
ALL_PRESETS=(2b 4b 9b)

# Processus d'un préréglage : une ligne "nom|modèle|tâches" par session tmux
# (session nommée "$SESSION-<préréglage>-<nom>")
jobs_for() {
    case "$1" in
        2b) echo "gsm8k|qwen3.5-2b|gsm8k"
            echo "hellaswag|qwen3.5-2b|hellaswag"
            echo "wikitext-mmlu|qwen3.5-2b|wikitext mmlu" ;;
        4b) echo "gsm8k|qwen3.5-4b|gsm8k"
            echo "loglik|qwen3.5-4b|hellaswag wikitext mmlu" ;;
        9b) echo "all|qwen3.5-9b|gsm8k hellaswag wikitext mmlu" ;;
        *)  return 1 ;;
    esac
}

session_exists() {
    tmux has-session -t "=$1" 2>/dev/null   # "=" : nom exact, pas de préfixe
}

# sessions tmux des processus des préréglages donnés
job_sessions() {
    local preset name model tasks
    for preset in "$@"; do
        while IFS="|" read -r name model tasks; do
            echo "$SESSION-$preset-$name"
        done < <(jobs_for "$preset")
    done
}

wait_gpu_free() {
    command -v nvidia-smi >/dev/null || return 0
    local waited=0 used
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
               | sort -n | tail -1 || true)
        if ! [[ "$used" =~ ^[0-9]+$ ]]; then
            echo "  (nvidia-smi illisible : pas de vérification de la mémoire GPU)"
            return 0
        fi
        if [ "$used" -lt "$GPU_FREE_MB" ]; then
            echo "  mémoire GPU libérée (${used} Mo utilisés)"
            return 0
        fi
        if [ "$waited" -ge "$GPU_WAIT_MAX" ]; then
            echo "  !! mémoire GPU encore occupée (${used} Mo) après ${GPU_WAIT_MAX} s, on continue"
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
    done
}

# --- affichage de l'avancement ------------------------------------------------
HISTORY=()   # lignes déjà affichées (réaffichées à chaque actualisation)

# écrit une ligne dans le journal et la garde pour l'affichage
say() {
    echo "$1"
    HISTORY+=("$1")
}

hms() {   # 3725 -> 1h02m
    local s=$1
    if [ "$s" -ge 3600 ]; then printf '%dh%02dm' $((s / 3600)) $((s % 3600 / 60))
    else printf '%dm%02ds' $((s / 60)) $((s % 60)); fi
}

progress_bar() {   # progress_bar fait total -> [########----------------]
    local done=$1 total=$2 width=24 n=0
    [ "$total" -gt 0 ] && n=$((done * width / total))
    printf '[%s%s]' "$(printf '%*s' "$n" '' | tr ' ' '#')" "$(printf '%*s' $((width - n)) '' | tr ' ' '-')"
}

# Redessine l'écran de l'orchestrateur. Écrit sur le terminal tmux seulement
# (pas dans le journal, qui garderait sinon une copie à chaque actualisation).
draw_progress() {
    { : > /dev/tty; } 2>/dev/null || return 0   # pas de terminal (journal seul)
    local preset=$1 sess f done total skipped start status now pct eff eta
    shift
    now=$(date +%s)
    {
        printf '\033[H\033[2J'   # efface l'écran
        printf '%s\n' "${HISTORY[@]}"
        echo
        echo "--- $preset en cours | $(date '+%F %T') | actualisé toutes les ${POLL} s ---"
        for sess in "$@"; do
            f="$LOG_DIR/$sess.progress"
            if [ -f "$LOG_DIR/$sess.done" ]; then
                printf '  %-28s terminé (code %s)\n' "$sess" "$(cat "$LOG_DIR/$sess.done")"
                continue
            fi
            if [ ! -f "$f" ]; then
                printf '  %-28s démarrage…\n' "$sess"
                continue
            fi
            IFS="|" read -r done total skipped start status < "$f"
            pct=0; [ "$total" -gt 0 ] && pct=$((100 * done / total))
            eff=$((done - skipped))    # configs calculées par ce processus
            eta="?"
            [ "$eff" -gt 0 ] && eta="~$(hms $(( (now - start) * (total - done) / eff )))"
            printf '  %-28s %s %3d/%-3d %3d%%  écoulé %s  reste %s\n' \
                "$sess" "$(progress_bar "$done" "$total")" "$done" "$total" "$pct" \
                "$(hms $((now - start)))" "$eta"
            printf '  %-28s   └ %s\n' "" "$status"
        done
        echo
        echo "(restant estimé d'après la durée moyenne des configs déjà faites)"
        echo "(Ctrl+B puis D : quitter sans arrêter)"
    } > /dev/tty
}

# --- orchestrateur : tourne dans la session tmux "$SESSION" ------------------
orchestrate() {
    mkdir -p "$LOG_DIR"
    say "Stamp : $STAMP | préréglages : $* | résultats : results/lmeval/$STAMP/"

    for preset in "$@"; do
        say ""
        say "=== $preset : démarrage $(date '+%F %T') ==="
        say "$(wait_gpu_free)"   # le modèle précédent doit avoir rendu la mémoire GPU
        sessions=()
        while IFS="|" read -r name model tasks; do
            sess="$SESSION-$preset-$name"
            log="$LOG_DIR/$sess.log"
            marker="$LOG_DIR/$sess.done"
            progress="$LOG_DIR/$sess.progress"
            rm -f "$marker" "$progress"
            if session_exists "$sess"; then
                say "  !! session '$sess' déjà ouverte (lancement précédent ?) : ignorée"
                continue
            fi
            # LMEVAL_PROGRESS : fichier d'avancement écrit par run_lmeval.py.
            # pipefail : code de retour de python (pas de tee), écrit dans le marqueur
            # de fin lu par l'orchestrateur. La session reste ouverte (exec bash).
            cmd="set -o pipefail; LMEVAL_PROGRESS='$progress' $PYTHON run_lmeval.py --models $model --tasks $tasks --stamp $STAMP 2>&1 | tee -a '$log'; status=\$?; echo \$status > '$marker'; echo; echo \"== $sess terminé (code \$status) : $log ==\"; exec bash"
            tmux new-session -d -s "$sess" -c "$PWD" "bash -c $(printf '%q' "$cmd")"
            sessions+=("$sess")
            say "  session '$sess' : $model | $tasks   (tmux attach -t $sess)"
        done < <(jobs_for "$preset")

        # attend la fin de tous les processus du préréglage (marqueur écrit, ou
        # session fermée à la main), en affichant l'avancement
        pending=(${sessions[@]+"${sessions[@]}"})
        while [ ${#pending[@]} -gt 0 ]; do
            still=()
            for sess in "${pending[@]}"; do
                marker="$LOG_DIR/$sess.done"
                if [ -f "$marker" ] || ! session_exists "$sess"; then
                    say "  $sess : terminé $(date '+%F %T') (code $(cat "$marker" 2>/dev/null || echo 'session fermée'))"
                else
                    still+=("$sess")
                fi
            done
            pending=(${still[@]+"${still[@]}"})
            [ ${#pending[@]} -gt 0 ] || break
            draw_progress "$preset" "${sessions[@]}"
            sleep "$POLL"
        done
        say "=== $preset : terminé ==="
        # écran propre : historique seul (le tableau du préréglage fini disparaît)
        { { printf '\033[H\033[2J'; printf '%s\n' "${HISTORY[@]}"; } > /dev/tty; } 2>/dev/null || true
    done

    say ""
    say "Tout est terminé $(date '+%F %T'). Figures de comparaison :"
    say "  $PYTHON compare_model.py results/lmeval/$STAMP --out plots/compare/$STAMP"
}

if [ "${1:-}" = "--orchestrate" ]; then
    shift
    orchestrate "$@"
    exit 0
fi

# --- lancement : vérifie, puis démarre l'orchestrateur dans tmux -------------
PRESETS=("$@")
[ ${#PRESETS[@]} -eq 0 ] && PRESETS=("${ALL_PRESETS[@]}")
for p in "${PRESETS[@]}"; do
    if ! jobs_for "$p" >/dev/null; then
        echo "Préréglage inconnu : $p (attendu : ${ALL_PRESETS[*]})" >&2
        echo "Usage : $0 [2b] [4b] [9b]   (sans argument : tous, dans l'ordre)" >&2
        exit 1
    fi
done

if ! command -v tmux >/dev/null; then
    echo "tmux introuvable (installé par setup.sh)" >&2
    exit 1
fi
# aucune session du lancement ne doit déjà exister (lancement précédent encore ouvert)
existing=()
for s in "$SESSION" $(job_sessions "${PRESETS[@]}"); do
    session_exists "$s" && existing+=("$s")
done
if [ ${#existing[@]} -gt 0 ]; then
    echo "Sessions tmux déjà ouvertes : ${existing[*]}" >&2
    echo "Les fermer (si leurs calculs sont finis) :" >&2
    for s in "${existing[@]}"; do echo "  tmux kill-session -t $s" >&2; done
    echo "ou choisir un autre préfixe : SESSION=autre_nom $0 ${PRESETS[*]}" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
orch_cmd="STAMP=$(printf '%q' "$STAMP") PYTHON=$(printf '%q' "$PYTHON") SESSION=$(printf '%q' "$SESSION") bash $(printf '%q' "./$SCRIPT") --orchestrate ${PRESETS[*]} 2>&1 | tee -a $(printf '%q' "$LOG_DIR/$SESSION.log"); exec bash"
tmux new-session -d -s "$SESSION" -c "$PWD" "bash -c $(printf '%q' "$orch_cmd")"

echo "Stamp        : $STAMP"
echo "Préréglages  : ${PRESETS[*]} (l'un après l'autre)"
echo "Résultats    : results/lmeval/$STAMP/"
echo "Journaux     : $LOG_DIR/"
echo
echo "Sessions tmux (créées au fur et à mesure, un modèle à la fois) :"
echo "  tmux attach -t $SESSION   (orchestrateur : avancement global)"
for s in $(job_sessions "${PRESETS[@]}"); do echo "  tmux attach -t $s"; done

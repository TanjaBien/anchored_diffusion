#!/usr/bin/env bash
# Reproduce the paper's weight sweeps for one dataset: DM once, then every method at every weight
# of its grid. Runs sequentially, or in parallel with one run per GPU.
#
#   bash scripts/sweep.sh burgers
#   GPUS="0 1 2 3" bash scripts/sweep.sh darcy
#   METHODS="pidm anchored" ANCHORED_C="1e3 1e4" bash scripts/sweep.sh advection
#
# Any further arguments go to main.py unchanged, e.g. --wandb_track true or --data_dir.
# The toy additionally loops over its profiles (TOY_PROFILES) and dimensions (TOY_DIMS).
set -euo pipefail

EQ="${1:?usage: scripts/sweep.sh <advection|burgers|darcy|toy> [main.py args...]}"
shift
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The weight grids of the paper's tables. Weights of different methods are not on the same scale:
# the anchored weight that reaches a given residual is k times the PIDM one, k ~ 2e6 on advection,
# 1.1e8 on Burgers and 1.1e10 on Darcy (the mean ||J_R(x0)||_F^2 times the ratio of the time weights).
#
# Burgers anchored at c <= 1: the paper's runs drew the normalisation's random tangents from the
# global RNG; METHODS=anchored ANCHORED_C="1e-3 1e-2 1e-1 1" ... --physics_probe_seed -1 repeats them.
case "$EQ" in
    advection) PIDM_C_DEFAULT="1e-3 1e-2 1e-1 1"
               ANCHORED_C_DEFAULT="2e3 2e4 2e5 2e6" ;;
    burgers)   PIDM_C_DEFAULT="1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1"
               ANCHORED_C_DEFAULT="1e-3 1e-2 1e-1 1 10 100 1e3 1e4" ;;
    darcy)     PIDM_C_DEFAULT="1e-7 1e-6 1e-5 1e-4 1e-3"
               ANCHORED_C_DEFAULT="1e3 1e4 1e5 1e6 1e7" ;;
    toy)       PIDM_C_DEFAULT="1e-3 1e-2 1e-1 1"
               ANCHORED_C_DEFAULT="1e-3 1e-2 1e-1 1" ;;
    *) echo "unknown dataset '$EQ'" >&2; exit 1 ;;
esac
# lin and norm, one ingredient each, are the ablation: add them with METHODS="dm pidm lin norm anchored"
METHODS_DEFAULT="dm pidm anchored"

METHODS="${METHODS:-$METHODS_DEFAULT}"
PIDM_C="${PIDM_C:-$PIDM_C_DEFAULT}"
ANCHORED_C="${ANCHORED_C:-$ANCHORED_C_DEFAULT}"
LIN_C="${LIN_C:-$ANCHORED_C}"
NORM_C="${NORM_C:-$ANCHORED_C}"
TOY_PROFILES="${TOY_PROFILES:-graded}"   # the paper's table; its figures also use flat
TOY_DIMS="${TOY_DIMS:-10}"
read -r -a GPU_LIST <<< "${GPUS:-}"
PYTHON="${PYTHON:-python}"

grid() {
    case "$1" in
        dm) echo 0 ;;
        pidm) echo "$PIDM_C" ;;
        lin) echo "$LIN_C" ;;
        norm) echo "$NORM_C" ;;
        anchored) echo "$ANCHORED_C" ;;
        *) echo "unknown method '$1'" >&2; exit 1 ;;
    esac
}

TASKS=()
if [ "$EQ" = toy ]; then VARIANTS=(); for P in $TOY_PROFILES; do for D in $TOY_DIMS; do VARIANTS+=("$P:$D"); done; done
else VARIANTS=("-"); fi
for VARIANT in "${VARIANTS[@]}"; do
    for METHOD in $METHODS; do
        for C in $(grid "$METHOD"); do
            TASKS+=("$VARIANT $METHOD $C")
        done
    done
done

run_task() {
    local variant="$1" method="$2" c="$3" name extra=()
    shift 3
    name="${method}_c${c}"
    if [ "$variant" != "-" ]; then
        extra=(--toy_profile "${variant%%:*}" --toy_dim "${variant##*:}")
        name="${variant%%:*}_d${variant##*:}_${name}"
    fi
    echo "[$(date +%H:%M:%S)] start $EQ $name"
    "$PYTHON" -u main.py --config "config/$EQ.yaml" --method "$method" --c_residual "$c" \
        --name "$name" "${extra[@]}" "$@"
    echo "[$(date +%H:%M:%S)] done  $EQ $name"
}

echo "$EQ: ${#TASKS[@]} runs"
if [ ${#GPU_LIST[@]} -eq 0 ]; then
    for TASK in "${TASKS[@]}"; do
        read -r VARIANT METHOD C <<< "$TASK"
        run_task "$VARIANT" "$METHOD" "$C" "$@"
    done
    exit 0
fi

# one worker per GPU, each taking every Nth run
PIDS=()
for ((SHARD = 0; SHARD < ${#GPU_LIST[@]}; SHARD++)); do
    (
        export CUDA_VISIBLE_DEVICES="${GPU_LIST[$SHARD]}"
        for ((I = SHARD; I < ${#TASKS[@]}; I += ${#GPU_LIST[@]})); do
            read -r VARIANT METHOD C <<< "${TASKS[$I]}"
            run_task "$VARIANT" "$METHOD" "$C" "$@" || echo "FAILED $EQ ${TASKS[$I]}"
        done
    ) &
    PIDS+=("$!")
done
STATUS=0
for PID in "${PIDS[@]}"; do wait "$PID" || STATUS=1; done
exit "$STATUS"

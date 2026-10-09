#!/usr/bin/env bash
# Reproduce the paper on one machine, inside a detached tmux session: for every dataset the weight
# sweep (DM, PIDM and anchored, scripts/sweep.sh), then the evaluation of any run that has no report
# yet, then the dataset's table (results/table_<dataset>.csv).
#
#   bash scripts/run_all.sh                    # start
#   tmux attach -t anchored                    # watch (detach again with Ctrl-b d)
#   tail -f logs/anchored.log                  # or follow the log
#   tmux kill-session -t anchored              # stop
#
# Options, as environment variables:
#   DATASETS="toy darcy advection burgers"     datasets, in this order (default: all four)
#   GPUS="0 1"                                 one run per GPU at a time (default: one run at a time)
#   ARGS="--wandb_track true"                  passed to every run
#   ARGS_BURGERS="--train_batch_size 4"        passed to the runs of one dataset (ARGS_TOY, ARGS_DARCY, ...);
#                                              Burgers needs ~30 GB of GPU memory at the paper's batch size 8
#   SESSION=anchored                           name of the tmux session and of the log
# The sweep's own variables (METHODS, PIDM_C, ANCHORED_C, TOY_PROFILES, TOY_DIMS) are passed through.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
SESSION="${SESSION:-anchored}"
DATASETS="${DATASETS:-toy darcy advection burgers}"
LOG="logs/$SESSION.log"
mkdir -p logs results

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' is already running: tmux attach -t $SESSION" >&2
    exit 1
fi

# the job, written out so that it runs with exactly these settings even if the environment changes
JOB="logs/$SESSION.job.sh"
{
    echo '#!/usr/bin/env bash'
    echo 'set -uo pipefail'
    echo "cd '$PWD'"
    for var in GPUS METHODS PIDM_C ANCHORED_C LIN_C NORM_C TOY_PROFILES TOY_DIMS PYTHON; do
        [ -n "${!var:-}" ] && echo "export $var='${!var}'"
    done
    echo "echo \"=== started \$(date) on \$(hostname): $DATASETS ===\""
    for dataset in $DATASETS; do
        extra_var="ARGS_$(echo "$dataset" | tr '[:lower:]' '[:upper:]')"
        extra="${ARGS:-} ${!extra_var:-}"
        cat <<EOF
echo "=== $dataset: training \$(date) ==="
bash scripts/sweep.sh $dataset $extra || echo "=== $dataset: some runs failed ==="
echo "=== $dataset: evaluation \$(date) ==="
\${PYTHON:-python} -m src.evaluation.generation trained_models/$dataset --all || echo "=== $dataset: evaluation failed ==="
\${PYTHON:-python} -m src.evaluation.tables $dataset || echo "=== $dataset: no table ==="
EOF
    done
    echo 'echo "=== finished $(date) ==="'
} > "$JOB"
chmod +x "$JOB"

tmux new-session -d -s "$SESSION" "bash '$JOB' 2>&1 | tee -a '$LOG'"
echo "started tmux session '$SESSION' ($DATASETS)"
echo "  watch:  tmux attach -t $SESSION    or    tail -f $LOG"
echo "  stop:   tmux kill-session -t $SESSION"

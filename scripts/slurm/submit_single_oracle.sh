#!/usr/bin/env bash
# Submit one standalone PSRO oracle follow-up run to gpu2.
#
#   HF_TOKEN=... WANDB_API_KEY=... \
#   TAG=D4_retrain SOURCE_ORACLE=D4 STEPS=100 \
#   scripts/slurm/submit_single_oracle.sh
#
# Credentials are forwarded through the job environment only; they are never
# written into the generated sbatch script or any run artifact.
set -eu -o pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

WORKSPACE=${WORKSPACE:-/zfsauton/scratch/wentsec/coldpsro_slurm}
PYTHON=${SSP_PYTHON:-/zfsauton/scratch/wentsec/envs/srt/bin/python}
PSRO_ROOT=${PSRO_ROOT:-$WORKSPACE/psro/role_lora_zero_sum_psro}
SUFFIX=${SUFFIX:?set SUFFIX to the source run suffix}
SOURCE_RUN=$PSRO_ROOT/$SUFFIX
OUT_ROOT=${OUT_ROOT:-$PSRO_ROOT/${SUFFIX}_followups}

TAG=${TAG:?set TAG}
SOURCE_ORACLE=${SOURCE_ORACLE:?set SOURCE_ORACLE}
STEPS=${STEPS:?set STEPS}
START_ADAPTER=${START_ADAPTER:-}
SAVE_STEPS=${SAVE_STEPS:-25}
MAX_CKPT_NUM=${MAX_CKPT_NUM:-1}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-role_lora_cold_psro__${SUFFIX}__${TAG}}

PARTITION=${PARTITION:-project}
QOS=${QOS:-qos_project_preempt}
NODELIST=${NODELIST:-gpu2}
GPUS=${GPUS:-4}
GRES=${GRES:-gpu:h200:$GPUS}
CPUS=${CPUS:-40}
MEM=${MEM:-440G}
TIME_LIMIT=${TIME_LIMIT:-2-00:00:00}
JUDGE_PORT=${JUDGE_PORT:-5321}
RAY_PORT=${RAY_PORT:-6321}
JOB_NAME=${JOB_NAME:-oracle_$TAG}

[ -f "$SOURCE_RUN/state.json" ] || { echo "no run at $SOURCE_RUN" >&2; exit 1; }
LOGDIR=${LOGDIR:-$OUT_ROOT/logs}
mkdir -p "$LOGDIR"

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is not set; running with W&B disabled" >&2
  export WANDB_MODE=disabled
fi

START_ARG=""
[ -n "$START_ADAPTER" ] && START_ARG="--start-adapter $START_ADAPTER"

# Optional: v5 attacker objective and early stopping.  EARLY_STOP=1 turns the
# criterion on; the thresholds default to run_single_oracle.py's own defaults.
EXTRA_ARGS=""
[ -n "${MICRO_TRAIN:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --micro-train ${MICRO_TRAIN}"
[ -n "${MICRO_ROLLOUT:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --micro-rollout ${MICRO_ROLLOUT}"
[ -n "${GENERATE_MAX_LEN:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --generate-max-len ${GENERATE_MAX_LEN}"
[ -n "${PROMPT_MAX_LEN:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --prompt-max-len ${PROMPT_MAX_LEN}"
[ -n "${ATTACKER_OBJECTIVE:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --attacker-objective ${ATTACKER_OBJECTIVE}"
if [ "${EARLY_STOP:-0}" = "1" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --early-stop"
  [ -n "${EARLY_STOP_METRIC:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --early-stop-metric ${EARLY_STOP_METRIC}"
  [ -n "${EARLY_STOP_THRESHOLD:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --early-stop-threshold ${EARLY_STOP_THRESHOLD}"
  [ -n "${EARLY_STOP_GUARD_METRIC:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --early-stop-guard-metric ${EARLY_STOP_GUARD_METRIC}"
  [ -n "${EARLY_STOP_GUARD_MIN:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --early-stop-guard-min ${EARLY_STOP_GUARD_MIN}"
  [ -n "${EARLY_STOP_PATIENCE:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --early-stop-patience ${EARLY_STOP_PATIENCE}"
  [ -n "${EARLY_STOP_WINDOW:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --early-stop-window ${EARLY_STOP_WINDOW}"
  [ -n "${EARLY_STOP_MIN_STEP:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --early-stop-min-step ${EARLY_STOP_MIN_STEP}"
  [ -n "${EARLY_STOP_CEILING_METRIC:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --early-stop-ceiling-metric ${EARLY_STOP_CEILING_METRIC}"
  [ -n "${EARLY_STOP_CEILING_MAX:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --early-stop-ceiling-max ${EARLY_STOP_CEILING_MAX}"
  [ -n "${EARLY_STOP_ABORT_MAX:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --early-stop-abort-max ${EARLY_STOP_ABORT_MAX}"
  [ -n "${EARLY_STOP_ABORT_PATIENCE:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --early-stop-abort-patience ${EARLY_STOP_ABORT_PATIENCE}"
fi

SCRIPT=$LOGDIR/oracle_${TAG}.sbatch
cat > "$SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=1
#SBATCH --nodelist=$NODELIST
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --gres=$GRES
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME_LIMIT
#SBATCH --open-mode=append
#SBATCH --output=$LOGDIR/oracle_${TAG}_%j.log
#SBATCH --error=$LOGDIR/oracle_${TAG}_%j.log

set -eu -o pipefail
unset TRANSFORMERS_CACHE
echo "=== oracle job \$SLURM_JOB_ID restart=\${SLURM_RESTART_COUNT:-0} on \$(hostname) at \$(date -Is) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader || true

exec $PYTHON $HERE/run_single_oracle.py \\
  --source-run $SOURCE_RUN \\
  --source-oracle $SOURCE_ORACLE \\
  --tag $TAG \\
  --out-root $OUT_ROOT \\
  --steps $STEPS \\
  --save-steps $SAVE_STEPS \\
  --max-ckpt-num $MAX_CKPT_NUM \\
  --gpus $GPUS \\
  --judge-port $JUDGE_PORT \\
  --ray-port $RAY_PORT \\
  --python $PYTHON \\
  --wandb-run-name $WANDB_RUN_NAME \\
  $START_ARG $EXTRA_ARGS
EOF

SBATCH_ARGS=()
[ -n "${DEPENDENCY:-}" ] && SBATCH_ARGS+=(--dependency="$DEPENDENCY")
sbatch --export=ALL "${SBATCH_ARGS[@]}" "$SCRIPT"

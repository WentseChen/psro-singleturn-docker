#!/usr/bin/env bash
# Submit the official defender evaluation for one finished PSRO run.
#
# Uses a separate interpreter from training: safety-eval-fork needs fire,
# tenacity, langdetect, immutabledict and fschat, and the validated training
# environment (vllm 0.8.2 / transformers 4.50.0) must not be perturbed. The
# eval venv is an overlay created with --system-site-packages, so it inherits
# torch/vllm/transformers and adds only those five packages.
#
#   SUFFIX=... LABEL_PREFIX=cold_psro scripts/slurm/submit_eval.sh
set -eu -o pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

WORKSPACE=${WORKSPACE:-/zfsauton/scratch/wentsec/coldpsro_slurm}
UPSTREAM=${SSP_UPSTREAM:-$WORKSPACE/upstream}
HF_HOME_DIR=${SSP_HF_HOME:-/zfsauton/scratch/wentsec/hf_srt}
PYTHON=${SSP_EVAL_PYTHON:-/zfsauton/scratch/wentsec/envs/srteval/bin/python}
PSRO_ROOT=${PSRO_ROOT:-$WORKSPACE/psro/role_lora_zero_sum_psro}
SUFFIX=${SUFFIX:?set SUFFIX to the run to evaluate}
LABEL_PREFIX=${LABEL_PREFIX:-cold_psro}
COORDINATOR=${COORDINATOR:-modal_role_lora_zero_sum_psro}
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-$WORKSPACE/psro/upstream_selfredteam_role_full_eval}
TASKS=${TASKS:-}
DEFENDERS=${DEFENDERS:-}

PARTITION=${PARTITION:-project}
QOS=${QOS:-qos_project_preempt}
NODELIST=${NODELIST:-gpu2}
GPUS=${GPUS:-2}
GRES=${GRES:-gpu:h200:$GPUS}
CPUS=${CPUS:-24}
MEM=${MEM:-240G}
TIME_LIMIT=${TIME_LIMIT:-1-00:00:00}
JOB_NAME=${JOB_NAME:-${LABEL_PREFIX}_eval}

RUN_ROOT=$PSRO_ROOT/$SUFFIX
[ -f "$RUN_ROOT/state.json" ] || { echo "no run at $RUN_ROOT/state.json" >&2; exit 1; }
LOGDIR=${LOGDIR:-$RUN_ROOT/logs}
mkdir -p "$LOGDIR" "$EVAL_OUTPUT_ROOT"

SCRIPT=$LOGDIR/eval_${LABEL_PREFIX}.sbatch
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
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=$LOGDIR/eval_${LABEL_PREFIX}_%j.log
#SBATCH --error=$LOGDIR/eval_${LABEL_PREFIX}_%j.log

set -eu -o pipefail
unset TRANSFORMERS_CACHE
echo "=== eval job \$SLURM_JOB_ID restart=\${SLURM_RESTART_COUNT:-0} on \$(hostname) at \$(date -Is) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader || true

exec $PYTHON $HERE/official_eval_local.py \\
  --psro-root $PSRO_ROOT \\
  --suffix $SUFFIX \\
  --upstream $UPSTREAM \\
  --eval-output-root $EVAL_OUTPUT_ROOT \\
  --hf-home $HF_HOME_DIR \\
  --gpus $GPUS \\
  --label-prefix $LABEL_PREFIX \\
  --coordinator $COORDINATOR \\
  --tasks "$TASKS" \\
  --defenders "$DEFENDERS"
EOF

SBATCH_ARGS=()
[ -n "${DEPENDENCY:-}" ] && SBATCH_ARGS+=(--dependency="$DEPENDENCY")
sbatch --export=ALL "${SBATCH_ARGS[@]}" "$SCRIPT"

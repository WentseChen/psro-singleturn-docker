#!/usr/bin/env bash
# Merge one finished PSRO evaluation into the README results table.
#
# CPU-only and seconds long, so it is chained off the eval stage rather than
# inserted into it: bookkeeping must never be able to block the next training
# stage. Re-running is safe - rows are matched on their label.
set -eu -o pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=${WORKSPACE:-/zfsauton/scratch/wentsec/coldpsro_slurm}
PSRO_ROOT=${PSRO_ROOT:-$WORKSPACE/psro/role_lora_zero_sum_psro}
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-$WORKSPACE/psro/upstream_selfredteam_role_full_eval}
SUFFIX=${SUFFIX:?set SUFFIX}
LABEL_PREFIX=${LABEL_PREFIX:?set LABEL_PREFIX}
VARIANT=${VARIANT:?set VARIANT}
PYTHON=${SSP_MERGE_PYTHON:-python3}
LOGDIR=${LOGDIR:-$PSRO_ROOT/$SUFFIX/logs}
mkdir -p "$LOGDIR"

SCRIPT=$LOGDIR/merge_${LABEL_PREFIX}.sbatch
cat > "$SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${LABEL_PREFIX}_merge
#SBATCH --nodes=1
#SBATCH --partition=${PARTITION:-project}
#SBATCH --qos=${QOS:-qos_project_preempt}
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=$LOGDIR/merge_${LABEL_PREFIX}_%j.log
#SBATCH --error=$LOGDIR/merge_${LABEL_PREFIX}_%j.log
set -eu -o pipefail
exec $PYTHON $HERE/merge_eval_results.py \\
  --psro-root $PSRO_ROOT \\
  --suffix $SUFFIX \\
  --eval-output-root $EVAL_OUTPUT_ROOT \\
  --label-prefix $LABEL_PREFIX \\
  --variant "$VARIANT"
EOF

SBATCH_ARGS=()
[ -n "${DEPENDENCY:-}" ] && SBATCH_ARGS+=(--dependency="$DEPENDENCY")
sbatch --export=ALL "${SBATCH_ARGS[@]}" "$SCRIPT"

#!/usr/bin/env bash
# Submit the cold role-LoRA zero-sum PSRO continuation to one 4xH200 node.
#
# The job body is scripts/slurm/cold_psro_local.py, which resumes from
# state.json on every entry, so --requeue is safe: a preempted job restarts the
# driver, finished cells and oracles are skipped, and an interrupted oracle
# retrains from the base adapter (the run contract writes no mid-role
# checkpoint, exactly as on Modal).
#
#   WANDB_API_KEY=... scripts/slurm/submit_cold_psro.sh
#
# WANDB_API_KEY is forwarded through the job environment only; it is never
# written into the generated sbatch script or into any run artifact.
set -eu -o pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

WORKSPACE=${WORKSPACE:-/zfsauton/scratch/wentsec/coldpsro_slurm}
WORK=${SSP_WORK:-$WORKSPACE/work}
HF_HOME_DIR=${SSP_HF_HOME:-/zfsauton/scratch/wentsec/hf_srt}
SFT_DIR=${SSP_SFT_DIR:-/zfsauton/scratch/wentsec/sft_data}
PYTHON=${SSP_PYTHON:-/zfsauton/scratch/wentsec/envs/srt/bin/python}
PSRO_ROOT=${PSRO_ROOT:-$WORKSPACE/psro/role_lora_zero_sum_psro}
RAW_PAYOFF_ROOT=${RAW_PAYOFF_ROOT:-$WORKSPACE/psro/raw_payoff_v2}
SUFFIX=${SUFFIX:?set SUFFIX to the seeded run suffix}

PARTITION=${PARTITION:-project}
QOS=${QOS:-qos_project_preempt}
NODELIST=${NODELIST-gpu2}
GPUS=${GPUS:-4}
GRES=${GRES:-gpu:h200:$GPUS}
CPUS=${CPUS:-40}
MEM=${MEM:-440G}
TIME_LIMIT=${TIME_LIMIT:-2-00:00:00}
CELL_WORKERS=${CELL_WORKERS:-3}
# Two runs share node gpu2 (4 GPUs each), so every port and Ray temp dir must
# differ between them; scripts/slurm/role_phase.sh scopes its Ray shutdown to
# its own temp dir for the same reason.
CELL_JUDGE_PORT=${CELL_JUDGE_PORT:-5310}
TRAIN_JUDGE_PORT=${TRAIN_JUDGE_PORT:-5311}
RAY_PORT=${RAY_PORT:-6311}
COORDINATOR=${COORDINATOR:-modal_role_lora_zero_sum_psro}
INIT_MODE=${INIT_MODE:-cold}
GENERATIONS=${GENERATIONS:-0}
GEN_ARG=""
[ "$GENERATIONS" != 0 ] && GEN_ARG="--generations $GENERATIONS"
JOB_NAME=${JOB_NAME:-${INIT_MODE}psro_$SUFFIX}

RUN_ROOT=$PSRO_ROOT/$SUFFIX
LOGDIR=${LOGDIR:-$RUN_ROOT/logs}
# A hot run starts at generation one, so it has nothing to seed: the
# coordinator writes its own contract and A0/D0 population on first entry.
FRESH_COLD_RUN=${FRESH_COLD_RUN:-0}
FRESH_ARG=""
[ "$FRESH_COLD_RUN" = 1 ] && FRESH_ARG="--fresh-cold-run"
if [ "$INIT_MODE" != hot ] && [ "$FRESH_COLD_RUN" != 1 ] && [ ! -f "$RUN_ROOT/state.json" ]; then
  echo "no seeded run at $RUN_ROOT/state.json; run seed_cold_psro_run.py first," >&2
  echo "or set FRESH_COLD_RUN=1 to let the coordinator write its own contract" >&2
  exit 1
fi
mkdir -p "$LOGDIR"

JUDGE=${JUDGE_MODEL_PATH:-$(ls -d "$HF_HOME_DIR"/hub/models--allenai--wildguard/snapshots/*/ | head -1)}

SCRIPT=$LOGDIR/cold_psro.sbatch
cat > "$SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=1
${NODELIST:+#SBATCH --nodelist=$NODELIST}
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --gres=$GRES
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME_LIMIT
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=$LOGDIR/driver_%j.log
#SBATCH --error=$LOGDIR/driver_%j.log

set -eu -o pipefail
# The login profile points TRANSFORMERS_CACHE at an incomplete cache and it
# silently overrides HF_HOME.
unset TRANSFORMERS_CACHE
echo "=== slurm job \$SLURM_JOB_ID restart=\${SLURM_RESTART_COUNT:-0} on \$(hostname) at \$(date -Is) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Refuse to start with no room to write: the run died at 18:00 on 2026-09-08
# because the scratch quota filled and every process lost its next write with
# no traceback. A cheap probe here turns that into a clear message.
if ! touch "$LOGDIR/.write_probe" 2>/dev/null; then
  echo "cannot write to $LOGDIR (quota?); refusing to start" >&2
  exit 2
fi
rm -f "$LOGDIR/.write_probe"

# The driver resumes from state.json on every entry, so a transient death (NFS
# stall, Ray teardown, node hiccup) is recoverable by re-entering it. Slurm
# only auto-requeues on preemption or node failure, never on exit 1, so retry
# in-band and keep the allocation.
ATTEMPTS=\${DRIVER_MAX_ATTEMPTS:-24}
for attempt in \$(seq 1 \$ATTEMPTS); do
  echo "=== driver attempt \$attempt/\$ATTEMPTS at \$(date -Is) ==="
  set +e
  $PYTHON $HERE/cold_psro_local.py \\
    --workspace $WORKSPACE \\
    --work $WORK \\
    --hf-home $HF_HOME_DIR \\
    --sft-dir $SFT_DIR \\
    --psro-root $PSRO_ROOT \\
    --raw-payoff-root $RAW_PAYOFF_ROOT \\
    --suffix $SUFFIX \\
    --python $PYTHON \\
    --judge-model $JUDGE \\
    --gpus $GPUS \\
    --cell-workers $CELL_WORKERS \\
    --coordinator $COORDINATOR \\
    --init-mode $INIT_MODE \\
    $FRESH_ARG \\
    $GEN_ARG \\
    --cell-judge-port $CELL_JUDGE_PORT \\
    --train-judge-port $TRAIN_JUDGE_PORT \\
    --ray-port $RAY_PORT
  rc=\$?
  set -e
  [ \$rc -eq 0 ] && exit 0
  echo "=== driver exited rc=\$rc; resuming from state.json in 60s ===" >&2
  sleep 60
done
echo "=== driver failed \$ATTEMPTS times; giving up ===" >&2
exit 1
EOF

# Accelerated payoff evaluation. RACE_CELLS=1 turns on the racing scheduler:
# every pending cell gets RACE_PILOT_EPISODES, the meta-strategy is solved on the
# pilot, and only the top RACE_M_ROW attackers x RACE_M_COL defenders (plus
# anything already carrying Nash weight above RACE_TAU) are extended to
# RACE_FULL_EPISODES. A reduced budget is an exact prefix of the full episode
# sequence, so a capped cell is the same estimand stopped early.
#   main runs      : RACE_CELLS=1 RACE_PILOT_EPISODES=256 RACE_M_ROW=2 RACE_M_COL=2
#   ultra-fast     : RACE_CELLS=1 RACE_PILOT_EPISODES=256 RACE_M_ROW=1 RACE_M_COL=1 \
#                    RACE_FULL_EPISODES=2000     (exploratory generations only)
for v in STEPS_PER_ROLE SAVE_STEPS_CONTRACT MAX_CKPT_NUM_CONTRACT MATRIX_EPISODES \
          LADDER_EARLY_STOP \
          ATK_PATIENCE ATK_MIN_DELTA ATK_MIN_STEP DEF_PATIENCE DEF_THRESHOLD \
          DEF_MIN_STEP EARLY_STOP_WINDOW PSRO_ATTACK_CACHE PSRO_PROMPT_SELECTION; do
  eval "val=\${$v:-}"; if [ -n "$val" ]; then export $v; fi
done
RACE_CELLS=${RACE_CELLS:-0}
RACE_PILOT_EPISODES=${RACE_PILOT_EPISODES:-256}
RACE_M_ROW=${RACE_M_ROW:-2}
RACE_M_COL=${RACE_M_COL:-2}
RACE_TAU=${RACE_TAU:-0.01}
export RACE_CELLS RACE_PILOT_EPISODES RACE_M_ROW RACE_M_COL RACE_TAU
[ -n "${RACE_FULL_EPISODES:-}" ] && export RACE_FULL_EPISODES
[ -n "${DRIVER_MAX_ATTEMPTS:-}" ] && export DRIVER_MAX_ATTEMPTS

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is not set; running with W&B disabled" >&2
  export WANDB_MODE=disabled
fi
SBATCH_ARGS=()
# DEPENDENCY chains the stages (cold train -> cold eval -> hot train -> hot eval)
# inside Slurm, so the sequence survives an SSH disconnect with no local agent.
[ -n "${DEPENDENCY:-}" ] && SBATCH_ARGS+=(--dependency="$DEPENDENCY")
sbatch --export=ALL "${SBATCH_ARGS[@]}" "$SCRIPT"

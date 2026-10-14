#!/usr/bin/env bash
# Container entrypoint: single-turn Cold PSRO, 8 generations, 8xH200, no Slurm.
# Replicates g2full.sbatch's env setup (minus the Slurm wrapper) and runs the driver with
# resume-on-crash. The driver spawns Ray + vLLM servers + judges + the training phase itself.
set -u -o pipefail
unset TRANSFORMERS_CACHE RAY_ADDRESS GC_FLAGS
HERE=/zfsauton/scratch/wentsec/coldpsro_slurm/psro_docker
source $HERE/psro.env
SUF=${RUN_TAG:-psro_docker}
PSRO_ROOT=$WORKSPACE/psro
RUN=$PSRO_ROOT/$SUF
mkdir -p $RUN $PSRO_ROOT/${SUF}_raw

# scratch writability (the quota killer on our cluster; harmless check for the co-worker)
touch $RUN/.write_probe 2>/dev/null || { echo "FATAL: $RUN not writable (mount a writable volume)"; exit 2; }

# per-run temp: short SSP_TMP (Ray AF_UNIX socket path < 107 bytes), own Ray dir, fresh caches
JID=$$
for d in /tmp/g2f_* /tmp/g2f; do [ -O "$d" ] && rm -rf "$d"; done
export SSP_TMP=/tmp/g2f RAY_TMPDIR=/tmp/ray_${JID}
mkdir -p $SSP_TMP/tmp $RAY_TMPDIR
export TMPDIR=$SSP_TMP/tmp VLLM_CACHE_ROOT=$SSP_TMP/vllm_cache TOPIC_API_CACHE=$SSP_TMP/orch_topic_${SUF}.sqlite3
export HF_HOME=$SSP_HF_HOME HF_DATASETS_CACHE=$SSP_TMP/hf_datasets_cache
export TRITON_CACHE_DIR=/tmp/triton_cache TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=16
export SHARED_BASE_CACHE=/dev/shm/${SUF}_sbmerged   # needs --shm-size >= 16g
for d in /dev/shm/ssp_rollout_lora_* $SHARED_BASE_CACHE; do [ -O "$d" ] && rm -rf "$d"; done
export ONEJOB_PORT_BASE=7900 RAY_PORT=6600 CELL_JUDGE_PORT=5600 TRAIN_JUDGE_PORT=5601
export GAME_DUMP=$RUN/game_dump.jsonl V7_SEED_STATS=$RUN/seed_stats.json SUFFIX=$SUF G2FULL_TAG=$SUF
export PATH="$ROLLOUT_ENV/bin:$PATH"    # ninja + toolchain for vLLM EngineCore JIT
export PYTHONPATH=$SSP_WORK:$SSP_REPO:${PYTHONPATH:-}

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || { echo "FATAL: no GPUs (run with --gpus all)"; exit 3; }
echo "=== LIVE CONFIG ==="; env | grep -E "^(GENERATIONS|STEPS_PER_ROLE|GPUS|V8_|KL_TARGET|PSRO_UNIFORM|TRAIN_BATCH|GROUP_SIZE|TRAINING_SEED)=" | sort | tr '\n' ' '; echo

# wandb: log metrics so they can be shared back. Needs WANDB_API_KEY (pass via -e).
if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "=== wandb: logging to project '$WANDB_PROJECT' (entity ${WANDB_ENTITY:-default}); share the run with us ==="
else
  export WANDB_MODE=offline
  echo "=== wandb: no WANDB_API_KEY -> MODE=offline. Runs saved under $RUN/wandb; 'wandb sync' them and share. ==="
fi

cd $SSP_REPO
for attempt in $(seq 1 ${DRIVER_MAX_ATTEMPTS:-8}); do
  echo "=== $SUF driver attempt $attempt at $(date -Is) ==="
  $SSP_PYTHON scripts/slurm/cold_psro_local.py \
      --workspace $WORKSPACE --work $SSP_WORK --hf-home $SSP_HF_HOME --sft-dir $SSP_SFT_DIR \
      --psro-root $PSRO_ROOT --raw-payoff-root $PSRO_ROOT/${SUF}_raw --suffix $SUF \
      --python $SSP_PYTHON --judge-model $JUDGE_MODEL_PATH --gpus $GPUS --cell-workers 3 \
      --coordinator modal_role_lora_zero_sum_psro --init-mode cold --fresh-cold-run \
      --generations $GENERATIONS --cell-judge-port $CELL_JUDGE_PORT \
      --train-judge-port $TRAIN_JUDGE_PORT --ray-port $RAY_PORT \
      2>&1 | tee -a $RUN/driver.log && { echo "=== driver done $(date -Is) ==="; break; }
  rc=$?; echo "=== driver exited rc=$rc at $(date -Is); cleaning GPU procs, resuming in 60s ==="
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do [ "$p" != "$$" ] && kill -9 $p 2>/dev/null; done
  sleep 60
done
rm -rf $SHARED_BASE_CACHE $RAY_TMPDIR $SSP_TMP 2>/dev/null
echo "=== PSRO finished; results under $RUN (state.json, matrices/, cells/, game_dump.jsonl) ==="

# --- share results back ---
echo "=== LoRA adapters (per oracle): $RUN/training/<ORACLE>/ckpt/global_step*_hf ==="
if [ -n "${HF_PUSH_REPO:-}" ]; then
  echo "=== pushing adapters to hf.co/$HF_PUSH_REPO ==="
  "$HERE/push_checkpoints.sh" "$SUF" || echo "WARN: push_checkpoints failed; run it manually: ./push_checkpoints.sh $SUF"
else
  echo "=== To share checkpoints: set HF_PUSH_REPO=<org>/<repo> + HF_TOKEN and run ./push_checkpoints.sh $SUF ==="
fi
echo "=== Please share: (1) the wandb run/report, (2) hf.co/\$HF_PUSH_REPO (LoRA adapters + state.json). ==="

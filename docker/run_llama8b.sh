#!/usr/bin/env bash
# Build/run the 4xH200 Llama-3.1-8B abliterated role-LoRA container.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${SSP_IMAGE:-safeselfplay:llama8b-h200-invalidmove-20260924-v5}
ACTION=${1:---help}
if [ "$#" -gt 0 ]; then shift; fi
case "$ACTION" in
  -h|--help|help)
    echo "Usage: $0 build | prepare | check | {psro|uniform|naive} --run-suffix NAME | role --role attacker"
    echo "Defaults: GPU_DEVICES=0,1,2,3; SSP_DOCKER_WORKSPACE=$REPO/output/docker-llama8b"
    exit 0 ;;
  build)
    # Stamp the commit being baked in. Without it a rebuilt image and a stale
    # one look identical, and the stale one reintroduces the bugs it was built
    # before -- the code is COPYed into the image, never mounted.
    REVISION=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
    if [ "$REVISION" = unknown ] && [ -f "$REPO/BUILD_BUNDLE.json" ]; then
      BUNDLE_REVISION=$(sed -n 's/.*"base_git_commit": "\([0-9a-f]*\)".*/\1/p' "$REPO/BUILD_BUNDLE.json")
      if [ -n "$BUNDLE_REVISION" ]; then REVISION=$BUNDLE_REVISION; fi
    fi
    if git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 &&
       ! git -C "$REPO" diff --quiet HEAD; then REVISION="$REVISION-dirty"; fi
    exec docker build --platform linux/amd64 -f "$REPO/docker/Dockerfile.llama8b" \
      --build-arg "SSP_GIT_SHA=$REVISION" \
      --build-arg "SSP_BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      -t "$IMAGE" "$@" "$REPO" ;;
  prepare|check|psro|uniform|naive|role) ;;
  *) echo "Unknown command: $ACTION" >&2; exit 2 ;;
esac

WORKSPACE=${SSP_DOCKER_WORKSPACE:-$REPO/output/docker-llama8b}
mkdir -p "$WORKSPACE"
WORKSPACE=$(cd "$WORKSPACE" && pwd)
OPTIONS=(run --rm --init --stop-timeout 180 --shm-size "${SSP_SHM_SIZE:-32g}"
         --ulimit nofile=65536:65536
         --mount "type=bind,source=$WORKSPACE,target=/workspace")
# Downloads and dry-runs do not require GPU allocation.
if [ "$ACTION" != prepare ] && { [ "${DRY_RUN:-0}" != 1 ] || [ "$ACTION" = check ]; }; then
  OPTIONS+=(--gpus "\"device=${GPU_DEVICES:-0,1,2,3}\"")
fi
if [ "${DETACH:-0}" = 1 ]; then OPTIONS+=(-d); fi
if [ -n "${CONTAINER_NAME:-}" ]; then OPTIONS+=(--name "$CONTAINER_NAME"); fi

# Forward variable NAMES only: keys do not appear in Docker's argv or build.
for name in HF_TOKEN HF_CHECKPOINT_REPO HF_CHECKPOINT_PRIVATE \
            WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_GROUP WANDB_MODE \
            GAME_REWARD_MODE ATTACKER_INVALID_MOVE TOPIC_PENALTY TOPIC_JUDGE_BACKEND TOPIC_LOCAL_MODEL \
            TOPIC_JUDGE_PORT TOPIC_JUDGE_UTIL TOPIC_JUDGE_STARTUP_ATTEMPTS \
            TOPIC_JUDGE_GPU TOPIC_JUDGE_MAX_LEN JUDGE_GPU \
            TOPIC_API_PROVIDER TOPIC_API_BASE_URL TOPIC_API_MODEL \
            TOPIC_API_KEY TOPIC_API_TIMEOUT TOPIC_API_ATTEMPTS TOPIC_API_CONCURRENCY \
            TOPIC_API_BATCH_TIMEOUT TOPIC_API_CACHE \
            RUN_SUFFIX GENERATIONS STEPS_PER_ROLE MATRIX_EPISODES STEPS \
            DRY_RUN INIT_KL_COEF PSRO_MIDRUN_RESUME \
            TRAINING_SEED PROMPT_MAX_LEN GENERATE_MAX_LEN TRAIN_BATCH ROLLOUT_BATCH \
            OFFTOPIC_ENABLE OFFTOPIC_UTIL OFFTOPIC_PENALTY OFFTOPIC_TIMEOUT_SECONDS \
            UTIL OPP_UTIL JUDGE_UTIL VLLM_SLEEP DS_SLEEP MICRO_TRAIN MICRO_ROLLOUT \
            SAVE_STEPS SAVE_STEPS_CONTRACT RACE_CELLS RACE_FULL_EPISODES \
            DRIVER_MAX_ATTEMPTS DRIVER_RETRY_WAIT CELL_JUDGE_GPU_FREE_FRACTION CELL_JUDGE_RELEASE_TIMEOUT \
            MATRIX_WAVE_PAIRS MATRIX_GENERATION_BATCH MATRIX_JUDGE_BATCH CELL_ENFORCE_EAGER \
            CELL_STREAMING CELL_SHARE_JUDGE_GPUS CELL_MAX_NUM_SEQS CELL_STREAM_WAVE_PAIRS CELL_ENGINE_UTILS CELL_JUDGE_THREADS \
            LADDER_EARLY_STOP EARLY_STOP_WINDOW EARLY_STOP_CKPT_WAIT \
            ATK_MIN_DELTA ATK_PATIENCE ATK_MIN_STEP ATK_CEILING_MAX \
            DEF_THRESHOLD DEF_GUARD_MIN DEF_PATIENCE DEF_MIN_STEP \
            ATTACKER_OBJECTIVE EXACT_LABEL_BALANCE LABEL_DRIFT_POLICY \
            OPPONENT OPPONENT_POOL_JSON START_ADAPTER HF_HUB_OFFLINE; do
  if [[ -v $name ]]; then OPTIONS+=(-e "$name"); fi
done
if [ "${TOPIC_JUDGE_BACKEND:-local}" != local ] && [ -n "${TOPIC_API_KEY_FILE:-}" ]; then
  [ -f "$TOPIC_API_KEY_FILE" ] || { echo "TOPIC_API_KEY_FILE is not a regular file" >&2; exit 2; }
  KEY_FILE=$(realpath "$TOPIC_API_KEY_FILE")
  OPTIONS+=(--mount "type=bind,source=$KEY_FILE,target=/run/secrets/topic_api_key,readonly"
            -e TOPIC_API_KEY_FILE=/run/secrets/topic_api_key)
fi
exec docker "${OPTIONS[@]}" "$IMAGE" "$ACTION" "$@"

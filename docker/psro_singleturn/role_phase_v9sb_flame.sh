#!/usr/bin/env bash
# Train one role-LoRA oracle on a single Slurm node, without Modal.
#
# Mirrors the argument set that modal_upstream_selfredteam_role_lora_v2.py passes
# to openrlhf.cli.train_ppo_ray, including the postfill-CoT SFT mixture. An
# attacker phase suppresses the defender turn and serves a frozen defender from
# the opponent adapter; a defender phase still runs the attacker turn, and
# optimizer_train_role decides which replay items receive updates.
#
#   ROLE=attacker GEN=1 scripts/slurm/role_phase.sh
#   ROLE=defender GEN=1 OPPONENT=$SSP_ROOT/psro/A1/ckpt/global_step100_hf \
#       scripts/slurm/role_phase.sh
#
# Set DRY_RUN=1 to print the resolved trainer command and exit.
set -eu -o pipefail

# --- workspace ---------------------------------------------------------------
# SSP_ROOT holds work/ (patched upstream tree), hf/ (HF cache), sft_data/ and
# psro/ (outputs). Populate the first two with scripts/slurm/prepare_sources.py
# and scripts/slurm/prepare_sft_data.py.
SSP_ROOT=${SSP_ROOT:?set SSP_ROOT to the workspace directory}
WORK=${SSP_WORK:-$SSP_ROOT/work}
HF_ROOT=${SSP_HF_HOME:-$SSP_ROOT/hf}
if [ "${USE_LIGER:-}" = 1 ]; then export PYTHONPATH=${LIGER_PATH:?}${PYTHONPATH:+:$PYTHONPATH}; fi
SFT_DIR=${SSP_SFT_DIR:-$SSP_ROOT/sft_data}
PSRO=${SSP_OUT:-$SSP_ROOT/psro}
PY=${SSP_PYTHON:-python}
# This repository: role_phase.sh lives in scripts/slurm/ inside it, so derive
# the root from $0 rather than requiring yet another exported variable.
SSP_REPO=${SSP_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
GPUS=${GPUS:-4}
# DeepSpeed JIT-builds FusedAdam on first use and that needs `ninja`, which
# lives in the interpreter's own bin rather than on the login PATH; without it
# the actor dies with "Ninja is required to load C++ extensions". The cold-PSRO
# driver already prepends this directory, so a phase launched through it worked
# while the same phase launched from submit_ladder.sh did not.
case ":$PATH:" in
  *":$(dirname "$PY"):"*) ;;
  *) PATH="$(dirname "$PY"):$PATH"; export PATH ;;
esac
# `ray` must come from the same interpreter as $PY; a stale ~/.local/bin/ray
# earlier on PATH shadows it and dies with "required file not found".
RAY_BIN=${RAY_BIN:-}
if [ -z "$RAY_BIN" ]; then
  _candidate=$(dirname "$PY")/ray
  if [ -x "$_candidate" ]; then RAY_BIN=$_candidate; else RAY_BIN=ray; fi
fi

ROLE=${ROLE:?attacker|defender}
GEN=${GEN:-1}
STEPS=${STEPS:-100}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-64}
OPPONENT=${OPPONENT:-}            # frozen opponent adapter; empty means base only
# Cold PSRO best-responds to a frozen Nash mixture rather than one adapter, so
# the pool is passed through verbatim; it is mutually exclusive with OPPONENT.
OPPONENT_POOL_JSON=${OPPONENT_POOL_JSON:-}
START_ADAPTER=${START_ADAPTER:-}  # warm-start this role instead of cold-starting

# Cold-PSRO parity switches. role_lora_v2.py always sets these two custom
# configs; the ladder predates them, so they stay opt-in and the PSRO driver
# turns them on.
EXACT_LABEL_BALANCE=${EXACT_LABEL_BALANCE:-0}
LABEL_DRIFT_POLICY=${LABEL_DRIFT_POLICY:-}
# Attacker objective: empty keeps the upstream observed-label branching;
# "intended-prompt-type-v5" scores the attacker against its assigned
# prompt_type and floors a relabelled prompt at ATTACKER_LABEL_DRIFT_PENALTY.
ATTACKER_OBJECTIVE=${ATTACKER_OBJECTIVE:-}
# Defender responses carry a CoT block, so a long compliance can reach the
# generation cap and be truncated mid-tag, which scores as a format
# violation.  Raise GENERATE_MAX_LEN to give it headroom; the default is
# left at the upstream 2048 so existing runs stay comparable.
PROMPT_MAX_LEN=${PROMPT_MAX_LEN:-2048}
GENERATE_MAX_LEN=${GENERATE_MAX_LEN:-2048}

# --- weights & biases ---------------------------------------------------------
# Same org/project/group/run-id scheme as modal_upstream_selfredteam_role_lora_v2
# so a requeued phase reattaches to its existing run instead of forking a new
# one. The key is read from WANDB_API_KEY only and is never written to disk.
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_PROJECT=${WANDB_PROJECT:-self-play}
WANDB_GROUP=${WANDB_GROUP:-upstream-selfredteam-role-lora-v2}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}

# --- memory -------------------------------------------------------------------
# vLLM's gpu_memory_utilization is a PRIVATE per-engine fraction of the whole
# card, not a cap on total card usage. The judge, the trainable engines, the
# frozen-opponent engines and the ZeRO-3 actor must therefore sum below 1.0.
# Measured on H200-141GB with the judge colocated on GPU 0:
#   judge 0.13 + trainable 0.28 + opponent 0.26 + actor (rest) ~= 0.93
# The judge floor is ~0.12; wildguard-7B weights alone are 14.3G.
# Disaggregated mode: rollout engines and both judges on cards 0..R-1, the
# trainer (actor + ref) on the rest as whole cards.
DISAGG_ROLLOUT_GPUS=${DISAGG_ROLLOUT_GPUS:-0}
export DISAGG_ROLLOUT_GPUS
if [ "$DISAGG_ROLLOUT_GPUS" -gt 0 ]; then
  TRAIN_GPUS=$((GPUS - DISAGG_ROLLOUT_GPUS))
  ENGINES=$DISAGG_ROLLOUT_GPUS
  JUDGE_GPU=${JUDGE_GPU:-0}
  TOPIC_JUDGE_GPU=${TOPIC_JUDGE_GPU:-$((DISAGG_ROLLOUT_GPUS - 1))}
  if [ "$DISAGG_ROLLOUT_GPUS" -eq 1 ]; then
    # one card holds both engines and both judges
    UTIL=${UTIL:-0.30}; OPP_UTIL=${OPP_UTIL:-0.30}; JUDGE_UTIL=${JUDGE_UTIL:-0.14}
  else
    UTIL=${UTIL:-0.38}; OPP_UTIL=${OPP_UTIL:-0.38}; JUDGE_UTIL=${JUDGE_UTIL:-0.13}
  fi
  if [ "${UNIFIED_ENGINE:-0}" = 1 ]; then
    # one engine per rollout card serves both roles; judges keep ~0.13 each
    UTIL=${UNIFIED_UTIL:-0.70}; OPP_UTIL=${OPP_UTIL:-0.10}
  fi
  export UNIFIED_ENGINE ENGINE_OLD_LOGP ATTACKER_ENGINES ATTACK_AHEAD
  COLOCATE_FLAG=--colocate_actor_ref
  if [ "${ROLLOUT_SERVER:-0}" = 1 ]; then
    # external vLLM servers on the rollout cards; no engines inside Ray
    ROLLOUT_CARDS=$DISAGG_ROLLOUT_GPUS
    SERVER_UTIL=${SERVER_UTIL:-0.70}
    ENGINES=0
    export DISAGG_ROLLOUT_GPUS=0 UNIFIED_ENGINE=0
  fi
else
  TRAIN_GPUS=$GPUS
  COLOCATE_FLAG=--colocate_all_models
fi
UTIL=${UTIL:-0.28}
OPP_UTIL=${OPP_UTIL:-0.26}
JUDGE_UTIL=${JUDGE_UTIL:-0.13}

# Which physical GPU the judges (WildGuard, and the local topic judge) run on.
# 0 colocates them with the trainable engines, which is what every run before
# the topic judge did and what the budget above is measured for.  That budget
# has no room for a second 8B judge: 0.13 + 0.28 + 0.26 + 0.24 = 0.91 leaves
# the ZeRO-3 actor ~0.09 and the run dies with "No available memory for the
# cache blocks".  Allocate one extra card (--gres one above GPUS) and set
# JUDGE_GPU to it: ray is started with --num-gpus=$GPUS so it never hands that
# card out, and both judges then have it to themselves.
JUDGE_GPU=${JUDGE_GPU:-0}

# Which card the local topic judge runs on.  Give it a DIFFERENT card from
# WildGuard rather than stacking both on GPU 0: vLLM's gpu_memory_utilization
# is measured against the whole card and it subtracts what other processes
# already hold there, so a second engine on the same card must be given a
# fraction that covers the first one's usage as well.  Stacking WildGuard
# (0.35 -> 49G, 269k KV tokens) and the topic judge (0.45 -> 63G budget) left
# the topic judge 14G for 16G of weights and it died with "No available memory
# for the cache blocks".  One judge per card keeps every fraction incremental.
# Default to the card after WildGuard's, so the two judges never share one: a
# second engine on an occupied card must be given a fraction covering the first
# one's usage as well, and the natural-looking 0.13 + 0.13 then starves it.  A
# single-GPU job has nowhere else to put it and falls back to sharing.
if [ "${GPUS:-4}" -gt 1 ]; then
  TOPIC_JUDGE_GPU=${TOPIC_JUDGE_GPU:-$(( (JUDGE_GPU + 1) % GPUS ))}
else
  TOPIC_JUDGE_GPU=${TOPIC_JUDGE_GPU:-$JUDGE_GPU}
fi
# Judge prompts are the contract plus one seed and one rewrite, and the reply
# is capped at 24 tokens, so 8192 buys nothing over 4096 and costs 4G of KV.
TOPIC_JUDGE_MAX_LEN=${TOPIC_JUDGE_MAX_LEN:-4096}

# --- throughput ---------------------------------------------------------------
# Generation is ~93% of a step, so engine-side overhead dominates and spare HBM
# buys nothing (385-token prompts, ~120-token responses). Measured per step on
# 4xH200: 56.1s -> 22.2s by dropping the sleep flags and using TP=1, -> 21.2s
# with prefix caching, -> 18.7s with CUDA graphs.
#
# Do NOT add --vllm_enable_sleep/--deepspeed_enable_sleep here: with the judge
# colocated they cost ~110s/step, because deepspeed offloads optimizer state to
# CPU on every step. They are only worthwhile when the judge has its own GPUs.
ENGINES=${ENGINES:-$GPUS}   # colocated mode asserts actor_gpus == ENGINES*TP
TP=${TP:-1}                 # TP>1 only adds an all-reduce per decode step for 8B
EAGER=${EAGER:-0}           # 1 restores --enforce_eager (disables CUDA graphs)
PREFIX=${PREFIX:-1}         # reuse the shared prompt prefix across calls
MICRO_ROLLOUT=${MICRO_ROLLOUT:-32}
MICRO_TRAIN=${MICRO_TRAIN:-8}          # train_batch_size / GPUS
TRAIN_BATCH=${TRAIN_BATCH:-32}
ROLLOUT_BATCH=${ROLLOUT_BATCH:-128}

# Colocated mode keeps the trainable engines resident during the optimizer
# step. That is the fastest option when the frozen opponent is a single
# adapter, but a PSRO mixture engine plus the judge on GPU 0 leaves the ZeRO-3
# actor short and it OOMs mid-step. VLLM_SLEEP releases the trainable engines
# for the duration of the step, which is what the Modal runs did.
VLLM_SLEEP=${VLLM_SLEEP:-0}
DS_SLEEP=${DS_SLEEP:-0}

ENGINE_PERF=()
[ "$EAGER" = 1 ] && ENGINE_PERF+=(--enforce_eager)
[ "$PREFIX" = 1 ] && ENGINE_PERF+=(--enable_prefix_caching)
[ "$VLLM_SLEEP" = 1 ] && ENGINE_PERF+=(--vllm_enable_sleep)
[ "$DS_SLEEP" = 1 ] && ENGINE_PERF+=(--deepspeed_enable_sleep)

# --- role-specific settings, per role_lora_v2.py ------------------------------
case $ROLE in
  attacker)
    LR=${LR:-1e-5}; SFT_STOP=${SFT_STOP:-30}
    SFT_DATA=$SFT_DIR/attacker_rewrite_1180_rl_continuation.jsonl
    SFT_PROBS=1.0; SFT_KEYS=() ;;
  defender)
    LR=${LR:-4e-5}; SFT_STOP=${SFT_STOP:-10}
    SFT_DATA=$SFT_DIR/defender_rl_sft/helpsteer3_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl,$SFT_DIR/defender_rl_sft/vanilla_benign_8b_T_0.6_topp_0.9_wgclean_postfill_cot_15000.jsonl
    SFT_PROBS=0.5,0.5
    SFT_KEYS=(--sft_input_key vanilla --sft_output_key completion) ;;
  *) echo "ROLE must be attacker|defender" >&2; exit 1 ;;
esac
for f in ${SFT_DATA//,/ }; do
  [ -f "$f" ] || { echo "missing SFT file: $f (run prepare_sft_data.py)" >&2; exit 1; }
done

# --target_modules is nargs="*" with a string default, so an unset value is
# iterated character-wise and PEFT looks for modules named {"a","l","-",...}.
LORA_TARGET_MODULES=(q_proj k_proj v_proj o_proj gate_proj up_proj down_proj)

TAG=${TAG:-${ROLE:0:1}${GEN}}
OUT=$PSRO/$TAG
mkdir -p "$OUT"

# --- environment --------------------------------------------------------------
unset TRANSFORMERS_CACHE            # overrides HF_HOME if a login shell set it
export HF_HOME=$HF_ROOT HF_HUB_CACHE=$HF_ROOT/hub
export TMPDIR=${TMPDIR:-$SSP_ROOT/tmp} TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
# Both are set by role_lora_v2.py immediately before `ray start`: the v1
# EngineCore subprocess exits silently under the LoRA path, and NCCL's cuMem
# allocator conflicts with vLLM's memory pool.
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_MODE=${WANDB_MODE:-disabled}
# The patched vllm_worker_wrap imports roll.third_party.vllm.TensorLoRARequest;
# Modal mounts the repo for this, so add both trees to the path here.
REPO_ROOT=${SSP_REPO}
export PYTHONPATH=$WORK:$REPO_ROOT:${PYTHONPATH:-}

BASE=${BASE_MODEL_PATH:-$(ls -d "$HF_ROOT"/hub/models--mlabonne--Meta-Llama-3.1-8B-Instruct-abliterated/snapshots/*/ | head -1)}
JUDGE=${JUDGE_MODEL_PATH:-$(ls -d "$HF_ROOT"/hub/models--allenai--wildguard/snapshots/*/ | head -1)}
IP=$(hostname -I | awk '{print $1}')
JUDGE_PORT=${JUDGE_PORT:-5100}
# Superseded local verifier, reachable only in GAME_REWARD_MODE=legacy.  It is
# off by default: it does not feed the shared training/matrix reward, and its
# 0.24 of GPU 0 on top of the policy, opponent and judge engines is what pushes
# the card past 1.0 and fails the run with "No available memory for the cache
# blocks".  Set OFFTOPIC_ENABLE=1 only when reproducing a pre-shared_binary run.
OFFTOPIC_ENABLE=${OFFTOPIC_ENABLE:-0}
OFFTOPIC_PORT=${OFFTOPIC_PORT:-$((JUDGE_PORT + 200))}
OFFTOPIC_UTIL=${OFFTOPIC_UTIL:-0.24}
# Which labeler answers "does this rewrite still ask what the seed asked?" in
# GAME_REWARD_MODE=shared_binary.  `local` runs the base model behind an
# OpenAI-compatible shim and needs no credential; `anthropic` calls Opus 5.
TOPIC_JUDGE_BACKEND=${TOPIC_JUDGE_BACKEND:-local}
TOPIC_JUDGE_PORT=${TOPIC_JUDGE_PORT:-$((JUDGE_PORT + 300))}
# Measured on H200-143771MiB with 4 GPUs, one judge per card: WildGuard and the
# topic judge each occupy 16995 and 16959 MiB, i.e. 0.118 of a card.  0.13 is
# that plus headroom, and matches JUDGE_UTIL, which has carried WildGuard since
# the first run.  The old 0.24 dates from the superseded verifier and is what
# made the card overflow when it was stacked on top of WildGuard.
TOPIC_JUDGE_UTIL=${TOPIC_JUDGE_UTIL:-0.13}
RAY_PORT=${RAY_PORT:-6100}
RM_URL=${EXTERNAL_JUDGE_URL:-http://$IP:$JUDGE_PORT/classify}
cd "$WORK"

# Does every frozen opponent that carries weight have an adapter?  Only then is
# it safe (and correct) to render the frozen defender with its CoT scaffold:
# the base model never learned the template and prefilling it is a jailbreak.
# A trained frozen defender is rendered WITH the scaffold in every payoff cell
# and benchmark, so attacker training must match.
# v9c (10-08): FORCE_DEFENDER_COT_FROZEN=1 renders EVERY frozen defender with the CoT scaffold, D0 included.  With
# one adapter-less D0 in the pool the old rule rendered ALL frozen defenders bare; under FORMAT_GRAMMAR (no EOS before
# </answer>) a bare defender cannot close its think, ~50% of turns ran to the 1792-token budget and were judged on a
# forced 256-token answer, while payoff cells render D0 WITH the scaffold (3% forced).  Training now matches the cells.
DEFENDER_COT_FROZEN=$($PY - "${OPPONENT_POOL_JSON:-}" "${OPPONENT:-}" "${FORCE_DEFENDER_COT_FROZEN:-0}" <<'PYPOOL'
import json, sys
pool_json, single = sys.argv[1], sys.argv[2]
if sys.argv[3] == "1":
    print("1"); sys.exit(0)
if pool_json:
    try:                       # the flag is a path in normal use
        pool = json.load(open(pool_json))
    except (OSError, ValueError):   # ... but accept inline JSON too
        pool = json.loads(pool_json)
    weighted = [e for e in pool if float(e.get("probability", 0.0)) > 1e-9]
    ok = bool(weighted) and all(e.get("adapter") for e in weighted)
else:
    ok = bool(single)          # a single named opponent adapter, or base -> False
print("1" if ok else "0")
PYPOOL
)

CUSTOM=$($PY - "$ROLE" "$SFT_STOP" "$EXACT_LABEL_BALANCE" "$LABEL_DRIFT_POLICY" \
        "${ACTOR_LR_SCHEDULER:-constant_with_warmup}" "$ATTACKER_OBJECTIVE" \
        "$DEFENDER_COT_FROZEN" <<'PY'
import json, sys
from roll.utils.shared_game_reward import shared_game_reward_enabled
role, sft_stop, balance, drift, scheduler, attacker_objective, defender_cot_frozen = sys.argv[1:8]
config = {
    "max_turns": 2, "reward_type": "general_sum", "remove_ties": True,
    "redistribute_after_ties": True,
    "no_defender_turn": role == "attacker",
    # Render the frozen defender WITH its CoT wrapper + prefill during attacker
    # training, matching how every payoff cell and benchmark renders a trained
    # defender.  no_defender_turn also selects which turns are optimizer items,
    # so it cannot carry this meaning too -- hence a separate key.  The base
    # model (D0) never learned the template and prefilling it is a known
    # jailbreak, so the renderer keeps the bare wrapper when the frozen
    # opponent has no adapter.
    "defender_cot_when_frozen": defender_cot_frozen == "1",
    "no_attacker_turn": False,
    "optimizer_train_role": role,
    "fixed_defender_from_opponent_vllm": role == "attacker",
    "fixed_attacker_from_opponent_vllm": role == "defender",
    "actor_lr_scheduler": scheduler,
    "postfill_cot_stop_after_step": int(sft_stop),
}
if balance == "1":
    config["exact_prompt_label_balance"] = True
if drift:
    config["label_drift_training_policy"] = drift
if attacker_objective:
    config["attacker_objective"] = attacker_objective
if shared_game_reward_enabled():
    config["remove_ties"] = False
    config["redistribute_after_ties"] = False
print(json.dumps(config))
PY
)

# The frozen opponent always needs a base model; the adapter is layered on from
# the first phase that faces a trained opponent.
OPP_ARGS=(--fixed_opponent_pretrain "$BASE"
          --fixed_opponent_vllm_gpu_memory_utilization "$OPP_UTIL")
if [ -n "$OPPONENT" ] && [ -n "$OPPONENT_POOL_JSON" ]; then
  echo "OPPONENT and OPPONENT_POOL_JSON are mutually exclusive" >&2; exit 1
fi
[ -n "$OPPONENT" ] && OPP_ARGS+=(--fixed_opponent_lora_path "$OPPONENT"
                                --fixed_opponent_lora_rank "$LORA_RANK")
[ -n "$OPPONENT_POOL_JSON" ] && OPP_ARGS+=(--fixed_opponent_pool_json "$OPPONENT_POOL_JSON"
                                           --fixed_opponent_lora_rank "$LORA_RANK")
[ "${ROLLOUT_SERVER:-0}" = 1 ] && OPP_ARGS=()
START_ARGS=(); [ -n "$START_ADAPTER" ] && START_ARGS=(--role_start_adapter "$START_ADAPTER")

# W&B run identity must be exported before `ray start`: Ray workers inherit the
# raylet environment, and ppo_actor.py reads WANDB_RUN_ID/WANDB_RESUME there.
WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_RUN_NAME=${WANDB_RUN_NAME:-$(basename "$PSRO")__$TAG}
  WANDB_RUN_ID=$(printf '%s' "$WANDB_RUN_NAME" | sha1sum | cut -c1-8)
  export WANDB_RUN_ID
  export WANDB_RESUME=${WANDB_RESUME:-allow}
  export WANDB_MODE=${WANDB_MODE:-online}
  export WANDB_DIR=${WANDB_DIR:-$OUT}
  WANDB_ARGS=(--use_wandb "$WANDB_API_KEY"
              --wandb_project "$WANDB_PROJECT"
              --wandb_group "$WANDB_GROUP" --wandb_run_name "$WANDB_RUN_NAME"
              --wandb_max_log 10000 --wandb_table_log_interval 1
              --wandb_table_csv_path "$OUT/run_tables")
  [ -z "$WANDB_ENTITY" ] || WANDB_ARGS+=(--wandb_org "$WANDB_ENTITY")
  echo "$TAG wandb entity=${WANDB_ENTITY:-account-default} project=$WANDB_PROJECT run=$WANDB_RUN_ID"
fi

TRAIN_CMD=("$PY" -m openrlhf.cli.train_ppo_ray
  --actor_num_nodes 1 --actor_num_gpus_per_node "$TRAIN_GPUS"
  --ref_num_nodes 1 --ref_num_gpus_per_node "$TRAIN_GPUS"
  --remote_rm_url "$RM_URL"
  --vllm_num_engines "$ENGINES" --vllm_tensor_parallel_size "$TP"
  $COLOCATE_FLAG --vllm_gpu_memory_utilization "$UTIL"
  --pretrain "$BASE" --save_path "$OUT" --ckpt_path "$OUT/ckpt"
  --save_steps "${SAVE_STEPS:-10}" --save_hf_ckpt --disable_ds_ckpt --max_ckpt_num "${MAX_CKPT_NUM:-1}"
  --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA"
  --target_modules "${LORA_TARGET_MODULES[@]}"
  "${OPP_ARGS[@]}" "${START_ARGS[@]}"
  --micro_train_batch_size "$MICRO_TRAIN" --train_batch_size "$TRAIN_BATCH"
  --micro_rollout_batch_size "$MICRO_ROLLOUT" --rollout_batch_size "$ROLLOUT_BATCH"
  --prompt_data "red_team/data/vanilla_harmful_dataset.jsonl, red_team/data/vanilla_benign_dataset.jsonl"
  --prompt_data_probs "0.5, 0.5"
  --eval_data "red_team/data/1k_vanilla_harmful_prompts_holdout.jsonl"
  --max_samples $((STEPS * ROLLOUT_BATCH)) --max_epochs 1
  --prompt_max_len "${PROMPT_MAX_LEN:-2048}" --generate_max_len "${GENERATE_MAX_LEN:-2048}"
  --zero_stage "${ZERO_STAGE:-3}" --num_episodes 1 --bf16 --seed "${TRAINING_SEED:-8888}" --top_p 1.0
  ${GC_FLAGS-"--gradient_checkpointing" "--gradient_checkpointing_use_reentrant"}
  # Reverse-KL anchor to the BASE model, on by default for BOTH roles at
  # coef 0.01. INIT_KL_COEF>0 turns the monitored reference KL into a loss term. The
  # estimator is k1 over policy samples, E_pi[log pi - log pi_ref], i.e.
  # REVERSE KL(pi || pi_ref). The reference is built with
  # --role_start_adapter, so it equals the BASE model only when that is
  # unset -- true for a cold oracle with mid-run resume disabled. Hot start
  # or a resumed segment would anchor to the round start instead, which the
  # driver now refuses when a penalty is active.
  ${REF_KL_FLAG-"--monitor_reference_kl"} --temperature 1.0 --lr_warmup_ratio "${LR_WARMUP_RATIO:-0.03}"
  --sft_data "$SFT_DATA" --sft_data_probs "$SFT_PROBS"
  --sft_steps 1 --sft_batches_per_step "${SFT_BATCHES_PER_STEP:-1}" "${SFT_KEYS[@]}"
  --actor_learning_rate "$LR" --init_kl_coef "${INIT_KL_COEF:-0.01}"
  # KL_TARGET: adaptive KL coefficient toward this KL (v9, 2026-10-08, after the r6bgate entropy blow-up)
  ${KL_TARGET:+--kl_target "$KL_TARGET" --kl_horizon "${KL_HORIZON:-2560}"}
  # USE_LIGER=1: fused RMSNorm/SwiGLU/RoPE kernels (liger-kernel from a side dir)
  ${USE_LIGER:+--use_liger_kernel}
  # KL_REF=adapter_off: reverse-KL loss (k3) to pi_0 computed by switching the
  # LoRA adapter off in the actor; no reference model is placed.
  ${KL_REF:+--use_kl_loss --kl_estimator "${KL_ESTIMATOR:-k3}"}
  --normalize_reward --packing_samples ${NO_ADV_STD_NORM:+--no_advantage_std_norm}
  --advantage_estimator reinforce
  --custom_configs "$CUSTOM"
  --actor_loss_coef 1.0 --postfill_cot_loss_coef 1.0
  --eval_steps "${EVAL_STEPS:-100000}" --eval_start_steps "${EVAL_START_STEPS:-100000}"
  --diversity_score_steps "${DIVERSITY_STEPS:-100000}"
  --vllm_sync_backend nccl "${ENGINE_PERF[@]}" --flash_attn
  "${WANDB_ARGS[@]}")

# ATTACKER_INVALID_MOVE=1 keeps the legacy (v6d) reward and floors invalid
# attacker moves (label drift, off-topic) at -3/-4; see
# roll/utils/role_lora_training_reward.py.  It needs the same topic judge as
# shared_binary.  Exported before `ray start` so the workers see it.
export ATTACKER_INVALID_MOVE=${ATTACKER_INVALID_MOVE:-0}
if [ "$ATTACKER_INVALID_MOVE" = 1 ] && [ "${GAME_REWARD_MODE:-legacy}" != legacy ]; then
  echo "$TAG ATTACKER_INVALID_MOVE=1 applies to GAME_REWARD_MODE=legacy only" >&2
  exit 1
fi
NEEDS_TOPIC_JUDGE=0
if [ "${GAME_REWARD_MODE:-legacy}" = shared_binary ] || [ "${GAME_REWARD_MODE:-legacy}" = v7 ] || [ "$ATTACKER_INVALID_MOVE" = 1 ]; then
  NEEDS_TOPIC_JUDGE=1
fi

# Point TOPIC_API_* at the selected labeler.  This must run before the reward
# contract check below: shared_game_reward_contract() resolves the topic judge
# configuration, so leaving it until the service launch made the default local
# backend fail with "Set TOPIC_API_MODEL and TOPIC_API_KEY" before any service
# started.  Configuring here is safe -- resolving the config makes no request.
if [ "$NEEDS_TOPIC_JUDGE" = 1 ]; then
  case "$TOPIC_JUDGE_BACKEND" in
    local)
      if [ "${TOPIC_JUDGE_MANAGED:-0}" = 1 ]; then
        # The Docker supervisor owns a stable endpoint shared by all phases
        # and PSRO cells. Preserve its model identity and URL in the contract.
        : "${TOPIC_API_BASE_URL:?managed topic endpoint is missing}"
        : "${TOPIC_API_MODEL:?managed topic model is missing}"
        export TOPIC_API_PROVIDER=openai_compatible TOPIC_API_KEY=local
      else
        # One base-model engine sharing GPU 0 with WildGuard, behind an
        # OpenAI-compatible shim so the same client, cache and contract serve both
        # backends.  Recall is the known weakness (see README); it is the default
        # because it costs nothing and needs no key.
        export TOPIC_API_PROVIDER=openai_compatible
        export TOPIC_API_BASE_URL="http://$IP:$TOPIC_JUDGE_PORT/v1"
        export TOPIC_API_MODEL=local-topic-judge
        export TOPIC_API_KEY=local            # unused; the client always sends one
      fi
      ;;
    anthropic)
      # Higher recall, one paid call per unlabeled (seed, rewrite) pair; the
      # sqlite cache in topic_api.py means a resumed run re-pays for nothing.
      export TOPIC_API_PROVIDER=anthropic
      : "${TOPIC_API_MODEL:=claude-opus-5}"; export TOPIC_API_MODEL
      if [ -z "${TOPIC_API_KEY:-}" ] && [ -z "${TOPIC_API_KEY_FILE:-}" ]; then
        echo "$TAG TOPIC_JUDGE_BACKEND=anthropic needs TOPIC_API_KEY or TOPIC_API_KEY_FILE" >&2
        exit 1
      fi
      ;;
    *)
      echo "$TAG TOPIC_JUDGE_BACKEND must be 'local' or 'anthropic' (got '$TOPIC_JUDGE_BACKEND')" >&2
      exit 1
      ;;
  esac
fi

if [ "${GAME_REWARD_MODE:-legacy}" = shared_binary ]; then
  REWARD_LIMIT=$($PY -c 'from roll.utils.shared_game_reward import topic_penalty; print(1 + topic_penalty())')
  TRAIN_CMD+=(--reward_clip_range "-$REWARD_LIMIT" "$REWARD_LIMIT")
fi

if [ "${DRY_RUN:-0}" = 1 ]; then
  for arg in "${TRAIN_CMD[@]}"; do
    if [ -n "${WANDB_API_KEY:-}" ] && [ "$arg" = "$WANDB_API_KEY" ]; then
      printf '%q ' '<redacted>'
    else
      printf '%q ' "$arg"
    fi
  done; echo
  exit 0
fi

if [ "${GAME_REWARD_MODE:-legacy}" = shared_binary ]; then
  "$PY" -c 'from roll.utils.shared_game_reward import shared_game_reward_contract; shared_game_reward_contract()'
  # The shared reward reaches training only through a patch that
  # prepare_sources.py injects into the working copy of language_game.py. A
  # work tree prepared before that patch existed trains on the LEGACY reward
  # while every banner and contract still says shared_binary -- silently wrong
  # results, which is far worse than a failed run. Verify the live file.
  GAME_LOOP=$WORK/openrlhf/trainer/ppo_utils/language_game.py
  for marker in compute_shared_game_reward attacker_cot_format_violation; do
    if ! grep -q "$marker" "$GAME_LOOP" 2>/dev/null; then
      echo "$TAG GAME_REWARD_MODE=shared_binary but $GAME_LOOP has no '$marker'." >&2
      echo "$TAG The working copy predates the reward patch and would train on the" >&2
      echo "$TAG LEGACY reward. Re-run: SSP_ROOT=\$SSP_ROOT SSP_WORK=\$WORK \\" >&2
      echo "$TAG   \"\$PY\" scripts/slurm/prepare_sources.py" >&2
      exit 1
    fi
  done
fi

if [ "$ATTACKER_INVALID_MOVE" = 1 ]; then
  # Same failure mode as above: a stale work tree would silently train on v6d.
  GAME_LOOP=$WORK/openrlhf/trainer/ppo_utils/language_game.py
  for marker in invalid_move_attacker_reward invalid_move_reward_enabled; do
    if ! grep -q "$marker" "$GAME_LOOP" 2>/dev/null; then
      echo "$TAG ATTACKER_INVALID_MOVE=1 but $GAME_LOOP has no '$marker'; re-run prepare_sources.py and patch_offtopic_labels.py" >&2
      exit 1
    fi
  done
fi

# --- judge --------------------------------------------------------------------
# Each service owns a process group so failed vLLM children release their GPUs.
# Keep one exit trap: installing the Ray cleanup must not forget the verifier.
JUDGE_PID=""
OFFTOPIC_PID=""
TOPIC_JUDGE_PID=""
RAY_STARTED=0
stop_service() {
  local pid=${1:-}
  [ -n "$pid" ] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in {1..25}; do
    kill -0 -- "-$pid" 2>/dev/null || break
    sleep 0.2
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
# The judge runs its own Ray cluster; its vLLM actor leaves the judge's process
# group, so a group kill can leave it holding GPU memory. Kill by temp dir.
kill_marked() {
  local marker=$1 pid
  for pid in $(pgrep -u "$(id -u)" -f "$marker" 2>/dev/null || true); do
    [ "$pid" = "$$" ] && continue
    kill -KILL "$pid" 2>/dev/null || true
  done
}
cleanup() {
  for spid in ${ROLLOUT_SERVER_PIDS:-}; do stop_service "$spid"; done
  stop_service "$TOPIC_JUDGE_PID"
  stop_service "$OFFTOPIC_PID"
  stop_service "$JUDGE_PID"
  [ -n "$JUDGE_PID" ] && kill_marked "${SSP_TMP:-/tmp}/ssp-judge-$JUDGE_PORT"
  if [ "$RAY_STARTED" = 1 ]; then ray_shutdown; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# The judge shares GPU 0: it is idle for most of a step, so a whole card is
# waste. --max_len must stay at the upstream 8192; at 4096 a long transcript
# returns 500 and takes the whole server down mid-run.
if [ -n "${EXTERNAL_JUDGE_URL:-}" ]; then
  echo "$TAG reusing external WildGuard judge $RM_URL"
else
CUDA_VISIBLE_DEVICES="${JUDGE_GPUS:-$JUDGE_GPU}" RAY_TMPDIR=${SSP_TMP:-/tmp}/ssp-judge-$JUDGE_PORT \
setsid "$PY" -m openrlhf.cli.serve_wildguard \
  --model_path "$JUDGE" --port "$JUDGE_PORT" --host "$IP" --bf16 \
  --num_gpus ${JUDGE_REPLICAS:-1} --tensor_parallel_size 1 ${JUDGE_EAGER_FLAG---enforce_eager} --seed 42 \
  --max_len 8192 --max_num_seqs ${JUDGE_MAX_SEQS:-32} --batch_size ${JUDGE_BATCH:-32} \
  --gpu_memory_utilization "$JUDGE_UTIL" > "$OUT/judge.log" 2>&1 &
JUDGE_PID=$!
JUDGE_READY=0
for i in $(seq 1 "${JUDGE_STARTUP_ATTEMPTS:-90}"); do
  kill -0 "$JUDGE_PID" 2>/dev/null || { echo "judge died" >&2; tail -20 "$OUT/judge.log" >&2; exit 1; }
  if curl --noproxy '*' -fsS -m 5 -X POST "$RM_URL" -H 'Content-Type: application/json' \
    -d '{"queries":[{"prompt":"hi","response":"hello","game_idx":0}],"prompts":null}' \
    2>/dev/null | grep -q '"labels"'; then
    JUDGE_READY=1
    break
  fi
  sleep 10
done
if [ "$JUDGE_READY" != 1 ]; then
  echo "judge did not become ready; see $OUT/judge.log" >&2
  tail -20 "$OUT/judge.log" >&2
  exit 1
fi
echo "$TAG judge up after ~$((i * 10))s"
if [ "${JUDGE_REPLICAS:-1}" -gt 1 ]; then
  # the proxy round-robins replicas: one request per replica blocks until each
  # has finished loading, so the next engine on these cards sizes its cache
  # against settled memory
  for r in $(seq 1 "$JUDGE_REPLICAS"); do
    curl --noproxy '*' -fsS -m 900 -X POST "$RM_URL" -H 'Content-Type: application/json' \
      -d '{"queries":[{"prompt":"hi","response":"hello","game_idx":0}],"prompts":null}' >/dev/null 2>&1 || true
  done
  echo "$TAG all $JUDGE_REPLICAS judge replicas warm"
fi
fi  # EXTERNAL_JUDGE_URL

# --- off-topic verifier -------------------------------------------------------
# Shares GPU 0 with the judge: both are idle for most of a step.  It answers one
# question WildGuard structurally cannot -- does the rewrite still ask what the
# seed asked -- which is what stops the attacker collapsing onto a single
# memorized prompt.  Optional: if it fails to come up we train without the term
# rather than lose the run, and the episode rows record the verdict as absent.
export OFFTOPIC_URL=""
if [ "$NEEDS_TOPIC_JUDGE" = 1 ]; then
  # The label is required here -- shared_game_reward.py raises rather than
  # assume "on-topic" -- so unlike the legacy verifier this must come up or the
  # phase fails now, instead of dying thousands of GPU-seconds into step 1.
  case "$TOPIC_JUDGE_BACKEND" in
    local)
      if [ "${TOPIC_JUDGE_MANAGED:-0}" = 1 ]; then
        echo "$TAG reusing managed local topic judge: $TOPIC_API_MODEL"
      else
        CUDA_VISIBLE_DEVICES="$TOPIC_JUDGE_GPU" TOPIC_JUDGE_MODEL="$BASE" TOPIC_JUDGE_PORT="$TOPIC_JUDGE_PORT" \
        TOPIC_JUDGE_UTIL="$TOPIC_JUDGE_UTIL" TOPIC_JUDGE_MAX_LEN="$TOPIC_JUDGE_MAX_LEN" \
        setsid "$PY" "$SSP_REPO/scripts/slurm/serve_topic_judge.py" \
          > "$OUT/topic_judge.log" 2>&1 &
        TOPIC_JUDGE_PID=$!
        TOPIC_JUDGE_READY=0
        for i in $(seq 1 "${TOPIC_JUDGE_STARTUP_ATTEMPTS:-90}"); do
          kill -0 "$TOPIC_JUDGE_PID" 2>/dev/null || { echo "$TAG topic judge died" >&2; tail -20 "$OUT/topic_judge.log" >&2; exit 1; }
          curl --noproxy '*' -fsS -m 5 "http://$IP:$TOPIC_JUDGE_PORT/health" 2>/dev/null | grep -Eq '"ok"[[:space:]]*:[[:space:]]*true' && { TOPIC_JUDGE_READY=1; break; }
          sleep 5
        done
        if [ "$TOPIC_JUDGE_READY" != 1 ]; then
          echo "$TAG topic judge did not become ready; see $OUT/topic_judge.log" >&2
          exit 1
        fi
        echo "$TAG local topic judge up after ~$((i * 5))s on :$TOPIC_JUDGE_PORT"
      fi
      ;;
    anthropic)
      # Nothing to start: the hosted judge needs no local GPU.
      echo "$TAG hosted topic judge: $TOPIC_API_MODEL"
      ;;
  esac
  # Fails fast on a bad URL, model or key, and prints the judge identity that
  # goes into the cache key and the run contract.
  # Per-card memory once the judges are resident and before ray claims
  # anything.  Rebalancing UTIL/OPP_UTIL/JUDGE_UTIL is guesswork without this:
  # vLLM charges each engine for what the whole card already holds.
  echo "$TAG GPU memory after judges (MiB used/total):"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
    | sed "s/^/$TAG   gpu /" || true
  "$PY" -c 'import json;from roll.utils.topic_api import topic_api_contract;print("topic judge contract:",json.dumps(topic_api_contract(),sort_keys=True))'
elif [ "$OFFTOPIC_ENABLE" = 1 ]; then
  CUDA_VISIBLE_DEVICES="$JUDGE_GPU" OFFTOPIC_MODEL="$BASE" OFFTOPIC_PORT="$OFFTOPIC_PORT" \
  OFFTOPIC_UTIL="$OFFTOPIC_UTIL" \
  setsid "$PY" "$SSP_REPO/scripts/slurm/serve_offtopic.py" \
    > "$OUT/offtopic.log" 2>&1 &
  OFFTOPIC_PID=$!
  for i in $(seq 1 "${OFFTOPIC_STARTUP_ATTEMPTS:-60}"); do
    kill -0 "$OFFTOPIC_PID" 2>/dev/null || { echo "$TAG off-topic verifier died; continuing without the penalty" >&2; break; }
    curl --noproxy '*' -fsS -m 5 "http://$IP:$OFFTOPIC_PORT/health" 2>/dev/null | grep -Eq '"ok"[[:space:]]*:[[:space:]]*true' && {
      export OFFTOPIC_URL="http://$IP:$OFFTOPIC_PORT/verify_topic"
      echo "$TAG off-topic verifier up after ~$((i * 5))s"; break; }
    sleep 5
  done
  if [ -z "$OFFTOPIC_URL" ]; then
    echo "$TAG off-topic verifier not ready; continuing without the penalty" >&2
    stop_service "$OFFTOPIC_PID"
    OFFTOPIC_PID=""
  fi
else
  echo "$TAG off-topic penalty disabled (OFFTOPIC_ENABLE=0)"
fi

# --- ray ----------------------------------------------------------------------
# One head per node: two concurrent phases on the same node corrupt each other's
# placement-group scheduling and both hang after "Connected to Ray cluster".
export RAY_TMPDIR=${SSP_TMP:-/tmp}/ssp-ray-$RAY_PORT
# `ray stop` kills every Ray process this user owns on the node, so it cannot
# be used while a second phase runs on the other half of the node. Every Ray
# process carries the temp dir in its command line, so match on that instead.
ray_shutdown() {
  if [ "${RAY_STOP_SCOPE:-temp_dir}" = node ]; then
    "$RAY_BIN" stop --force >/dev/null 2>&1 || true
    return
  fi
  for signal in TERM KILL; do
    for pid in $(pgrep -u "$(id -u)" -f "$RAY_TMPDIR" 2>/dev/null || true); do
      [ "$pid" = "$$" ] && continue
      kill "-$signal" "$pid" 2>/dev/null || true
    done
    if [ "$signal" = TERM ]; then sleep 5; fi
  done
}
# Pin the trainer's engines to the vLLM V0 runner. The trainable group always
# falls back to V0 (sleep and LoRA weight sync are V1-unsupported) and so does
# the frozen-opponent group whenever it carries an adapter -- which is every
# phase except the very first attacker, whose opponent is the bare base model.
# That one phase came up half V0 / half V1, and the V1 engines profile memory
# differently and died with "No available memory for the cache blocks" inside
# the same 0.13 judge + 0.28 trainable + 0.26 opponent budget every other phase
# fits in. This has to be exported before `ray start` for the same reason
# WANDB_RUN_ID does: the actors inherit the raylet's environment, not the
# environment of the command that launches the job. The judge is already up on
# its own V1 engine by this point and is unaffected.
export VLLM_USE_V1=${VLLM_USE_V1:-0}
RAY_GPUS=$GPUS
if [ "${ROLLOUT_SERVER:-0}" = 1 ]; then
  RENV=${ROLLOUT_ENV:-/zfsauton/scratch/wentsec/envs/vllm30}
  RS_PORT=${RS_PORT:-$((RAY_PORT + 1000))}
  ROLLOUT_SERVER_PIDS=""; URLS="${EXTERNAL_ROLLOUT_SERVERS:-}"
  for ((c = 0; c < ROLLOUT_CARDS; c++)); do
    [ -n "${EXTERNAL_ROLLOUT_SERVERS:-}" ] && break
    port=$((RS_PORT + c))
    env -u VLLM_USE_V1 -u VLLM_ENABLE_V1_MULTIPROCESSING -u PYTHONPATH \
      CUDA_VISIBLE_DEVICES=$c VLLM_ALLOW_RUNTIME_LORA_UPDATING=True PATH="$RENV/bin:$PATH" \
      setsid "$RENV/bin/python" -m vllm.entrypoints.openai.api_server --model "$BASE" --served-model-name base --host 127.0.0.1 --port $port \
        --enable-lora --max-loras ${SERVER_MAX_LORAS:-24} --max-cpu-loras 48 --max-lora-rank "$LORA_RANK" \
        --gpu-memory-utilization "$SERVER_UTIL" --max-model-len 4096 --enable-prefix-caching \
        --max-num-seqs ${SERVER_MAX_NUM_SEQS:-512} --seed 0 --dtype bfloat16 \
        > "$OUT/rollout_server_$c.log" 2>&1 &
    ROLLOUT_SERVER_PIDS="$ROLLOUT_SERVER_PIDS $!"
    URLS="${URLS:+$URLS,}http://127.0.0.1:$port"
  done
  for url in ${URLS//,/ }; do
    for i in $(seq 1 180); do
      curl --noproxy '*' -fsS -m 3 "$url/health" >/dev/null 2>&1 && break
      sleep 5
    done
    curl --noproxy '*' -fsS -m 3 "$url/health" >/dev/null || { echo "$TAG rollout server $url never came up" >&2; tail -30 "$OUT"/rollout_server_*.log >&2; exit 1; }
  done
  # frozen opponents: opp_<i> in pool order (a base-model entry needs no adapter)
  "$PY" - "$URLS" <<'PYEOF'
import json, os, sys, urllib.request
urls = sys.argv[1].split(",")
# v9sb: ONEJOB_SHARED_BASE=1 -> load concat(pi0 LoRA, adapter) (roll.utils.shared_base_lora)
if os.environ.get("ONEJOB_SHARED_BASE") == "1":
    sys.path.insert(0, os.environ["SHARED_BASE_SSP_REPO"])
    from roll.utils.shared_base_lora import merged_path
else:
    merged_path = lambda p: p
raw = os.environ.get("OPPONENT_POOL_JSON", "").strip()
single = os.environ.get("OPPONENT", "").strip()
pool = json.loads(raw) if raw else ([{"adapter": single}] if single else [])
for i, entry in enumerate(pool):
    if not entry.get("adapter"):
        continue
    for url in urls:
        req = urllib.request.Request(f"{url}/v1/load_lora_adapter", method="POST",
            data=json.dumps({"lora_name": "cell_" + __import__("hashlib").sha1(os.path.abspath(entry["adapter"]).encode()).hexdigest()[:16], "lora_path": merged_path(entry["adapter"])}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=600).read()
        except urllib.error.HTTPError as exc:  # already loaded by a cell or an earlier oracle
            if "already" not in exc.read().decode(): raise
print(f"loaded {sum(1 for e in pool if e.get('adapter'))} opponent adapters on {len(urls)} servers")
PYEOF
  export ROLLOUT_SERVERS=$URLS ROLLOUT_BASE_NAME=base ROLLOUT_LORA_DIR=/dev/shm/ssp_rollout_lora_$RAY_PORT
  # Ray (trainer only) sees just the trainer cards
  export CUDA_VISIBLE_DEVICES=${TRAIN_CARDS:-$(seq -s, $ROLLOUT_CARDS $((GPUS - 1)))}
  RAY_GPUS=$TRAIN_GPUS
  echo "$TAG rollout servers $URLS; trainer cards $CUDA_VISIBLE_DEVICES"
fi
ray_shutdown; rm -rf "$RAY_TMPDIR"
RAY_STARTED=1
"$RAY_BIN" start --head --disable-usage-stats --temp-dir="$RAY_TMPDIR" \
  --port="$RAY_PORT" --include-dashboard=false \
  --ray-client-server-port=$((RAY_PORT + 2000)) --num-gpus="$RAY_GPUS" \
  --node-manager-port=$((RAY_PORT + 2100)) --object-manager-port=$((RAY_PORT + 2200)) \
  --metrics-export-port=$((RAY_PORT + 2300)) --dashboard-agent-listen-port=$((RAY_PORT + 2400)) \
  --dashboard-agent-grpc-port=$((RAY_PORT + 2500)) --runtime-env-agent-port=$((RAY_PORT + 2600)) \
  --object-store-memory=${RAY_OBJECT_STORE_BYTES:-21474836480} ${RAY_MIN_WORKER_PORT:+--min-worker-port=$RAY_MIN_WORKER_PORT --max-worker-port=$RAY_MAX_WORKER_PORT} >/dev/null
export RAY_ADDRESS=$IP:$RAY_PORT

echo "$TAG start $(date +%H:%M:%S) role=$ROLE lr=$LR steps=$STEPS opponent=${OPPONENT:-${OPPONENT_POOL_JSON:-none}}" >> "$PSRO/state.txt"
"${TRAIN_CMD[@]}"
echo "$TAG done rc=$? $(date +%H:%M:%S)" >> "$PSRO/state.txt"

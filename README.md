# SafeSelfPlay

SafeSelfPlay trains separate attacker and defender LoRA policies using
[Self-RedTeam](https://github.com/mickelliu/selfplay-redteaming), with
latest-opponent self-play or PSRO opponent mixtures.

This README covers setup, API credentials, launching scripts and troubleshooting.
Benchmark tables, protocols, checkpoint locations and historical experiments
are in [EXPERIMENTS.md](EXPERIMENTS.md).

## Docker teacher handoff

**从 GitHub 克隆或收到源码小包后：先看 [docker/QUICKSTART.source.llama8b.md](docker/QUICKSTART.source.llama8b.md)。**
`build` 自动下载运行依赖和上游数据，`prepare` 下载模型；包内保留本地 / Claude label 与三组训练命令。

可从 [GitHub Releases](https://github.com/xudongwu-0/SafeSelfPlay/releases/latest) 下载约 1.75 MB 的
`safeselfplay-llama8b-h200-source.build.tar.gz`。它包含源码和构建脚本，不包含镜像、模型或 API key；
老师在本地构建环境，并按 Quick Start 配置 W&B。

源码小包内的入口说明是 `START_HERE.md`，对应 [源码 Quick Start](docker/QUICKSTART.source.llama8b.md)。
顺序是：构建镜像 → 选择本地 / Claude label
→ 下载模型并检查环境 → 三个训练命令 → 早停与续跑。
完整容器说明见 [docker/README.llama8b.md](docker/README.llama8b.md)。
Docker 版设置 `TOPIC_JUDGE_BACKEND=local` 即可使用本地冻结模型，不需要 Claude key；
其服务贯穿训练和矩阵计算，启动入口自动处理 API 配置与 GPU 分配。
本文下方的裸 Slurm 脚本说明与 Docker 入口有所区别。

## Shared training / PSRO reward and topic judge

`GAME_REWARD_MODE=shared_binary` makes training and the PSRO matrix share one
terminal reward, with an off-topic deduction supplied by a judge that sees the
seed. Design, the payoff derivation and the measured local-vs-Opus comparison:
[TOPIC_JUDGE_AND_SHARED_REWARD.md](TOPIC_JUDGE_AND_SHARED_REWARD.md).

Default configuration (local judge, no credential required):

```bash
export GAME_REWARD_MODE=shared_binary
export TOPIC_PENALTY=2
export TOPIC_JUDGE_BACKEND=local     # default; TOPIC_API_* is wired automatically
```

Hosted judge instead of the local one:

```bash
export TOPIC_JUDGE_BACKEND=anthropic
export TOPIC_API_MODEL=claude-opus-5
read -rsp 'Anthropic API key: ' TOPIC_API_KEY; echo
export TOPIC_API_KEY                 # or TOPIC_API_KEY_FILE=/absolute/path
```

Check the configuration without starting training:

```bash
"$SSP_PYTHON" -m roll.utils.topic_api --check
```

## Entrypoints

Run commands from this repository's root.

| Task | Script |
| --- | --- |
| Slurm PSRO | `scripts/slurm/submit_cold_psro.sh` |
| Local PSRO on an allocated GPU node | `scripts/slurm/cold_psro_local.py` |
| Local uniform-history / naive self-play | `scripts/slurm/history_selfplay.py` |
| One local role phase | `scripts/slurm/role_phase.sh` |
| Slurm latest-opponent ladder | `scripts/slurm/submit_ladder.sh A1 D1 A2 D2` |
| Slurm defender evaluation | `scripts/slurm/submit_eval.sh` |
| Report saved checkpoint/mixture metrics | `scripts/slurm/eval_policy_report.py` |
| Modal Self-RedTeam reproduction | `launch_selfredteam_official_h200.py` |
| Modal A1/D1 training | `run_selfredteam_lora_a1d1_h200x4.sh` |
| Modal latest-opponent continuation | `run_selfredteam_lora_next_round_h200x4.sh` |
| Modal cold/warm PSRO | `run_role_lora_cold_psro5_h200x4.sh` |
| Modal defender evaluation | `run_selfredteam_eval.sh` |

## Source and API credentials

```bash
git clone git@github.com:xudongwu-0/SafeSelfPlay.git
cd SafeSelfPlay
./setup_selfredteam_upstream.sh

export HF_TOKEN='<your Hugging Face token>'
# Optional for Slurm; omit to disable W&B.
export WANDB_API_KEY='<your W&B key>'
export WANDB_ENTITY='<your W&B user or team>'
export WANDB_PROJECT='self-play'
```

The helper prepares `../selfplay-redteaming` at revision
`0c56e503e8ae1b1b0fcd2214c92ea31fef1cb123`; the patches depend on its source
layout. Weights, caches and experiment outputs are not included in Git.

`HF_TOKEN` needs access to gated `allenai/wildguard` and `allenai/wildguardmix`,
plus any other gated evaluation models. Slurm jobs inherit exported variables.
The original Modal reproduction requires both HF and W&B credentials in the
`roll-secrets` secret described below.

Harmfulness/refusal reward labels use a local/deployed WildGuard HTTP service.
The shared binary reward additionally requires the external topic API configured above.
Slurm constructs `/classify` from `JUDGE_PORT` or
`TRAIN_JUDGE_PORT`. It sets `OFFTOPIC_URL` to the optional verifier's
`/verify_topic` endpoint when ready. The external API and model IDs used in the
historical attacker-transfer experiment are recorded in
[EXPERIMENTS.md](EXPERIMENTS.md#attacker-results); its transfer launcher is not
included in this checkout.

## Slurm setup

Use an environment with the upstream requirements. The validated stack is
Python 3.10, torch 2.6.0, vLLM 0.8.2, transformers 4.50.0, PEFT 0.15.2,
DeepSpeed 0.16.5, Ray 2.44.0, numpy 1.26.4 and `setuptools<81`.
See [Slurm runtime notes](scripts/slurm/README.md) for memory settings.

Override the original cluster's paths explicitly:

```bash
export SSP_ROOT="/scratch/$USER/safeselfplay"
export WORKSPACE="$SSP_ROOT"
export SSP_PYTHON='/absolute/path/to/training-env/bin/python'
export SSP_UPSTREAM_SRC="$PWD/../selfplay-redteaming"
export SSP_UPSTREAM="$SSP_UPSTREAM_SRC"
export SSP_WORK="$SSP_ROOT/work"
export SSP_HF_HOME="$SSP_ROOT/hf"
export SSP_SFT_DIR="$SSP_ROOT/sft_data"
export PSRO_ROOT="$SSP_ROOT/psro/role_lora_zero_sum_psro"
export RAW_PAYOFF_ROOT="$SSP_ROOT/psro/raw_payoff_v2"
export HF_HOME="$SSP_HF_HOME"
unset TRANSFORMERS_CACHE

"$SSP_PYTHON" scripts/slurm/prepare_sources.py
"$SSP_PYTHON" scripts/slurm/prepare_sft_data.py
"$SSP_PYTHON" scripts/slurm/patch_defender_cot.py
"$SSP_PYTHON" scripts/slurm/patch_offtopic_labels.py

# Download on a machine with network access.
"$SSP_PYTHON" - <<'PY'
from huggingface_hub import snapshot_download
for model in (
    'mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated',
    'allenai/wildguard',
):
    print(snapshot_download(model))
PY
```

Launchers discover model snapshots under `$SSP_HF_HOME/hub`. For explicit
snapshots, set `BASE_MODEL_PATH` / `JUDGE_MODEL_PATH` for training, and also
set `COLD_PSRO_BASE_MODEL_DIR` to the same base snapshot for PSRO payoff workers.
Reapply both patches after rebuilding `$SSP_WORK`; `--check` verifies them.

## Training commands

Run on an allocated 4×H200 node after [Slurm setup](#slurm-setup). These run
in the foreground and submit no jobs. Export the paths and the reward mode
once, then pick a workflow:

```bash
export BASE_MODEL_PATH='/absolute/path/to/llama-8b-abliterated/snapshot'
export JUDGE_MODEL_PATH='/absolute/path/to/wildguard/snapshot'
export COLD_PSRO_BASE_MODEL_DIR="$BASE_MODEL_PATH"
export GAME_REWARD_MODE=shared_binary TOPIC_PENALTY=2
export INIT_KL_COEF=0 PSRO_MIDRUN_RESUME=0 OFFTOPIC_ENABLE=0
export LADDER_EARLY_STOP=1 MICRO_ROLLOUT=8
export STEPS_PER_ROLE=300 SAVE_STEPS_CONTRACT=10 MATRIX_EPISODES=4000
"$SSP_PYTHON" -m roll.utils.topic_api --check
```

Each workflow uses all four GPUs; run them sequentially.

### Cold PSRO — A1/D1 through A8/D8, 300 steps per cell

```bash
"$SSP_PYTHON" scripts/slurm/cold_psro_local.py \
  --workspace "$SSP_ROOT" --work "$SSP_WORK" --hf-home "$SSP_HF_HOME" \
  --sft-dir "$SSP_SFT_DIR" --psro-root "$PSRO_ROOT" \
  --raw-payoff-root "$RAW_PAYOFF_ROOT" --python "$SSP_PYTHON" \
  --judge-model "$JUDGE_MODEL_PATH" --gpus 4 --cell-workers 3 \
  --fresh-cold-run --suffix psro8_run1 --generations 8
```

Each round trains the attacker then the defender, both cold-started from base,
against an opponent drawn from the current Nash mixture, then fills the payoff
matrix. Output: `$PSRO_ROOT/<suffix>`.

### Uniform history baseline — 8 rounds, 300 steps

```bash
"$SSP_PYTHON" scripts/slurm/history_selfplay.py \
  --method uniform --run-suffix uniform8_run1 --rounds 8 --steps 300
```

Same cold start as PSRO, but the opponent is sampled **uniformly over every
available opposite-role policy, base included**: with `D0,D1,D2` in the pool
each is played with probability `1/3`. Output:
`$SSP_ROOT/selfplay_baselines/uniform/<run-suffix>`.

### Naive self-play — 6 rounds, 100 steps

```bash
"$SSP_PYTHON" scripts/slurm/history_selfplay.py \
  --method naive --run-suffix naive6_run1 --rounds 6 --steps 100
```

Warm-starts each role from the previous same-role adapter and faces only the
newest opponent. Requires `INIT_KL_COEF=0`. Output:
`$SSP_ROOT/selfplay_baselines/naive/<run-suffix>`.

### Single role, single round

```bash
ROLE=attacker GEN=1 STEPS=100 bash scripts/slurm/role_phase.sh
```

Trains one role for one generation and stops; no matrix, no progression. Use
`ROLE=defender` for the other side, and `DRY_RUN=1` to assemble and print the
command without training. To reproduce a specific oracle from an existing run
with its recorded opponent mixture and hyperparameters, use
`scripts/slurm/run_single_oracle.py` instead.

### Smoke test

```bash
SAVE_STEPS_CONTRACT=1 "$SSP_PYTHON" scripts/slurm/history_selfplay.py \
  --method uniform --run-suffix smoke1 --rounds 1 --steps 2
```

Runs real training and real judge calls; the configuration check alone does not
validate credentials or model availability.

Step counts are maximum budgets — early stopping stays enabled. The rules and
the metrics they use are in
[TOPIC_JUDGE_AND_SHARED_REWARD.md](TOPIC_JUDGE_AND_SHARED_REWARD.md#early-stopping).

## Launch on Slurm

Set scheduler values for your cluster. Otherwise the submit scripts use the
original Auton `gpu2` / `project` / `qos_project_preempt` configuration.

```bash
export PARTITION='<your partition>'
export QOS='<your QoS>'
export NODELIST='<your GPU node>'
export GRES='gpu:h200:4'
export SUFFIX="cold_psro_$(date +%Y%m%d_%H%M%S)"

# Fresh A0/D0 population; no archive or seed script required.
FRESH_COLD_RUN=1 GENERATIONS=8 STEPS_PER_ROLE=300 MATRIX_EPISODES=4000 \
  bash scripts/slurm/submit_cold_psro.sh
```

Follow `$PSRO_ROOT/$SUFFIX/logs/driver_<job-id>.log` and `state.json`.
Resubmit the same suffix to recover a stopped run; completed roles/cells are
reused. With the default base-KL penalty, incomplete roles restart from base.

To continue an archived run, first supply all required seed arguments:

```bash
"$SSP_PYTHON" scripts/slurm/seed_cold_psro_run.py \
  --seed-dir /absolute/path/to/archived/run \
  --psro-root "$PSRO_ROOT" --suffix "$SUFFIX"
GENERATIONS=5 bash scripts/slurm/submit_cold_psro.sh
```

For one role or a latest-opponent ladder:

```bash
DRY_RUN=1 ROLE=attacker GEN=1 bash scripts/slurm/role_phase.sh
ROLE=attacker GEN=1 bash scripts/slurm/role_phase.sh
bash scripts/slurm/submit_ladder.sh A1 D1 A2 D2
```

`DRY_RUN=1` prints the command and redacts the W&B key without starting
services. Prepared source, SFT files and model paths are still required.

| Variable | Default | Purpose |
| --- | --- | --- |
| `INIT_KL_COEF` | `0.01` | Base-model reverse-KL penalty for both roles; `0` disables it. |
| `PSRO_MIDRUN_RESUME` | `0` with KL; `1` without | Continue a partial role's adapter; requires KL disabled and does not restore optimizer state. |
| `TOPIC_JUDGE_BACKEND` | `local` | `local` runs the base model on its own card; `anthropic` calls the hosted judge. Only used with `GAME_REWARD_MODE=shared_binary`. |
| `TOPIC_API_MODEL` | `claude-opus-5` / `local-topic-judge` | Judge model; part of the cache key and the run contract. |
| `JUDGE_GPU` | `0` | Card WildGuard runs on. |
| `TOPIC_JUDGE_GPU` | `(JUDGE_GPU + 1) % GPUS` | Card the local topic judge runs on. Keep it off WildGuard's card: a second engine on an occupied card must be given a fraction covering the first one's usage too, and `0.13 + 0.13` then starves it. A 1-GPU job falls back to sharing. |
| `TOPIC_JUDGE_UTIL` | `0.13` | Local topic judge's fraction of its card. Measured footprint is ~17 GiB of 143771 MiB (0.118) on an H200. |
| `TOPIC_JUDGE_MAX_LEN` | `4096` | Judge context. The prompt is a contract plus one seed and one rewrite and the verdict is capped at 24 tokens, so 8192 only cost ~4 GiB of KV. |
| `TOPIC_JUDGE_PORT` | `JUDGE_PORT + 300` | Local topic judge port. |
| `TOPIC_JUDGE_STARTUP_ATTEMPTS` | `90` | Health probes before failing the role; the label is required, so this does not fall back. |
| `OFFTOPIC_ENABLE` | `0` | Start the superseded legacy verifier. Leave at `0`: it does not feed the shared reward and its 0.24 of GPU 0 overcommits the card. |
| `TOPIC_PENALTY` | `2` | Deduction for an off-topic verdict, on a payoff otherwise in `[-1, 1]`. At 2 an off-topic win ties the worst on-topic outcome; strict dominance needs `> 2`. |
| `OFFTOPIC_PENALTY` | `0.5` | Penalty consumed by zero-sum reward projection. |
| `OFFTOPIC_UTIL` | `0.24` | Additional verifier fraction of GPU 0, shared with the judge and training. |
| `OFFTOPIC_TIMEOUT_SECONDS` | `15` | Verifier read timeout; connect timeout is 3 seconds, with no retries. |
| `JUDGE_STARTUP_ATTEMPTS` / `OFFTOPIC_STARTUP_ATTEMPTS` | `90` / `60` | Health probes before failing the role / disabling the verifier. |
| `RACE_CELLS` | `0` | `1` enables pilot cells and selective budget extension. |
| `DRIVER_MAX_ATTEMPTS` | `24` | Driver retries, 60 seconds apart; use `1` when diagnosing configuration errors. |

The `OFFTOPIC_*` settings in this table describe the legacy local verifier.
In legacy mode a missing verdict receives no penalty; its label path does not
complete the training reward integration. For the implemented common training /
matrix reward, use `GAME_REWARD_MODE=shared_binary` and the `TOPIC_API_*` settings
above. Its required verdicts never fall back to unpenalized samples.

## Evaluation

Use a separate evaluation environment; `submit_eval.sh` documents its extra
dependencies. Keep the Slurm path and scheduler variables above exported.

```bash
export SSP_EVAL_PYTHON='/absolute/path/to/evaluation-env/bin/python'
export EVAL_OUTPUT_ROOT="$SSP_ROOT/psro/upstream_selfredteam_role_full_eval"
DEFENDERS=D1,D2 LABEL_PREFIX=my_run bash scripts/slurm/submit_eval.sh
```

This script calls `sbatch` itself. It covers WildGuard, WildJailbreak, DAN,
HarmBench, OR-Bench and XSTest. Historical SR12/SORRY-Bench launchers under
`hard_bench/*.sbatch` are not included in Git.

To report saved metrics or list mixture components needing evaluation:

```bash
export STD_EVAL="$EVAL_OUTPUT_ROOT"
export HARD_BENCH='/absolute/path/to/hard_bench'
"$SSP_EVAL_PYTHON" scripts/slurm/eval_policy_report.py \
  --suffix "$SUFFIX" --arm-prefix my_run --policy D1
"$SSP_EVAL_PYTHON" scripts/slurm/eval_policy_report.py \
  --suffix "$SUFFIX" --arm-prefix my_run --policy nash:defender --plan
# nash:defender@5 selects generation 5; omit --plan to combine saved metrics.
```

This report does not launch missing evaluations. See
[mixture interpretation](EXPERIMENTS.md#recorded-nash-mixture-example).
For a standalone local checkpoint, activate the evaluation environment first:

```bash
SHARED_ENV=/dev/null GPU_ID=0 \
CHECKPOINT_PATH=/absolute/path/to/adapter OUTPUT_DIR=/absolute/path/to/results \
  bash scripts/run_local_defender_eval.sh
```

The local helper requires `HF_HOME`, HF authentication and the pinned evaluation
fork, and resumes completed metrics.

## Modal setup and launch

Modal builds the GPU runtime remotely; no local GPU is required.

```bash
python3 -m pip install modal requests
modal setup
modal secret create roll-secrets HF_TOKEN="$HF_TOKEN" WANDB_API_KEY="$WANDB_API_KEY"
modal deploy modal_selfredteam_wildguard.py

# Original reproduction.
modal deploy modal_selfredteam_official_h200.py
python3 launch_selfredteam_official_h200.py

# Separate jobs: choose the workflow you need.
./run_selfredteam_lora_a1d1_h200x4.sh
SOURCE_RUN_SUFFIX='<completed-A1-D1-suffix>' ./run_selfredteam_lora_next_round_h200x4.sh
./run_role_lora_cold_psro5_h200x4.sh
COLD_START=false ./run_role_lora_cold_psro5_h200x4.sh
```

Modal PSRO defaults are five generations, 100 steps per role, rank/alpha 64,
attacker/defender LR `1e-5`/`4e-5`, and 4,000 retained episodes per cell.
Slurm-specific switches above do not configure Modal training.

```bash
# Extend a completed run; other run settings must match.
RUN_SUFFIX='<completed-suffix>' GENERATIONS=6 ./run_role_lora_cold_psro5_h200x4.sh

modal deploy modal_selfredteam_official_eval.py
CHECKPOINT_PATH=/output/.../D3/ckpt/global_step100_hf \
OUTPUT_SLUG=D3_full_eval TRAINED_LABEL=our_D3 ./run_selfredteam_eval.sh

# Recover only a failed defender phase.
ATTACKER_ADAPTER=/output/.../A3/ckpt/global_step100_hf \
DEFENDER_START_ADAPTER=/output/.../D2/ckpt/global_step100_hf \
TARGET_GENERATION=3 ./run_selfredteam_lora_defender_only_h200x4.sh
```

State, checkpoint inventory and `matrices/final_gN.json` live under
`/output/role_lora_zero_sum_psro/<RUN_SUFFIX>/`. Volumes belong to the selected
Modal workspace; a new workspace needs its checkpoint inputs copied or
downloaded. Set `MODAL_BIN` if the executable is not on `PATH`.

## Startup and resume checks

- **Source hashes:** reward hashes are no longer required. Other implementation
  hashes are provenance only and do not block PSRO startup, resume or extension.
  Old states containing reward hashes work without manual edits. Parameter,
  reward-version, checkpoint and dataset/cache checks still apply. Use a new
  suffix when changing scoring semantics so old and new results are separate.
- **Judge not ready:** role training now exits after unsuccessful health probes,
  prints the judge log tail and cleans up service processes. Check gated model
  access, ports and memory. The payoff-cell judge still has a separate loop of
  180 probes, each with up to 10 seconds for the request and 10 seconds sleep.
- **Slow required reward requests:** WildGuard still uses up to 12 attempts,
  each with a 180-second request timeout plus backoff: roughly 38 minutes
  before failure. Check `judge.log`. Optional off-topic requests use the short
  timeout above.
- **GPU memory / idle Ray:** judge, topic judge, opponent, trainable engines and
  optimizer share GPU 0. `No available memory for the cache blocks` at startup
  means the fractions exceed 1.0: the usual cause is `OFFTOPIC_ENABLE=1` adding
  the superseded legacy verifier on top of the local topic judge. Use
  `TOPIC_JUDGE_BACKEND=anthropic` to take the topic judge off the card entirely. Avoid sharing GPUs, ports or Ray directories between
  phases; distinct ports alone do not isolate GPU allocations.
- **Credentials and offline nodes:** Slurm works without W&B. Pre-download gated
  models and evaluation data before using an offline compute node; an HF token
  alone cannot provide network access.
- **KL and continuation:** base-KL defaults to restarting incomplete roles from
  base. Adapter continuation/hot start requires `INIT_KL_COEF=0` in the current
  local driver. Conflicts fail before moving an existing checkpoint.
- **Paths and retry loops:** override cluster-specific paths, check disk quota,
  and use `DRIVER_MAX_ATTEMPTS=1` while diagnosing deterministic failures.
  Do not edit a shell script while a running job is reading it.

## Load published checkpoints

See [EXPERIMENTS.md](EXPERIMENTS.md) for artifact identities and measured results.
The reproduction is a full model; role policies are adapters on the base model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

repo = "xudongwu/SafeSelfPlay-checkpoints"
model = AutoModelForCausalLM.from_pretrained(
    repo, subfolder="self-redteam-reproduction/step200"
)
tokenizer = AutoTokenizer.from_pretrained(
    repo, subfolder="self-redteam-reproduction/step200"
)

# Alternatively, load a published role adapter.
base = AutoModelForCausalLM.from_pretrained(
    "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
)
model = PeftModel.from_pretrained(base, repo, subfolder="lora/D3")
```

## Archive upload

```bash
hf auth login
LOCAL_RUN_ROOT=/absolute/path/to/archived/run ./scripts/upload_cold_psro_run.sh
```

The uploader checks archive `SHA256SUMS` and checkpoint identity. These are
artifact integrity checks, separate from training source hashes. Published
artifacts are indexed in [EXPERIMENTS.md](EXPERIMENTS.md).

## Attribution

This repository is a research fork of [ROLL](https://github.com/alibaba/ROLL).
The language game, upstream optimizer and general-sum reward follow
[Self-RedTeam](https://github.com/mickelliu/selfplay-redteaming). Separate role
adapters, checkpoint lineage and PSRO integration are SafeSelfPlay additions.

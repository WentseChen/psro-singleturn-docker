# 全部实验运行说明（invalid-move reward，镜像 v6，4×H200）

> **2026-09-25 升级到镜像 v6 的操作请只看 `docker/V6_UPGRADE_20260925.md`**（独立、完整）。本文件其余部分是历史说明。

> **给执行本文件的 agent**：先完整读完本文件再动手。按「执行清单」的顺序做，每一步的检查都要做并记录输出。
> 遇到「停下并报告」的情况，**不要自行变通**，把原样的命令输出发给 Wentse（陈文泽）等回复。

## 执行清单（按顺序）

### 硬性规则（任何时候都不能违反）

1. **不要删除、移动、重命名任何 run 目录**（FSx 上的 `.../runs/...` 和 `SSP_DOCKER_WORKSPACE`）。checkpoint 只存在那里。
2. **续跑的 run 不能改 suffix、seed、`--rounds`、`--steps-per-role` 或任何训练环境变量**；只允许换镜像和加三个 `CELL_*` 变量。
3. **只用镜像 v6**（`…-invalidmove-20260925-v6`，digest 见文末）。`…-20260923`、`-v2`、`-v3`、`-v4`、`-v5` 一律不用。
4. **一个 4×H200 节点只跑一个 run**。启动前 `nvidia-smi` 必须显示 4 张卡上没有进程。
5. **不要在一个角色训练中途停 run**（除非本文件明确要求）：中途停会让该角色从 step 0 重训。要停就等该角色 final checkpoint 上传后。
6. 每个 run 启动时都必须带上：`export CELL_STREAMING=1 CELL_ENFORCE_EAGER=0 CELL_SHARE_JUDGE_GPUS=0`。
7. 不要上传基座权重、`wandb/`、`topic_labels.sqlite3` 或任何 key；不要把 token 写进文件或命令历史。
8. **同一个 seed 同一时间只跑一个 run，不要 fork。** W&B run ID 由 suffix 和角色名决定：两个进程用同一个 suffix 训练同一个角色，
   会写进**同一个 W&B run**，曲线混在一起无法区分（2026-09-24 s8889 的 streaming fork 就把 D2 的 W&B 记录从 step 100 起覆盖了）；
   若共用同一个 run 目录还会互相覆盖 `state.json` 和 checkpoint。确实需要第二个分支时，必须用**新的 suffix 和新的 run 目录**。

### 要做的 4 个 run

| run | 现状 | 要做的事 | suffix |
| --- | --- | --- | --- |
| PSRO s8888 | 已停（D1 在 step 120 因格式塌缩被保护机制停掉，容器已退出） | **v6 续跑**（任务 B） | `psro8_local_inv1-45906ffc-s8888-g8`（原样） |
| PSRO s8889 | 在跑（A2） | **等 A2 结束后换 v6 续跑**（任务 C） | `psro8_local_inv1-45906ffc-s8889-g8`（原样） |
| uniform（FSP） | 旧 run 用旧 reward（`shared_binary`），不可比 | **停掉旧的，v6 新 suffix 从头跑**（任务 D） | `uniform8_local_inv4_s8888` |
| naive | 旧 run 用旧 reward，已停 | **v6 新 suffix 从头跑**（任务 D） | `naive6_local_inv4_s8888` |

旧 naive / uniform run 的 checkpoint 与目录保留不动（规则 1），只是不再继续训练。

**启动方式**：本文件的命令用 `docker/run_llama8b.sh`。如果你们用 Kubernetes / 自己的脚本启动（上一轮的 run 目录在
`/shared/dedicated-fsx-…/home/zzhihan/selfplay/runs/…`，就是这种方式），就用**上一轮完全相同的启动方式**，只改两处：
镜像换成 v6，环境变量里加三个 `CELL_*`。容器内的入口命令等价于
`psro --run-suffix <suffix> --rounds 8 --steps-per-role 300`（uniform：`uniform … --rounds 8 --steps-per-role 300`；
naive：`naive … --rounds 6 --steps-per-role 100`），FSx 目录挂在与上一轮相同的路径。

### 任务 A：每个节点的准备（第 1–3 节）

1. 按第 1 节拉取 v6 镜像；按第 2 节设置环境变量（包括三个 `CELL_*`）。
2. `df -h <run 目录所在盘>`：可用空间 ≥ 150 GB，否则停下并报告。
3. 按第 3 节 `prepare`、`check`：`check` 显示的 `git_sha` 必须是文末「镜像信息」里的构建 commit，否则停下并报告。
4. 做 HF 写权限预检（第 3 节）。401/403 → 停下并报告（需要换 token）。

### 任务 B：PSRO s8888 续跑（现在就做）

按第 4 节末「已有的 PSRO run」逐条执行：同一个 FSx run 目录挂在同一路径、同一个 suffix、同样参数、v6 镜像 + `CELL_*`。
- 启动前：`state.json` 和 `training/A1/ckpt/global_step240_hf/adapter_model.bin` 都存在，否则停下并报告。
- 启动后 10 分钟内检查 driver 日志：
  - 有 `no state at … initializing a fresh cold run` → **立刻停掉**并报告（没找到旧 run，会从头跑）。
  - 有 `Training settings changed` 或 `Missing adapter checkpoint` → 停下并报告。
  - 应看到 `next action: {'kind': 'oracle', 'role': 'defender', 'target': 'D1'}`，随后出现新的 W&B run `…s8888-g8__D1`。
- D1 训练中每 30 分钟看 W&B `defender/cot_format_violation`：**连续 20 步 > 0.3 → 停掉 run 并报告**（上次就是这样塌缩的）。

### 任务 C：PSRO s8889 换 v6（等时机）

1. 不要打断 A2。等 HF 上出现 `psro8_local_inv1-45906ffc-s8889-g8 A2 step<N> (final)`，并且 driver 日志显示已进入收益格（`prefetching … payoff cell`）。
2. 然后 `docker stop -t 180 <s8889 容器>`（或删除对应 pod），`nvidia-smi` 确认 4 卡空闲。
3. 与任务 B 相同的方式、相同的检查续跑（suffix 换成 s8889 的）。启动后应看到 `next action` 为收益格（`kind: cell`）或下一个角色 D2，**不能**是 fresh run。

### 任务 D：uniform 与 naive 从头跑

1. 停掉仍在跑的旧 uniform 训练（`uniform-b`，旧 reward）。旧目录保留。
2. 在各自的空闲节点上按第 4 节的 uniform / naive 命令启动（新 suffix，带 `CELL_*`）。
3. 启动日志第一行应为 `Starting uniform: …; reward mode=legacy` / `Starting naive: …; reward mode=legacy`。
   如果显示 `shared_binary`，停掉并报告。
4. 节点不够时顺序：PSRO s8888 > PSRO s8889 > uniform > naive。

### 任务 E：运行期间（每 30–60 分钟一次）

按第 5 节的表检查每个 run。另外：
- 每个角色结束后确认 HF 上出现该角色的 `(final)` checkpoint（含 `adapter_model.*`）；没有就按第 6 节手动上传。
- W&B run 显示 `crashed` 不一定是故障（早停也显示 `crashed`）：看 `training/<角色>/early_stop.json`，
  `returncode=0` 正常；`returncode=2`（COLLAPSE）或其他非 0 → 看 `training.log` 末尾 100 行并报告。
- 某个 run 超过 2 小时既没有新的 W&B step 也没有新的 HF 上传 → 查容器/pod 是否还在（`docker ps -a`），把日志末尾报告。
- 容器自动重试（最多 5 次）后仍失败 → 停下并报告，不要手动反复重启。

### 需要报告给 Wentse 的内容

- 每个 run 启动后：命令（去掉 token）、driver 日志前 30 行、`nvidia-smi`。
- 每天一次：每个 run 当前角色和 step、已完成的角色列表、最近一次 HF 上传。
- 任何「停下并报告」的情况：原样输出。

## 0. 节点要求

- Linux x86_64，单节点 4×H200，已装 NVIDIA 驱动、Docker、NVIDIA Container Toolkit（见 `QUICKSTART.source.llama8b.md` 第 0 节）。
- 能访问 GitHub、ghcr.io、Hugging Face、W&B。
- 工作目录（`SSP_DOCKER_WORKSPACE`）所在磁盘每个 seed 至少留 **150 GB**（模型约 35 GB；一个完整 8 代 run 约 65 GB checkpoint 与日志）。
- GitHub 账号需能访问私有仓库 `xudongwu-0/SafeSelfPlay`；ghcr 镜像为私有时需被授权。
- **每个 run 占一个 4 卡节点**（PSRO s8888、PSRO s8889、uniform、naive 共 4 个 run）。节点不够就排队依次跑，
  不要在同一节点同时跑两个 run（会争 GPU 和 Ray 端口）。

## 1. 代码与镜像（每个节点都做一次）

```bash
git clone -b wentse/v6d-invalid-move-reward https://github.com/xudongwu-0/SafeSelfPlay.git
cd SafeSelfPlay

docker login ghcr.io          # 镜像为私有时需要；token 需 read:packages
docker pull ghcr.io/wentsechen/safeselfplay:llama8b-h200-invalidmove-20260925-v6
docker tag  ghcr.io/wentsechen/safeselfplay:llama8b-h200-invalidmove-20260925-v6 \
            safeselfplay:llama8b-h200-invalidmove-20260925-v6
export SSP_IMAGE=safeselfplay:llama8b-h200-invalidmove-20260925-v6
```

## 2. 环境变量

```bash
export SSP_DOCKER_WORKSPACE="$HOME/safeselfplay-data"   # 模型、checkpoint、日志都在这里
export GPU_DEVICES=0,1,2,3
export TOPIC_JUDGE_BACKEND=local RUN_LABELER=local      # 本地 topic judge，不需要 API key
# 不要设置 GAME_REWARD_MODE / ATTACKER_INVALID_MOVE：镜像默认就是新 reward（legacy + 1）。

read -rsp 'HF token: ' HF_TOKEN; echo; export HF_TOKEN              # 需有 allenai/wildguard 权限
read -rsp 'W&B key: '  WANDB_API_KEY; echo; export WANDB_API_KEY
export WANDB_ENTITY=2373025856w-the-university-of-hong-kong WANDB_PROJECT=self-play WANDB_MODE=online
export HF_CHECKPOINT_REPO=xudongwu/SafeSelfPlay-checkpoints        # 每个角色结束自动上传最终 checkpoint

# 收益格加速（必须设置；原因与实测见第 4 节「收益格加速设置」）。三项都不是矩阵/合约字段，不改变收益值。
export CELL_STREAMING=1         # 流式收益格：连续批处理，每局单独推进，不再等一批 64 条里最长的回复。镜像默认 0。
export CELL_ENFORCE_EAGER=0     # 用 CUDA graph 解码。带 LoRA 的格子被 vLLM 降级到 V0 引擎，eager 下解码慢 4 倍。
export CELL_SHARE_JUDGE_GPUS=0  # 不在 judge 所在的卡（GPU 0、GPU 3）上加开生成引擎：开着会在启动时 CUDA OOM。
```

## 3. 下载模型并检查（每个节点一次；工作目录在共享存储上则只需一次）

```bash
bash docker/run_llama8b.sh prepare
bash docker/run_llama8b.sh check
```

`check` 会打印四张 GPU 和 `image build: git_sha=…`，该 commit 必须与本文件末尾「镜像信息」一致，不是的话停下，镜像不对。

**上传权限预检（必须做，10 秒）**：`HF_TOKEN` 需要能**写** `xudongwu/SafeSelfPlay-checkpoints`，否则自动上传只会在日志里记一行 `checkpoint publish FAILED`，训练继续——上一轮就是这样一个 checkpoint 都没传上去。

```bash
docker run --rm --entrypoint bash -e HF_TOKEN -e NODE="$(hostname)" "$SSP_IMAGE" -c \
  'echo "preflight $NODE" > /tmp/p.txt && huggingface-cli upload xudongwu/SafeSelfPlay-checkpoints /tmp/p.txt "_preflight/$NODE.txt"'
```

报错（401/403）就换一个对该仓库有 write 权限的 token，再开始训练。

## 4. 启动（每个节点一个 run）

先 `nvidia-smi` 确认 4 张卡上没有残留进程（旧容器先 `docker stop -t 180 <name>`）。

```bash
# PSRO seed 8888 / 8889：已有 inv1 run，**不要用下面的新 suffix**，按本节末「已有的 PSRO run」用原 run 目录和原 suffix 续跑。
# 只有在旧 run 目录已经丢失、必须从头开始时，才用新 suffix：
#   TRAINING_SEED=8888 DETACH=1 CONTAINER_NAME=ssp-psro-s8888 \
#     bash docker/run_llama8b.sh psro --run-suffix "psro8_${RUN_LABELER}_inv4_s8888" --rounds 8 --steps-per-role 300

# uniform（seed 8888）
TRAINING_SEED=8888 DETACH=1 CONTAINER_NAME=ssp-uniform-s8888 \
  bash docker/run_llama8b.sh uniform --run-suffix "uniform8_${RUN_LABELER}_inv4_s8888" --rounds 8 --steps-per-role 300

# naive self-play（seed 8888）
TRAINING_SEED=8888 DETACH=1 CONTAINER_NAME=ssp-naive-s8888 \
  bash docker/run_llama8b.sh naive --run-suffix "naive6_${RUN_LABELER}_inv4_s8888" --rounds 6 --steps-per-role 100

docker logs -f ssp-psro-s8888      # 第一行应为 `Starting psro: …; reward mode=legacy`
docker stop -t 180 ssp-psro-s8888  # 需要停止时
```

- racing（`RACE_CELLS=1`）是镜像默认：每格先跑 256 局 pilot，只有 Nash 相关的行/列补到 4000 局。
  新 attacker / 新 defender 那一行/列总是跑满 4000 局。按第 2 节的加速设置，一个 4000 局的格子在 4×H200 上约 15 分钟
  （未加速时带 LoRA 的格子要 70–140 分钟）。
- 中断后用**同一条命令、同一个 `--run-suffix`** 续跑；不要改 seed 或参数后复用 suffix（会被拒绝）。
- 失败会在容器内自动重试（最多 5 次，间隔 60 秒），从 `state.json` 续跑。
- 已经在跑、但启动时没设第 2 节三个 `CELL_*` 变量的 run：`docker stop -t 180 <name>`，在 shell 里 export 这三个变量，
  再用**同一条命令、同一个 suffix** 重启。它们不是合约字段，已算好的格子保留，续跑不会被拒绝。

### 已有的 PSRO run（invalid-move reward，`inv1`）改用 v6 续跑

PSRO s8888 / s8889 的旧 run 与 v6 的 reward、矩阵设置完全相同，**不要从头跑**，用 v6 + **原来的 run 目录和 suffix** 续跑：
已完成的角色和收益格全部保留，只有中断的那个角色从 step 0 重训（`PSRO_MIDRUN_RESUME=0`）。
我们已用 v4 的代码对 s8888 在 HF 上的 `state.json` 做过续跑模拟：合约通过、A1 的 checkpoint SHA 校验通过，下一步正是训练 D1。

**续跑的四个必须条件**（任何一个不满足都会变成从头跑或报错）：

1. **同一个 run 目录**：checkpoint 在 FSx 上，`state.json` 里记录的是绝对路径，例如
   `/shared/dedicated-fsx-data-repo-training-phx-gl-eks-us-west-2/home/zzhihan/selfplay/runs/inv1-45906ffc-s8888-g8/psro/role_lora_zero_sum_psro/...`。
   新 pod / 容器必须把同一个 FSx 目录挂在**同一个路径**，`SSP_ROOT` 指向同一个 run 目录。
2. **同一个 suffix**：`psro8_local_inv1-45906ffc-s8888-g8` / `psro8_local_inv1-45906ffc-s8889-g8`（与原来逐字相同）。
3. **同样的训练参数**：`TRAINING_SEED`、`--rounds 8`、`--steps-per-role 300` 以及其他环境变量与原来一致（不一致会报 `Training settings changed`，不会训练）。
4. **镜像换成 v6**，并加上三个加速变量（它们不属于训练参数，加上不会导致上面的报错）：
   ```bash
   export CELL_STREAMING=1 CELL_ENFORCE_EAGER=0 CELL_SHARE_JUDGE_GPUS=0
   ```

**启动前检查**（路径按你们的 run 目录替换）：

```bash
RUN=<SSP_ROOT>/psro/role_lora_zero_sum_psro/psro8_local_inv1-45906ffc-s8888-g8
ls $RUN/state.json $RUN/training/A1/ckpt/global_step240_hf/adapter_model.bin   # 两个都必须存在
```

**启动后检查**（driver 日志前几行）：
- **不能**出现 `no state at … initializing a fresh cold run`——出现就说明没找到旧 run，立刻停掉。
- s8888 应出现 `next action: {'kind': 'oracle', 'role': 'defender', 'target': 'D1'}`。
- 出现 `Missing adapter checkpoint` 说明挂载路径不对，停掉检查条件 1。

**什么时候停旧容器**：已经死掉的 run（s8888）直接重启；还在跑的 run（s8889）**等当前角色的 final checkpoint 上传到 HF 之后**
（进入收益格阶段）再 `docker stop -t 180`，否则当前角色要从 step 0 重训。

**s8888 的 D1 要盯一下**：上次 D1 在 step 90 之后 defender 的 CoT 格式违规率升到 91%→100%，被塌缩保护（abort）主动停掉，
这是训练不稳定，不是代码 bug。v6 会从 step 0 重训 D1（失败自动重试最多 5 次）。请看 W&B
`defender/cot_format_violation`：**连续 20 步超过 0.3 就停掉并通知我们**，不要让它把 5 次重试都跑完。

备份：s8888 的 A1 final checkpoint 与 `state.json` 在 HF（`xudongwu/SafeSelfPlay-checkpoints/psro8_local_inv1-45906ffc-s8888-g8/`），
SHA 与 FSx 上的一致；FSx 上的 run 目录丢失时再找我们用它恢复。

### 收益格加速设置（为什么要设第 2 节那三个 `CELL_*`）

**问题**：除 A0×D0 外，所有收益格都慢得多。同一代码、4000 局：A0×D0 21 分钟，A1×D0 73 分钟（远程节点约 2 小时 20 分钟）。

**原因**：只要格子里有训练过的 LoRA（A0×D0 以外的所有格子），收益格就用 `enable_lora=True` 建引擎，
vLLM 0.8.2 的 V1 引擎不支持 LoRA，会降级到 V0（cell 日志里有 `LORA is experimental on VLLM_USE_V1=1. Falling back to V0 Engine`）。
V0 + eager 模式解码速度只有 V1 的 1/4（H200 实测 1716 vs 6842 tok/s）。再加上旧的 lockstep 路径每批 64 条要等最长的回复，
格子所在 GPU 利用率只有约 23%。收益值本身没问题，只是慢。

**两个开关的作用**：
- `CELL_ENFORCE_EAGER=0`：用 CUDA graph 解码，V0+LoRA 恢复到 5460 tok/s（3.2 倍）。单独使用（lockstep）时 GPU 利用率到 95%。
- `CELL_STREAMING=1`：流式收益格，每张卡一个连续批处理引擎，每局单独推进（攻击 → 防御 → 判分），没有批次等待。
  流式模式下引擎默认就用 CUDA graph；仍显式设置 `CELL_ENFORCE_EAGER=0`，以防关掉流式时退回 eager。

**GPU 共享与 OOM**：流式模式默认（`CELL_SHARE_JUDGE_GPUS=1`）会在 WildGuard（GPU 0）和 topic judge（GPU 3）所在的卡上
再开生成引擎。实测 4 个格子里 3 个的这类引擎启动时 CUDA OOM（`engine pool: engines failed … Tried to allocate 3.40 GiB`）：
引擎按整卡 90% 的上限定大小，vLLM 显存分析之后还要再分配 3.4 GiB，judge 旁边放不下。
不致命，格子会只用剩下的引擎算完，但会留下 traceback、白占那张卡，还有把 judge 挤到 OOM 的风险。
设 `CELL_SHARE_JUDGE_GPUS=0` 后每个格子只用自己的卡，不再碰 judge 的卡，速度几乎不变。

**实测**（4×H200，真实 PSRO driver `cold_psro_local.py`，4 个格子 A0/A1 × D0/D1，每格 1000 局，2 个格子并行）：

| 设置 | 耗时 | 问题 |
| --- | --- | --- |
| 原设置（lockstep + eager；单格测试 A1×D0 1000 局） | 单格 19.4 分钟 | 慢 |
| `CELL_STREAMING=0 CELL_ENFORCE_EAGER=0` | 4 格共 16.0 分钟 | 无 |
| `CELL_STREAMING=1 CELL_ENFORCE_EAGER=0` | 4 格共 8.3 分钟 | judge 卡上的引擎 3/4 次 OOM（不致命） |
| **`CELL_STREAMING=1 CELL_ENFORCE_EAGER=0 CELL_SHARE_JUDGE_GPUS=0`（采用）** | **4 格共 8.7 分钟** | **无** |

三种加速设置的收益值彼此一致，也与 4000 局的值一致，差异都在 1000 局的 95% 置信区间内（A1×D0：0.112–0.119 vs 0.1205；
A1×D1：−0.748 到 −0.744 vs −0.744）。
确认生效：`<workspace>/psro/role_lora_zero_sum_psro/<suffix>/logs/cells/*.log` 里应有 `engine pool ready on GPUs ['1']`（或 `['2']`）
和 `stream: … episodes judged` 行，且没有 `engines failed`。

## 5. 运行中要看的指标

| 什么时候 | 看什么 | 正常 | 异常 → 处理 |
| --- | --- | --- | --- |
| 每个 attacker 阶段 | W&B `attacker/debug/n_generated_benign` | 大致持平（不应持续下降） | 降到接近 0 → 通知我们 |
| 每个 attacker 阶段 | `train/topic/attacker_invalid_moves` | 逐步降到接近 0 | 持续上升 → 通知我们 |
| 阶段结束 | W&B 显示 `crashed` | 早停正常现象；`training/<A_k>/early_stop.json` 里 `returncode=0` 即正常 | `returncode≠0` → 看 `training.log` 末尾 |
| 每个 defender 阶段 | W&B `defender/cot_format_violation` | 接近 0（< 0.05） | 连续 20 步 > 0.3 → 停掉并报告（格式塌缩，会被保护机制以 returncode=2 停掉） |
| A1 结束后 | 进入矩阵计算，约 30 分钟内开始 D1 | 出现新的 D1 W&B run | 超过 1 小时无新 run → 看 `<workspace>/psro/role_lora_zero_sum_psro/<suffix>/logs/cells/*.log`，确认有 `stream:` 行（没有说明第 2 节的 `CELL_*` 没生效） |

在我们的 2×H200 测试里（同一 reward，A1 vs D0，截至 151 step），每批 benign 数量稳定在 26–36，
attacker 成功率 0.33 → 0.96，非法动作 0.38 → 0.02；用 step 40 checkpoint 算的 512 局收益格只需 516 个候选即填满（benign 漂移 0.8%）。上一轮 `shared_binary` 在 80–160 step 内 benign 降到 0，导致矩阵卡住。

## 6. Checkpoint 上传规则

1. 每个角色阶段（A1、D1、A2……）结束立即上传最终 checkpoint：
   `training/<A_k|D_k>/ckpt/global_step<N>_hf/`（N 为 early_stop 的停止 step，无早停取最大 N），
   加上 `early_stop.json`、`manifest.json`。设置了 `HF_CHECKPOINT_REPO` 时会自动上传，但**必须确认** repo 里出现
   `adapter_model.*`；上一轮自动上传静默失败了。失败时手动上传：
   ```bash
   huggingface-cli upload xudongwu/SafeSelfPlay-checkpoints \
     <run>/training/A1/ckpt/global_step<N>_hf <suffix>/A1 \
     --commit-message "<suffix> A1 step<N>"
   ```
2. 训练中每 50 step 把最新 checkpoint 上传到 `<suffix>/<A_k>/latest/`（覆盖）。
3. 每算完一个收益格，上传 `<run>/state.json` 和 `<run>/matrices/`。
4. 不要上传基座权重、`wandb/`、`topic_labels.sqlite3` 或任何 key。

## 镜像信息

| 项目 | 值 |
| --- | --- |
| 镜像 | `ghcr.io/wentsechen/safeselfplay:llama8b-h200-invalidmove-20260925-v6` |
| digest | `sha256:4df8a3c5bc9ee375fdf10e63c462fbf69515fdf48f7d6c4d66ebb330cc9a470b` |
| 构建 commit（`check` 应显示） | `98254744f30d24d5e1808b809c782910cfb7a6fc` |
| 默认配置 | `GAME_REWARD_MODE=legacy`、`ATTACKER_INVALID_MOVE=1`、`RACE_CELLS=1`、`CELL_STREAMING=0`、4×H200（`CELL_STREAMING` 由第 2 节改为 1） |

**v6 相对 v5（修复 A4 训练崩溃，所有 run 都应换 v6）**：WildGuard judge 输入超长导致训练退出的 bug。生成长度按策略模型 tokenizer 限 2048/步，但 WildGuard 用自己的 tokenizer 重新计数，连续符号/数字/空白会膨胀 3–4 倍，攻击+防守拼起来可超过 judge 的 8192 上限：vLLM 拒绝整批（HTTP 500，`Prompt length of 14500 is longer than the maximum model length of 8192`），且把该批已入队的请求留在引擎里，下一批返回多余结果（`Output count mismatch: expected 32, got 60`），客户端重试 12 次后训练退出。v6 在 judge 服务端把每一步截到 2048 个 WildGuard token（正常文本不受影响，截断会在 judge 日志打印 `WildGuard move capped`），并在 generate 失败时清空引擎。真实 WildGuard（H200）冒烟测试：旧代码复现上述两个报错；v6 三批全部 200、32/32 标签，正常样本标签与旧代码完全一致。其余训练、reward、设置与 v5 完全相同。**崩溃的 run 用 v6 + 原 run 目录 + 原 suffix 续跑**（不要换 suffix、不要从头跑；**s8889 stream fork 例外，见文首「v6 操作」**）；其它在跑的 run 可在下次重启时换 v6。构建 commit `457a03e`。

**v5 相对 v4**：topic judge 服务端合并并发请求成批生成，本地 judge 客户端并发 4→32（256 条改写 13.7 s → 3.4 s，255/256 判定相同），
训练每步约快 4 秒，收益格也更快。其余训练、reward、设置与 v4 完全相同；v4 上的 run 可直接用 v5、同一 suffix 续跑。

**v4 相对 v3**：
- 每个训练阶段结束后（包括早停），杀掉训练用 WildGuard judge 的 Ray 残留进程，并确认 4 张卡都已释放，再启动收益格；
  每次进入训练前也确认 4 张卡都已释放（v3 只检查 GPU 0）。
- 包含流式收益格代码（默认关闭；已在 H200 上测完，按第 2 节开启）。

**v3 相对 v2**：收益格结束切换到训练时，cell judge 可能未释放 GPU 0，导致下一个训练阶段起不来（seed 8889 曾卡在 A1 前）；
v3 强制清理并等待 GPU 0 空闲，driver 失败自动续跑。

旧标签 `…-invalidmove-20260923`、`-v2`、`-v3`、`-v4` **不要再用于新启动**。

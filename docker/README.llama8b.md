# Llama 8B abliterated：4×H200 Docker 交付说明

模型：`mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated`；奖励模型：`allenai/wildguard`。
本目录提供三组实验的构建文件、启动入口和镜像导出脚本。

**只收到源码小包时，请从 [QUICKSTART.source.llama8b.md](QUICKSTART.source.llama8b.md) 开始。**
该版本由老师执行 `build` 下载依赖与上游数据，再执行 `prepare` 下载模型；
源码包的 `START_HERE.md` 已按此顺序编排。下文的完整镜像导入方式适用于另行收到 `.image.tar.gz` 的情况。

本次交付默认使用 v6d 奖励 + attacker 非法动作（标签漂移 / 偏题）固定惩罚 `-3/-4`，
并适配 Claude Opus 5 的短 JSON 标签请求。先看包内 `START_HERE.md`，
其中先选择本地模型或 Opus API 做 label，再执行三个训练命令。

`.build.tar.gz` 是源码构建包；`.image.tar.gz` 是可 `docker load` 导入的完整镜像。
本次发布源码构建包，在目标服务器构建后的镜像标签为 `safeselfplay:llama8b-h200-invalidmove-20260924-v5`。
2026-09-23 版包含运行全程的本地 judge 服务、矩阵评估分卡、旧奖励补丁检测及 Hugging Face checkpoint 上传修复。
源码构建版在老师的服务器安装依赖并执行检查，实际版本写入镜像内 `/opt/runtime-packages.txt`。
构建脚本将 Git 提交写入 OCI `org.opencontainers.image.revision` 标签，便于核对重建结果。

## Docker 是什么

Docker 镜像把程序、Python/CUDA 用户态依赖和启动方式装进一个可搬运的环境；
从镜像启动的进程叫容器。老师的机器不需要逐个安装本项目的 Python 包。
容器共用宿主机内核，GPU 仍由宿主机驱动提供：镜像不会把一台普通电脑变成 H200 服务器。
参见 [Docker 容器介绍](https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-a-container/)。

## 0. 代码更新后必须重建镜像

镜像里的代码是 `COPY` 进去的，不是挂载的：运行时只绑定 `$WORKSPACE`（数据、
checkpoint、模型缓存），**不包含仓库代码**。因此拉取新代码后必须重建镜像，
否则容器里跑的仍是旧代码。判定模型分卡的修复就在其中——旧镜像会把两个 judge
放在同一张卡上，训练在第 0 步就以
`No available memory for the cache blocks` 失败。

```bash
git pull                                  # 取得已合并的修复
bash docker/run_llama8b.sh build          # 会把当前 commit 打进镜像
bash docker/run_llama8b.sh check          # 打印镜像内的 commit，并核对分卡修复
```

`check` 会读取镜像内 `BUILD_INFO` 并检查烘焙进去的 `role_phase.sh` 是否包含
`TOPIC_JUDGE_GPU`；镜像过旧时直接报错，不会等到训练开始才失败。
`$WORKSPACE` 不受影响，缓存与 checkpoint 会保留，只需替换镜像。

## 1. 服务器要求和构建

- Linux x86_64，单机 4×H200 141GB。该预设不适用于直接在 Mac 上训练。
- 可用的 Docker Engine、NVIDIA 驱动及
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。
- 构建时能访问 Docker Hub、GitHub、PyPI；首次下载模型时能访问 Hugging Face。
- 工作目录预留充足磁盘：模型、镜像、每代 LoRA 和 PSRO 原始对局均会占空间。
  Docker 镜像存储与下方工作目录可能位于不同磁盘，二者都要有余量。
- 如果集群由 Slurm 管理，在已分配的计算节点上启动容器；容器内部不需要运行 `sbatch`。

解压后进入目录（已有仓库则直接进入仓库根目录）：

```bash
tar -xzf safeselfplay-llama8b-h200-source.build.tar.gz
cd SafeSelfPlay-llama8b-h200

docker version
nvidia-smi
bash docker/run_llama8b.sh build
```

构建使用固定 digest 的 `vllm/vllm-openai:v0.8.2`、CUDA 12.4、Python 3.12、torch 2.6、
transformers 4.50、PEFT 0.15.2、DeepSpeed 0.16.5 和 Ray 2.44。
上游 Self-RedTeam 源码固定到 `0c56e503e8ae1b1b0fcd2214c92ea31fef1cb123`。
构建会解压上游训练数据并应用本仓库补丁；不在每次训练启动时重新安装依赖。
构建最后执行 `pip check` 和 LlamaFactory 导入检查，并将实际包版本写入 `/opt/runtime-packages.txt`。
vLLM 的旧版二进制依赖由基础镜像提供，请勿把它改成空白 Python 镜像重新安装。

## 2. 选择 label 模型：下面两种方式任选一种

默认 reward（无需设置）：v6d 奖励 + attacker 非法动作固定惩罚。

```bash
export GAME_REWARD_MODE=legacy          # 镜像默认值
export ATTACKER_INVALID_MOVE=1          # 镜像默认值
```

- 保留 v6d 的全部奖励项（Self-RedTeam v2：改写标签一致 ±0.5、拒答 ±0.5、有害回复 ±1、自身 CoT 格式 ±1），defender 奖励与 v6d 完全相同。
- attacker **非法动作**：改写后的有害性标签与 seed 不一致（benign↔harmful 任一方向，WildGuard 标签缺失也算），或 label 模型判定偏题，记 **-3**；两者同时出现记 **-4**；CoT 格式违规再 -2。与 defender 如何回复无关。
- 合法动作最低 0（CoT 违规时 -2），因此任何非法动作都严格更差。WildGuard 无法分类的回复，attacker 最多得 0。
- PSRO 矩阵沿用 v6d 的零和投影，但与训练一致：偏题、harmful seed 被改成非 harmful（或无标签）的对局，attacker 收益固定为 -1（收益区间下界，合法对局最低 -0.75）；benign→harmful 对局仍丢弃。矩阵版本号为 `role-lora-psro-terminal-projection-v4+invalid-move-v1`。
- 为什么改：`shared_binary` 直接丢弃 benign→harmful 对局且对忠实的 benign 改写罚 -0.75，attacker 在 80–160 step 内把全部 benign seed 改成 harmful，PSRO 收益格收集不到 benign 局而停住。
- 与旧版 `shared_binary` 结果不可混用，请使用新的 `--run-suffix`（下文统一为 `_inv1`）。需要旧行为时设 `ATTACKER_INVALID_MOVE=0 GAME_REWARD_MODE=shared_binary`。

两种 label 方式都输出 binary 偏题标签（`true` = 偏题），供上面的非法动作判定和矩阵偏题惩罚使用。

判定模型默认是 **方式 A（本地 Llama 8B abliterated）**，不需要任何 API key。

### 方式 A：用本地模型，不需要 Claude API key

```bash
export TOPIC_JUDGE_BACKEND=local
export RUN_LABELER=local
# 默认复用已下载的 Llama 8B abliterated，独立加载为冻结的 label 模型。
# 可选：换成其他本地模型；仓库 ID 会由 prepare 下载到缓存中。
# export TOPIC_LOCAL_MODEL='组织名/模型名'
# 或填容器内可见的路径，例如 /workspace/models/my-topic-model。
```

容器自动配置本地接口并启动 label 服务，无需手动填写 `TOPIC_API_*`。
即使 shell 留有旧 Claude 配置，本地模式也会覆盖接口与 key，避免调用云端。
同一个本地服务贯穿全部训练阶段和 PSRO 矩阵计算，不会每个阶段换一次 label 模型。

本地模型使用容器内 GPU 3（`TOPIC_JUDGE_GPU=3`），默认显存比例 0.13（实测约 17 GiB），上下文 4096；
训练仍使用四张卡，显存预留与已验证的 A3 实验一致（`UTIL=0.28`、`OPP_UTIL=0.26`、`VLLM_SLEEP=0`），
判定模型必须与 WildGuard 分卡：vLLM 的 `gpu_memory_utilization` 是整卡比例并会扣掉同卡上
其它进程已占的显存，两个 judge 叠在一张卡上时第二个连权重都放不下，直接以
`No available memory for the cache blocks` 退出。
PSRO 算矩阵时，GPU 0 运行 WildGuard，GPU 1、2 运行两个 cell worker，GPU 3 保留给 label。
自定义更大的模型需重新核对显存，当前默认预算按 8B 准备。
日志在持久化目录的 `topic_judge.log`，退出时自动清理该服务。

### 方式 B：用 Claude Opus 5 API

```bash
export TOPIC_JUDGE_BACKEND=anthropic
export TOPIC_API_PROVIDER=anthropic
export TOPIC_API_BASE_URL=https://api.anthropic.com
export TOPIC_API_MODEL=claude-opus-5
export RUN_LABELER=opus5
read -rsp 'Anthropic API key: ' TOPIC_API_KEY; echo
export TOPIC_API_KEY
# 或改用：export TOPIC_API_KEY_FILE=/absolute/path/to/anthropic.key
# 启动脚本会将 key 文件只读挂载到容器中。
```

模型 ID 来自 [Anthropic 官方文档](https://platform.claude.com/docs/en/models/overview)。
这里使用 Anthropic 原生 Messages API；key 由运行者提供，交付包没有内置 Claude key。
客户端明确关闭 Opus 5 默认 thinking、使用 low effort 和 128-token 预算，只收取 JSON 标签。
这一方式不加载本地 label 模型，PSRO 矩阵使用三个 cell worker。

两种方式的有效标签都缓存到持久化目录的 `topic_labels.sqlite3`。
模型名写入运行记录和缓存身份；换模型必须用新的 run suffix，不混用两种模型的结果。
无法获得有效标签时明确报错，不自动回退到另一种模型，也不把失败当作不偏题。
本地模型与 Claude 的判定质量可能不同，相同 reward 公式不表示标签相同。

Hugging Face token 需要具备 WildGuard 的访问权限；先在其模型页面接受访问条款。
镜像不包含 API key，Docker 构建也会排除历史私有文件 `docker/wandb.fixed.json`。
训练前在宿主机配置下列变量，启动脚本会将它们传入容器：

```bash
read -rsp 'W&B API key: ' WANDB_API_KEY; echo
export WANDB_API_KEY
read -rp 'W&B entity: ' WANDB_ENTITY
export WANDB_ENTITY
export WANDB_PROJECT=self-play
export WANDB_GROUP=upstream-selfredteam-role-lora-v2
export WANDB_MODE=online
```

三个方法各角色使用独立 run name；由名称生成稳定 run ID，重启沿用原 run。
每步日志、表格，以及 eval 每 10 step、从第 50 step 开始的节奏沿用原训练器。

```bash
# 持久化目录；三个实验复用这里的模型缓存。以后继续使用同一目录。
export SSP_DOCKER_WORKSPACE="$HOME/safeselfplay-data"
export GPU_DEVICES=0,1,2,3

# 隐藏输入 token，避免写入 shell 历史。无需把 token 发给老师。
read -rsp 'HF token: ' HF_TOKEN; echo
export HF_TOKEN

# W&B 凭据使用上面设置的运行时环境变量。

bash docker/run_llama8b.sh prepare
bash docker/run_llama8b.sh check
```

`prepare` 下载 base/judge 和所选本地 label 模型，不需要 GPU。`check` 检查判定配置、缓存路径和四张 GPU，
不会启动本地 label 引擎或请求 Claude。模型就绪后可 `unset HF_TOKEN`。
模型缓存、日志和 checkpoint 保存到宿主机工作目录；退出或删除容器不会删掉这些文件。
镜像默认以 root 运行，输出文件可能归 root 所有，交接时保留此目录的可读写权限。

## 3. 三组训练命令

**一轮 = 先训练 attacker A_i，再训练 defender D_i。** A0/D0 是同一个未训练的 base，
作为两种角色参与策略池。三组分别顺序运行，每组独占四张卡。

| 实验 | 轮数 | 每角色 step 上限 | 初始化方式 | 对手 |
| --- | --- | --- | --- | --- |
| PSRO | 8 | 300 + 早停 | 每代从 base 新建 LoRA | 历史策略池的 Nash 混合 |
| Uniform history | 8 | 300 + 早停 | 每代从 base 新建 LoRA，与 cold PSRO 一致 | 历史策略池均匀采样 |
| Naive self-play | 6 | 100 + 早停 | 继承上一代同角色 LoRA，重建优化器 | 只对战最新对手 |

```bash
# 1. PSRO：A1/D1 到 A8/D8，最终含 base 的 9×9 收益矩阵
bash docker/run_llama8b.sh psro --run-suffix "psro8_${RUN_LABELER}_inv1" --rounds 8 --steps-per-role 300

# 2. 新 baseline：历史对手均匀采样，A1/D1 到 A8/D8
bash docker/run_llama8b.sh uniform --run-suffix "uniform8_${RUN_LABELER}_inv1" --rounds 8 --steps-per-role 300

# 3. Naive：只对战最新对手，A1/D1 到 A6/D6
bash docker/run_llama8b.sh naive --run-suffix "naive6_${RUN_LABELER}_inv1" --rounds 6 --steps-per-role 100
```

实际入口是 `docker/run_llama8b.sh` → `docker/llama8b_entrypoint.py`。
PSRO 调用 `scripts/slurm/cold_psro_local.py`；两个 baseline 调用
`scripts/slurm/history_selfplay.py`；三者最终均通过 `scripts/slurm/role_phase.sh` 训练。
脚本目录虽然叫 slurm，这些本地入口不依赖 Slurm 或 Modal 服务。

Uniform 的“平均”是**每局按相等概率抽一个历史策略**，不平均模型权重。
例如训练 A3 时对手是 D0/D1/D2，各 1/3；训练 D3 时是 A0/A1/A2/A3，各 1/4。
有限次数的随机对局，其实际计数不保证完全相等。Naive 的 A_i 对战 D_{i-1}，D_i 对战 A_i。

两个混合策略的每角色上限统一为 300 step，可能因早停提前结束；
naive 保留此前的 100-step 上限，也可以显式设置 `--steps-per-role 300`。
共同预设：LoRA rank/alpha 64、训练种子 8888、
rollout batch 128、train batch 32、micro batch 8。PSRO 每个收益格 4000 局，
训练完成时共 81 格；两个 baseline 不在训练期间额外求解收益矩阵。
因此轮数相同也不代表三种方法的总算力相同。

**当前源码预设统一设置 `GAME_REWARD_MODE=legacy`、`ATTACKER_INVALID_MOVE=1`、
`INIT_KL_COEF=0`、`OFFTOPIC_ENABLE=0`、`PSRO_MIDRUN_RESUME=0`。**
当前 naive 继承权重的路径无法保持正 KL 惩罚始终参考 base，因此三组统一关闭 KL 惩罚；
参考 KL 仍监控。偏题判定使用上方选定的本地模型或 Claude API；WildGuard 始终启用。
这些是明确的实验配置，与仓库裸脚本的 `INIT_KL_COEF=0.01`、`TOPIC_JUDGE_BACKEND=local` 默认值不同。
如改变 reward 配置，应三组一致修改并使用新的 run suffix。

调整预算使用 `--steps-per-role`；PSRO 格子预算使用 `--matrix-episodes 4000`。
先做短烟雾训练可以运行以下命令，但它仍会真实占用四张卡并训练：

```bash
SAVE_STEPS_CONTRACT=1 bash docker/run_llama8b.sh uniform --run-suffix "${RUN_LABELER}_smoke1" --rounds 1 --steps-per-role 2
```

### 原早停规则

入口是 `scripts/slurm/cold_psro_local.py::_build_criterion`，三组共用。
这套是完整 ladder 的规则；`run_single_oracle.py` 的较早实验默认值不同，不用于本预设。
默认 `LADDER_EARLY_STOP=1`、窗口 20 step，每个窗口至少有 3 个有效观测。

- **Attacker：平台期停止。** 主指标是 `attacker/info/successful_harmful_request_rate`。
  窗口平均值连续 3 个窗口没有比此前最佳值增加超过 0.01，且当前窗口
  `attacker/cot_format_violation ≤ 0.30`，达到最小 step 60 后停止。
  首个窗口用于建立最佳值，因此默认最早实际触发在 step 80。
- **Defender：满足阈值后停止。** 连续 3 个窗口同时满足攻击成功率平均值 ≤ 0.01、
  `defender/wildguard_actual_benign_compliance_acc ≥ 0.912`、
  `defender/cot_format_violation ≤ 0.10`，且达到 step 60。
  缺少合规率或格式指标的窗口不视为满足条件。
- **崩溃退出。** 任一角色的格式错误率连续 2 个窗口 ≥ 0.80，停止并将阶段标为失败，
  不把退化策略添加到历史池。

早停先等待触发 step 的完整 adapter 写完，再结束训练，使用该 step 的 checkpoint；
不选择训练集上的“最好 checkpoint”。checkpoint 等待上限保持原来的 900 秒，
超时则报错，不会把半写入文件当成可用模型。默认每 10 step 保存；
若调整保存间隔，驱动会按预算、窗口和间隔的最大公约数缩短间隔，确保停止点能保存。
`training/<角色>/early_stop.json` 记录窗口均值、配置和退出原因；`manifest.json`
记录实际训练步数。已经完成的早停阶段在重启时复用。

可通过 `ATK_MIN_DELTA`、`ATK_PATIENCE`、`ATK_MIN_STEP`、`ATK_CEILING_MAX`、
`DEF_THRESHOLD`、`DEF_GUARD_MIN`、`DEF_PATIENCE`、`DEF_MIN_STEP`、`EARLY_STOP_WINDOW`
调整规则；修改后使用新的 run suffix。`LADDER_EARLY_STOP=0` 才关闭早停。

## 4. 查看结果、停止和续跑

工作目录下：

```text
hf/                                              # 训练、WildGuard 及所选 label 模型缓存
topic_judge.log                                   # 本地 label 服务日志
psro/role_lora_zero_sum_psro/psro8_<labeler>_inv1/
  state.json                                     # 进度、策略池、收益格
  docker_run_settings.json                       # 本次容器实验设置，无 API key
  matrices/final_g8.json                          # 最终 9×9 矩阵及 Nash 概率
  matrices/final.json                             # 最新最终矩阵
  training/A1/training.log                        # 每阶段各有日志
  training/A1/early_stop.json                     # 窗口测量和停止原因
  training/A1/ckpt/global_step300_hf/              # 未早停时；早停则为实际 step
selfplay_baselines/uniform/uniform8_<labeler>_inv1/ # state.json / phases / training
selfplay_baselines/naive/naive6_<labeler>_inv1/     # 同上
```

PSRO 最终写入 `training_completed=true`；`completed` 仍为 false，
因为独立的标准 defender benchmark 尚未执行。两种 baseline 的 `completed=true` 仅表示训练结束。
三条训练命令不会自动提交 Modal benchmark，不会将未做的评测记为完成。
PSRO 产出的矩阵用于对手选择，不是独立 held-out benchmark。
实验结果和分析请写入 [EXPERIMENTS.md](../EXPERIMENTS.md)。

需要后台运行时，例如：

```bash
DETACH=1 CONTAINER_NAME=ssp-psro8 bash docker/run_llama8b.sh psro --run-suffix "psro8_${RUN_LABELER}_inv1" --rounds 8 --steps-per-role 300
docker logs -f ssp-psro8
# 需要停止时：
docker stop -t 180 ssp-psro8
```

重新执行**相同目录、相同 suffix、相同设置**的训练命令即可续跑。
已完成的角色和收益格复用；当前未完成角色按本预设从该阶段原定起点重训，
不恢复其中间 optimizer 状态。不要删除已被历史对手池引用的 checkpoint。
改变轮数/参数请使用新 suffix，容器记录会阻止混写不同实验。

同一工作目录有训练锁，防止重复启动。同一组 GPU 上即使用不同工作目录，也请勿并行跑三组。
奖励服务启动失败会报错退出；外部偏题 API 有批次期限、单次请求超时和有限重试，
失败不会被替换成不偏题。`legacy` 模式的本地可选服务才允许缺失判定。
运行中出错先查看对应 `training/<角色>/training.log`，不要靠反复清空输出目录解决。
奖励源码不再以 hash 变化阻止启动；checkpoint 的内容校验保留，以免历史对手被替换后污染实验。

## 5. 发给老师或离线搬运

**构建源码包**：把本包发给老师，解压后按第 1–3 节操作。包含代码和启动说明，
不包含模型权重、HF token、本机缓存或历史训练 checkpoint。
GitHub 分发不包含凭据，使用以下命令生成源码构建包：

```bash
python3 docker/package_llama8b.py --source-only-guide --output output/public-source.build.tar.gz
```

**已构建镜像**：在有 Docker 权限的机器 build 成功后执行：

```bash
bash docker/export_llama8b.sh /path/safeselfplay-llama8b-h200-labels.image.tar.gz

# 老师的服务器：导入镜像，然后在源码包目录运行 prepare/check/训练命令。
docker load -i /path/safeselfplay-llama8b-h200-labels.image.tar.gz
```

不要对 `.build.tar.gz` 执行 `docker load`；只有 `.image.tar.gz` 是可导入镜像。
镜像与模型缓存分开交付。目标服务器断网时，先在联网机器 `prepare`，
再复制整个工作目录中的 `hf/` 到目标工作目录，保留 Hugging Face 缓存符号链接。
随后可设置 `export HF_HUB_OFFLINE=1`。镜像构建本身仍需要网络，或在联网机器构建后导出。
参见 [Docker 镜像导出](https://docs.docker.com/reference/cli/docker/image/save/)。

### 下载到 janus 的 Mac

在 **Mac 终端**执行，使用已有的 SSH 别名 `sz-4090-d02`：

```bash
scp sz-4090-d02:/export/pgs/wuxudong/work/selfplay/SafeSelfPlay/docker/download_to_mac.command ~/Downloads/
bash ~/Downloads/download_to_mac.command
```

脚本下载两个包到 `~/Downloads/SafeSelfPlay-llama8b-labels-20260919`，使用 `rsync -P` 保留中断进度。
重新运行即可继续下载。Mac 用于保存和转交，实际训练在老师的 Linux 4×H200 服务器执行。

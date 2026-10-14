# 源码构建版：Llama 8B / 4×H200

这是 Docker **源码构建流程**，支持 GitHub checkout 和源码小包，使用当前主分支的本地 / Claude label 实现。
包含源码、Dockerfile、运行脚本和少量项目数据。镜像构建排除 API key，凭据通过运行时环境变量传入。
运行依赖、其余上游训练数据和模型权重由老师的服务器下载。

2026-09-23 版同步最新主分支，包含运行全程的本地 judge 服务、矩阵评估分卡、旧奖励补丁检测及 Hugging Face checkpoint 上传修复。
更新后需重新执行 `build`，才能让容器使用新版代码。

**只跑 PSRO 两个 seed：直接按 [`docker/RUN_PSRO_INVALIDMOVE.md`](RUN_PSRO_INVALIDMOVE.md) 操作（使用预构建镜像，无需 build）。**

**本版（分支 `wentse/v6d-invalid-move-reward`）默认 reward 改为 v6d + attacker 非法动作固定惩罚，见第 2 节。镜像标签为 `…-invalidmove-20260924-v5`（v2 含矩阵非法动作修正与 racing 默认）；已构建过旧镜像也必须重新 `build`，否则仍会运行旧代码。**

训练模型：`mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated`。按顺序执行：**联网构建 → 选择 label → 下载模型 → 三组训练**。
下列命令用于 Linux x86_64 / 4×H200 服务器的 Bash；Mac 用于下载和转交这个压缩包。

## 0. 准备老师服务器的环境

本预设在 **Linux x86_64、单机 4×H200 141GB** 上运行。下面安装命令以 Ubuntu 22.04 / 24.04 为例，
需要管理员权限；已配置好 Docker GPU 环境的服务器直接做下方环境检查。
Mac 只保存和转交文件。Slurm 集群应在已分配的 GPU 节点运行训练。

| 在哪里准备 | 内容 |
| --- | --- |
| 宿主机 | NVIDIA 驱动、Docker Engine、NVIDIA Container Toolkit |
| Docker 镜像内自动准备 | CUDA 12.4、Python 3.12、PyTorch 2.6、vLLM 0.8.2、训练依赖和数据 |
| 持久化工作目录 | `prepare` 下载的模型、W&B 日志、checkpoint、PSRO 对局与矩阵 |

先运行 `nvidia-smi` 确认宿主机驱动正常、能识别四张 H200。若该命令失败，先由管理员安装或修复驱动；
Container Toolkit 的安装不能代替 GPU 驱动。宿主机无需另建 Python 环境，也无需单独安装本项目的 CUDA 工具包。

### Ubuntu 尚未安装 Docker 时

以下步骤使用 Docker 官方软件源，适用于新装 Docker 的服务器；已有其他 Docker / containerd 安装时先按
[Docker 官方 Ubuntu 安装说明](https://docs.docker.com/engine/install/ubuntu/) 处理版本冲突。

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
```

启动脚本使用当前用户的 `docker` 命令。如果 `docker version` 显示无权连接 Docker，
请管理员按服务器管理规则授予 Docker 使用权限。采用 Docker 用户组的服务器可执行：

```bash
sudo usermod -aG docker "$USER"
```

执行后退出 SSH 并重新登录，使组权限生效，再继续后续步骤。

### 配置 NVIDIA Container Toolkit

尚未安装 Toolkit 时执行下面的安装命令；重配并重启 Docker 会影响正在运行的容器，
已有训练任务的节点应由管理员安排配置时间。步骤依据
[NVIDIA 官方安装说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 检查宿主机及 Docker GPU 访问

```bash
uname -m                            # 应为 x86_64
docker version                      # 应能连接 Server
nvidia-smi -L                       # 本节点应有四张 H200
docker run --rm --gpus all ubuntu:22.04 nvidia-smi
```

最后一条会下载一个小的 Ubuntu 检查镜像，应能在容器内看到 GPU；它不启动训练。
网络需能访问 Docker Hub、Ubuntu 软件源、NVIDIA 软件源、GitHub、PyPI 和 Hugging Face；
选择 Claude 时还需能访问配置的 API 地址。使用 HF token 前请取得 `allenai/wildguard` 的模型访问权限。

镜像导入后约占 20 GB 量级，源码构建还需要额外缓存和临时空间；模型权重、训练结果另占空间。
先查看 Docker 的存储目录和工作目录所在磁盘，确保均有余量：

```bash
docker info --format '{{.DockerRootDir}}'
mkdir -p "$HOME/safeselfplay-data"
df -h "$HOME/safeselfplay-data"
# 再对上面显示的 DockerRootDir 执行 df -h，检查它所在的磁盘。
```

## 1. 联网构建 Docker 镜像

服务器需已安装 Docker Engine、NVIDIA 驱动和 NVIDIA Container Toolkit。
构建时需要访问 Docker Hub、Ubuntu 软件源、GitHub、PyPI；下载模型需要访问 Hugging Face。
小包只是减少交接文件大小，服务器仍需下载完整运行环境和模型，并预留相应磁盘空间。

从 GitHub 获取代码（已克隆则进入现有仓库）：

```bash
git clone https://github.com/xudongwu-0/SafeSelfPlay.git
cd SafeSelfPlay
git rev-parse HEAD   # 查看本次使用的代码版本
```

如果收到的是源码压缩包，使用下面的解压方式进入目录即可：

```bash
tar -xzf safeselfplay-llama8b-h200-source.build.tar.gz
cd SafeSelfPlay-llama8b-h200
```

进入代码目录后构建镜像：

```bash
export SSP_IMAGE=safeselfplay:llama8b-h200-invalidmove-20260924-v5
export SSP_DOCKER_WORKSPACE="$HOME/safeselfplay-data"
export GPU_DEVICES=0,1,2,3

docker version
nvidia-smi
bash docker/run_llama8b.sh build
```

`build` 自动拉取固定版本的 CUDA/vLLM 基础镜像，安装 Python 依赖，下载固定版本的上游源码和数据，生成训练工作副本及 SFT 数据，并执行 `pip check` 与训练模块导入检查。
首次构建可能较久；成功后继续第 2 节。此包不含预制镜像，**不需要执行 `docker load`**。

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

本地模型使用容器内 GPU 3，默认显存比例 0.13、上下文 4096；训练仍使用四张卡，
rollout / opponent 显存比例为 0.28 / 0.26，rollout 引擎不休眠。
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

## 3. 下载训练模型并检查环境

```bash
read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
bash docker/run_llama8b.sh prepare
bash docker/run_llama8b.sh check
```

HF token 需有 `allenai/wildguard` 的访问权限。`prepare` 不使用 GPU；`check`
检查配置、模型缓存和四张 GPU，不会启动本地 label 引擎或调用 Claude API。
GitHub checkout 不含 W&B key。训练前在同一个 Bash 会话配置 W&B：

```bash
read -rsp 'W&B API key: ' WANDB_API_KEY; echo
export WANDB_API_KEY
read -rp 'W&B entity（有写权限的团队或用户名）: ' WANDB_ENTITY
export WANDB_ENTITY
export WANDB_PROJECT=self-play
export WANDB_GROUP=upstream-selfredteam-role-lora-v2
export WANDB_MODE=online
```

Docker 构建会排除历史私有文件 `docker/wandb.fixed.json`，请使用上述运行时配置。
如果不使用 W&B，可显式设置 `WANDB_MODE=disabled` 并清除 `WANDB_API_KEY`。

可选：将每个角色的最终 checkpoint 上传到自己的 Hugging Face 仓库。
默认关闭；需要上传时在训练前设置（token 需有写权限）：

```bash
export HF_CHECKPOINT_REPO='your-account/your-checkpoints'
export HF_CHECKPOINT_PRIVATE=1
# 保留已配置的 HF_TOKEN；启动脚本会传入这些变量。
```

上传失败会记录错误并保留本地训练结果。未设置 `HF_CHECKPOINT_REPO` 时不会上传。

## 4. 三个训练命令

同一台四卡节点上依次运行，一轮是先训练 attacker，再训练 defender。
使用新的 run suffix，避免与旧版 reward 的结果混用。

```bash
# PSRO：8 轮，每角色最多 300 step + 原早停，Nash 混合对手
bash docker/run_llama8b.sh psro --run-suffix "psro8_${RUN_LABELER}_inv1" --rounds 8 --steps-per-role 300

# Uniform history：8 轮，每角色最多 300 step + 原早停，历史对手等概率采样
bash docker/run_llama8b.sh uniform --run-suffix "uniform8_${RUN_LABELER}_inv1" --rounds 8 --steps-per-role 300

# Naive self-play：6 轮，每角色最多 100 step + 原早停，只对战最新对手
bash docker/run_llama8b.sh naive --run-suffix "naive6_${RUN_LABELER}_inv1" --rounds 6 --steps-per-role 100
```

Uniform 包含 base 策略，例如 D0/D1/D2 各以 1/3 概率被抽取，不平均模型权重。
PSRO 和 uniform 每代从 base 初始化；naive 继承上一代同角色 LoRA。
三组关闭 KL 惩罚，保留原早停；训练器仍使用 REINFORCE advantage。
首次可先运行两步测试：

```bash
SAVE_STEPS_CONTRACT=1 bash docker/run_llama8b.sh uniform --run-suffix "${RUN_LABELER}_smoke1" --rounds 1 --steps-per-role 2
```

该测试会使用四张 GPU 并实际运行所选 label 模型；选择 Claude 时会产生 API 请求。详细早停、续跑和结果位置见
解压目录下的 `docker/README.llama8b.md`。源码包元信息见 `BUILD_BUNDLE.json`。
构建时会执行依赖和训练模块导入检查。

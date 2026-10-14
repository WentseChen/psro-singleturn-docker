# Llama 8B / 4×H200：本地模型 / Claude Opus 5 label 与三组训练

本包包含本地判定模型分卡、默认偏题惩罚 2 和端口冲突检查。训练模型是
`mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated`；label 模型只负责偏题判定，训练过程中保持冻结。
在 Linux 4×H200 服务器的 **Bash** 中执行；Mac 只用于下载和转交文件。

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

## 1. 导入镜像并展开启动脚本

```bash
tar -xzf safeselfplay-llama8b-h200-labels.build.tar.gz
docker load -i safeselfplay-llama8b-h200-labels.image.tar.gz
cd SafeSelfPlay-llama8b-h200
export SSP_IMAGE=safeselfplay:llama8b-h200-invalidmove-20260924-v5
export SSP_DOCKER_WORKSPACE="$HOME/safeselfplay-data"
export GPU_DEVICES=0,1,2,3
```

若只有源码包，在解压目录执行 `bash docker/run_llama8b.sh build`。
已有完整镜像则直接使用 `docker load`，无需再次构建。

## 2. 选择 label 模型：下面两种方式任选一种

两种方式都使用 binary 标签：偏题 `true` 扣 2，不偏题 `false` 这一项为 0。
WildGuard 继续判断有害性、拒答；训练和 PSRO 矩阵共用
`R_A=(r_A-r_D)/4-2*b`，`R_D=-R_A`。

```bash
export GAME_REWARD_MODE=shared_binary
export TOPIC_PENALTY=2
```

`TOPIC_PENALTY=2` 保持为默认值。当前 reward 版本为 `shared-terminal-binary-topic-v2`：
`r_A,r_D` 保留各自的 CoT 格式奖励，再叠加偏题惩罚；未罚分的投影使用
`[-1.5, 1.5]` 界，默认 attacker 总收益界为 `[-3.5, 1.5]`。
恢复格式项后，不能沿用旧版“罚 2 即保证所有偏题对局都不占优”的解释。
更新前的 v1 运行与新版奖励不同，请使用新的 `--run-suffix`，不要混用两版结果。

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
当前镜像与 GitHub checkout 均不含 key，
训练前按 [源码 Quick Start](QUICKSTART.source.llama8b.md) 第 3 节设置 W&B。

## 4. 三个训练命令

同一台四卡节点上依次运行，一轮是先训练 attacker，再训练 defender。
使用新的 run suffix，避免与旧版 reward 的结果混用。

```bash
# PSRO：8 轮，每角色最多 300 step + 原早停，Nash 混合对手
bash docker/run_llama8b.sh psro --run-suffix "psro8_${RUN_LABELER}_run1" --rounds 8 --steps-per-role 300

# Uniform history：8 轮，每角色最多 300 step + 原早停，历史对手等概率采样
bash docker/run_llama8b.sh uniform --run-suffix "uniform8_${RUN_LABELER}_run1" --rounds 8 --steps-per-role 300

# Naive self-play：6 轮，每角色最多 100 step + 原早停，只对战最新对手
bash docker/run_llama8b.sh naive --run-suffix "naive6_${RUN_LABELER}_run1" --rounds 6 --steps-per-role 100
```

Uniform 包含 base 策略，例如 D0/D1/D2 各以 1/3 概率被抽取，不平均模型权重。
PSRO 和 uniform 每代从 base 初始化；naive 继承上一代同角色 LoRA。
三组关闭 KL 惩罚，保留原早停；训练器仍使用 REINFORCE advantage。
首次可先运行两步测试：

```bash
SAVE_STEPS_CONTRACT=1 bash docker/run_llama8b.sh uniform --run-suffix "${RUN_LABELER}_smoke1" --rounds 1 --steps-per-role 2
```

该测试会使用四张 GPU 并实际运行所选 label 模型；选择 Claude 时会产生 API 请求。详细早停、续跑和结果位置见
解压目录下的 `docker/README.llama8b.md`；本次构建检查见 `BUILD_VALIDATION.json`。

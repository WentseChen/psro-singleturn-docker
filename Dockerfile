# Single-turn Cold PSRO (attacker/defender self-play), 8 generations, 8xH200.
# The two conda envs are PATH-PINNED (shebangs/RPATHs hardcode absolute paths), so the image
# reproduces the exact paths rather than relocating. Envs + code arrive as tarballs built by
# stage.sbatch (which must run on a GPU node to capture the node-local vllm30 env).
#
# devel base (nvcc) is required: vLLM's EngineCore JIT-compiles kernels with ninja at startup.
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash coreutils procps git curl ca-certificates libgomp1 numactl && \
    rm -rf /var/lib/apt/lists/*

# Reproduce the pinned directory layout.
RUN mkdir -p /scratch/wentsec/envs /zfsauton/scratch/wentsec/envs \
             /zfsauton/scratch/wentsec/coldpsro_slurm /zfsauton/scratch/wentsec/models \
             /zfsauton/scratch/wentsec/hf_srt

# --- conda envs (self-contained; bundle their own torch+CUDA) ---
# ctx/vllm30.tar.zst  -> /scratch/wentsec/envs/vllm30       (rollout: vLLM 0.30.0)
# ctx/srt.tar.zst     -> /zfsauton/scratch/wentsec/envs/srt (training: openrlhf/deepspeed)
ADD ctx/vllm30.tar /scratch/wentsec/envs/
ADD ctx/srt.tar    /zfsauton/scratch/wentsec/envs/

# --- code + small data (COPY, not models) ---
ADD ctx/ssp_v10xm_g2.tar   /zfsauton/scratch/wentsec/
ADD ctx/work_v9zp_g2.tar   /zfsauton/scratch/wentsec/coldpsro_slurm/
ADD ctx/role_phase.tar     /zfsauton/scratch/wentsec/coldpsro_slurm/async_prof/
ADD ctx/sft_data.tar       /zfsauton/scratch/wentsec/

COPY psro.env entrypoint.sh fetch_models.sh push_checkpoints.sh /zfsauton/scratch/wentsec/coldpsro_slurm/psro_docker/
RUN chmod +x /zfsauton/scratch/wentsec/coldpsro_slurm/psro_docker/*.sh

# NO weights are baked and NO checkpoints of ours are used. fetch_models.sh pulls only public,
# off-the-shelf models at first start: the 3 judges + the abliterated Llama-3.1-8B that doubles as
# the BASE policy. Cold PSRO trains from generation 0 on that open base; each oracle is a LoRA r64.
# No pi0 / no SFT checkpoint is required.
WORKDIR /zfsauton/scratch/wentsec/coldpsro_slurm/psro_docker
ENTRYPOINT ["/bin/bash", "-lc", "./fetch_models.sh && ./entrypoint.sh"]

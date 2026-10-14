# Single-turn Cold PSRO (attacker/defender self-play), 8 generations, 8xH200 — SOURCE BUILD.
#
# NOTHING of ours (no env tarball, no data, no checkpoints) is shipped. The image is built entirely
# from source + public packages + public upstream data:
#   - training/rollout env rebuilt by pip from the exact working freeze (requirements-srt.txt),
#   - the Self-RedTeam working tree + SFT data GENERATED at build time by prepare_sources.py /
#     prepare_sft_data.py from the public upstream repo (pinned commit) + a small in-repo seed,
#   - the 3 judge models + base policy pulled at first *run* (all public, off-the-shelf).
#
# The pinned absolute paths below are kept IDENTICAL to our internal layout so entrypoint.sh /
# psro.env need no edits — only the way these paths are populated changed (pip+generate, not untar).
#
# Our working env is Python 3.10 + torch 2.6.0+cu124 + vLLM 0.8.2, so we build on a CUDA 12.4 base
# and reconstruct it with pip (the public vllm/vllm-openai:0.8.2 image is py3.12 — wrong ABI for our
# cp310 wheels — so it is NOT used here).
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 MAX_JOBS=8
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3.10-dev python3-pip \
      git curl ca-certificates build-essential libaio-dev pkg-config procps util-linux numactl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python

# --- reproduce the pinned absolute layout (same paths the pipeline + env binaries pin to) ---
RUN mkdir -p /zfsauton/scratch/wentsec/envs /scratch/wentsec/envs \
             /zfsauton/scratch/wentsec/coldpsro_slurm/async_prof \
             /zfsauton/scratch/wentsec/hf_srt /zfsauton/scratch/wentsec/sft_data

# ========================= 1. training + rollout env (srt) =========================
# A venv at the pinned path, reconstructed from the EXACT working freeze (torch 2.6 / vLLM 0.8.2 /
# transformers 4.50 / peft 0.15.2 / deepspeed 0.16.5 / ray 2.44 / flash-attn 2.7.4.post1 cp310 wheel).
COPY docker/psro_singleturn/requirements-srt.txt /opt/requirements-srt.txt
RUN python -m venv /zfsauton/scratch/wentsec/envs/srt \
    && /zfsauton/scratch/wentsec/envs/srt/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /zfsauton/scratch/wentsec/envs/srt/bin/pip install --no-cache-dir -r /opt/requirements-srt.txt \
    && /zfsauton/scratch/wentsec/envs/srt/bin/python -m pip check || true
ENV SSP_PYTHON=/zfsauton/scratch/wentsec/envs/srt/bin/python

# ========================= 2. faithful-judge env (vllm30) =========================
# The Qwen3.5-9B faithful-rewrite judge needs a newer vLLM than 0.8.2 (new model arch), served in its
# own venv at the pinned rollout path. The EXACT pin lives only on the gpu2 node (the env is node-local
# and could not be frozen from the build host) — set it with --build-arg and CONFIRM via `pip freeze`
# of the live rollout env (see README "Validate on a GPU node").
ARG VLLM_JUDGE_VERSION=0.10.1.1
RUN python -m venv /scratch/wentsec/envs/vllm30 \
    && /scratch/wentsec/envs/vllm30/bin/pip install --no-cache-dir --upgrade pip \
    && /scratch/wentsec/envs/vllm30/bin/pip install --no-cache-dir "vllm==${VLLM_JUDGE_VERSION}"

# ========================= 3. pipeline code (COPY, not mounted) =========================
# Code is baked in -> after any code change you MUST rebuild the image (a stale image = stale code).
ENV SSP_REPO=/zfsauton/scratch/wentsec/ssp_v10xm_g2 \
    SSP_ROOT=/zfsauton/scratch/wentsec/coldpsro_slurm \
    SSP_UPSTREAM_SRC=/zfsauton/scratch/wentsec/coldpsro_slurm/upstream \
    SSP_WORK=/zfsauton/scratch/wentsec/coldpsro_slurm/work_v9zp_g2 \
    SSP_SFT_DIR=/zfsauton/scratch/wentsec/sft_data \
    PYTHONPATH=/zfsauton/scratch/wentsec/ssp_v10xm_g2
COPY . $SSP_REPO
COPY docker/psro_singleturn/role_phase_v9sb_flame.sh /zfsauton/scratch/wentsec/coldpsro_slurm/async_prof/role_phase_v9sb_flame.sh
COPY docker/psro_singleturn/entrypoint.sh docker/psro_singleturn/psro.env \
     docker/psro_singleturn/fetch_models.sh docker/psro_singleturn/push_checkpoints.sh \
     /zfsauton/scratch/wentsec/coldpsro_slurm/psro_docker/
RUN chmod +x /zfsauton/scratch/wentsec/coldpsro_slurm/psro_docker/*.sh \
             /zfsauton/scratch/wentsec/coldpsro_slurm/async_prof/role_phase_v9sb_flame.sh

# ========================= 4. GENERATE the work tree + SFT data (no data shipped) =========================
# Public upstream at the pinned commit -> patched working tree ($SSP_WORK) + SFT set ($SSP_SFT_DIR),
# from the in-repo seed data/safety_selfplay/attacker_rewrite_1180.jsonl (1180 rows, public-derived).
RUN git clone https://github.com/mickelliu/selfplay-redteaming "$SSP_UPSTREAM_SRC" \
    && git -C "$SSP_UPSTREAM_SRC" checkout --detach 0c56e503e8ae1b1b0fcd2214c92ea31fef1cb123 \
    && ( [ -f "$SSP_UPSTREAM_SRC/red_team/data/data.zip" ] \
         && python -m zipfile -e "$SSP_UPSTREAM_SRC/red_team/data/data.zip" "$SSP_UPSTREAM_SRC/red_team/data" || true )
RUN cd "$SSP_REPO" \
    && $SSP_PYTHON scripts/slurm/prepare_sources.py \
    && $SSP_PYTHON scripts/slurm/prepare_sft_data.py \
    && $SSP_PYTHON scripts/slurm/patch_defender_cot.py \
    && $SSP_PYTHON scripts/slurm/patch_offtopic_labels.py

# Build provenance.
ARG SSP_GIT_SHA=unknown
LABEL org.opencontainers.image.title="Cold PSRO single-turn (abliterated Llama, 8xH200)" \
      org.opencontainers.image.revision="$SSP_GIT_SHA"
RUN printf 'git_sha=%s\n' "$SSP_GIT_SHA" > /opt/BUILD_INFO \
    && $SSP_PYTHON -m pip freeze > /opt/runtime-packages-srt.txt

WORKDIR /zfsauton/scratch/wentsec/coldpsro_slurm/psro_docker
ENTRYPOINT ["/bin/bash", "-lc", "./fetch_models.sh && ./entrypoint.sh"]

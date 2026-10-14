# Cold PSRO (single-turn) — 8×H200 source-build handoff

This repo releases the **8×H200, PSRO** training setup we use to train the attacker/defender safety
self-play policies, as **source only** — no data, no environment, no checkpoints are shipped. Everything
is rebuilt on your box from public sources at `docker build` time.

- **Hardware:** a single node with **8× H200** (~140 GB each). This is an 8-GPU setup — not 4-GPU.
- **Method:** Cold PSRO, 8 generations (A0..A8 / D0..D8), single-turn, GRPO, LoRA rank 64.
- **Runs locally:** one Docker container, no Slurm/multi-node, **no Modal** — the driver is
  `scripts/slurm/cold_psro_local.py` and runs entirely on the box. (You'll see a coordinator *name*
  `modal_role_lora_zero_sum_psro` and `modal_*.py` modules in the tree — those are internal identifiers
  from the upstream codebase; the Modal control plane is stubbed out and never contacted.)

## → Start here

**[`docker/psro_singleturn/README.md`](docker/psro_singleturn/README.md)** — the full handoff: build, run,
validate, share results back. Everything you need is there.

Quick version:
```bash
git clone git@github.com:WentseChen/psro-singleturn-docker.git
cd psro-singleturn-docker
docker build -f docker/psro_singleturn/Dockerfile \
  --build-arg SSP_GIT_SHA=$(git rev-parse HEAD) -t psro:single-turn .
# then see docker/psro_singleturn/README.md §2 to run on 8× H200
```

The rest of this repository is the pipeline source code (an on-prem fork of
[`mickelliu/selfplay-redteaming`](https://github.com/mickelliu/selfplay-redteaming)); you do not edit it
by hand — the Docker build consumes it.

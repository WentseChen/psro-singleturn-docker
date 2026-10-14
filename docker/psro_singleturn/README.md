# Cold PSRO (single-turn) — source-build Docker handoff

Attacker/defender self-play safety RL (Self-RedTeam style), **single-turn**, **8 PSRO generations**,
on a single **8× H200** node. One driver process spawns everything inside the container — Ray, the
per-card vLLM rollout servers, the three judges (WildGuard / topic / faithful-rewrite), and the GRPO
training phase. **No Slurm**, no multi-node, no external orchestration.

- **Policy model:** public abliterated Llama-3.1-8B (`mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated`).
- **Judges:** WildGuard (harm), the same abliterated Llama (topic), Qwen3.5-9B (faithful-rewrite).
- **Reward:** `v8-table-r2-seedskip-faithful0.8`.

**We ship you NO data, NO environment, and NO checkpoints — only source.** Everything else is rebuilt
on your box from public sources at `docker build` time:

| Piece | Where it comes from | Shipped by us? |
|---|---|---|
| Training/rollout env (torch 2.6 / vLLM 0.8.2 / …) | `pip install -r requirements-srt.txt` | **no** (pip) |
| Faithful-judge env (newer vLLM for Qwen3.5-9B) | `pip install vllm==<pin>` in a 2nd venv | **no** (pip) |
| Self-RedTeam working tree + SFT data | generated at build by `prepare_sources.py` / `prepare_sft_data.py` from public upstream `mickelliu/selfplay-redteaming@0c56e50` + the in-repo 1180-row seed | **no** (generated) |
| Base policy + 3 judge models | pulled at first **run** (`fetch_models.sh`, pinned revisions) | **no** (HF) |
| Pipeline code + the 1180-row SFT seed | this git repo | yes (source only) |

You provide the **8× H200** and (to share results back) wandb + HF credentials. Cold PSRO trains from
generation 0 on the open base; each oracle is a fresh LoRA rank-64 adapter.

**Please share two things back** (§6): the **wandb run** and the **LoRA adapters pushed to a HF repo**.

> ⚠️ **This source-build path has not been built or run end-to-end by us** (our validation ran from the
> pinned conda envs on our cluster; the GPUs we hand this to are not ours). It is a faithful
> reconstruction from the exact working freeze + the proven upstream/prepare flow. **Do the two-step
> validation in §1.5 on your GPU node before a long run** — in particular confirm the faithful-judge
> vLLM pin (§1.5.3).

---

## 0. Host prerequisites

- Linux x86_64, a single machine with **8× H200** (~140 GB each). Not for a laptop/Mac.
- NVIDIA driver on the host + **Docker Engine** + the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (so `docker run --gpus all` sees GPUs). The toolkit does **not** replace the driver.
- Build needs network to Docker Hub, PyPI, GitHub; first run needs Hugging Face. `allenai/wildguard` is
  gated — accept its terms with your HF account first.
- Disk: image ~30 GB + HF cache ~55 GB (3 judges + base) + outputs. Docker's storage dir and your
  output volume may be on different disks — check both.
- If the box is Slurm-managed, start the container on an **allocated GPU node** (no `sbatch` inside).

Sanity-check:
```bash
uname -m                                   # x86_64
docker version                             # client reaches Server
nvidia-smi -L                              # 8 H200
docker run --rm --gpus all ubuntu:22.04 nvidia-smi
docker info --format '{{.DockerRootDir}}'  # then df -h that path AND your output dir
```

---

## 1. Get the code and build

```bash
git clone git@github.com:WentseChen/psro-singleturn-docker.git
cd psro-singleturn-docker
git rev-parse HEAD                         # the code version you are building

docker build -f docker/psro_singleturn/Dockerfile \
  --build-arg SSP_GIT_SHA=$(git rev-parse HEAD) \
  -t psro:single-turn .
```
The build: installs the pinned Python stack (torch 2.6.0+cu124, vLLM 0.8.2, transformers 4.50.0, peft
0.15.2, deepspeed 0.16.5, ray 2.44.0, flash-attn 2.7.4.post1) on a CUDA 12.4 base; creates a second
venv with the newer vLLM for the Qwen3.5-9B judge; clones public upstream at `0c56e50` and **generates**
the working tree + SFT data from the in-repo seed. No tarballs, no weights. First build is slow.

**Code is baked in (`COPY`), not mounted — after any code change, rebuild the image.**

### 1.5. Validate on a GPU node (do this before a long run)

We could not build/run this from our side; confirm these three on your node:

1. **Build succeeds** (watch for pip resolution / flash-attn wheel errors). The exact working freeze is
   in `requirements-srt.txt`; it is cp310 (Python 3.10) — the Dockerfile uses a py3.10 base to match.
2. **2-step smoke** (uses all 8 cards + real judges, finishes in minutes):
   ```bash
   docker run --rm --gpus all --shm-size=32g \
     -e RUN_TAG=smoke1 -e GENERATIONS=1 -e STEPS_PER_ROLE=4 \
     -v /ABS/psro_out:/zfsauton/scratch/wentsec/coldpsro_slurm/psro \
     -v /ABS/hf_cache:/zfsauton/scratch/wentsec/hf_srt \
     psro:single-turn
   ```
   Healthy = models fetch, driver prints `=== LIVE CONFIG ===`, `next action` → `training A1` → a
   `global_step*_hf` appears under `…/psro/smoke1/training/A1/ckpt/`, no `No available memory for the
   cache blocks` / OOM.
3. **Faithful-judge vLLM pin.** The Qwen3.5-9B judge needs a newer vLLM than 0.8.2, served in the second
   venv. We could not read the exact version from our node (that env is node-local). The Dockerfile
   defaults `--build-arg VLLM_JUDGE_VERSION=0.10.1.1`; if the judge fails to serve Qwen3.5-9B, set it to
   the version that supports that arch and rebuild. (If you have access to our gpu2 rollout env, the
   authoritative pin is `/scratch/wentsec/envs/vllm30/bin/pip freeze | grep -i '^vllm=='`.)

---

## 2. Run

Keep secrets out of shell history / `docker inspect` / `ps`: read them into host env with `read -rsp`,
then pass **names only** (`-e NAME`, no `=value`).

```bash
read -rsp 'W&B API key (blank = offline): ' WANDB_API_KEY; echo
read -rp  'W&B entity (blank = default): '  WANDB_ENTITY
read -rsp 'HF push token (blank = skip push): ' HF_TOKEN; echo
export WANDB_API_KEY WANDB_ENTITY HF_TOKEN
export HF_PUSH_REPO=your-org/coldpsro-singleturn

docker run --rm --gpus all --shm-size=32g \
  -e RUN_TAG=run1 \
  -e WANDB_API_KEY -e WANDB_ENTITY -e WANDB_PROJECT=coldpsro-singleturn-baseline \
  -e HF_PUSH_REPO -e HF_TOKEN \
  -v /ABS/path/to/psro_out:/zfsauton/scratch/wentsec/coldpsro_slurm/psro \
  -v /ABS/path/to/hf_cache:/zfsauton/scratch/wentsec/hf_srt \
  psro:single-turn
```

**Mounts:**
1. **output (required)** → `/zfsauton/scratch/wentsec/coldpsro_slurm/psro`. Results in `…/psro/$RUN_TAG/`;
   mount a host dir so they survive the container. (Container runs as root → outputs may be root-owned.)
2. **HF cache (recommended)** → `/zfsauton/scratch/wentsec/hf_srt`. Models download here on first start;
   mounting it means the ~55 GB download happens once.

**Env:** `RUN_TAG` (fresh = clean, reuse = resume, §5); `WANDB_API_KEY`/`WANDB_ENTITY`/`WANDB_PROJECT`
(omit → wandb `offline`, local files you can `wandb sync`); `HF_PUSH_REPO`+`HF_TOKEN` (auto-push adapters
at the end; else `./push_checkpoints.sh $RUN_TAG` later).

**Detach / stop:** add `-d --name psro-run1`; then `docker logs -f psro-run1`; stop with
`docker stop -t 180 psro-run1` (lets the driver flush checkpoints).

**Notes:**
- **`--shm-size=32g` required** (Ray/vLLM IPC + LoRA staging). Do **not** add `--ipc=host` — it makes
  Docker ignore `--shm-size` and fall back to the host `/dev/shm` (often tiny → OOM). Raise shm if tight.
- First start runs `fetch_models.sh` (base + 3 judges at pinned revisions), then the driver, which prints
  a `=== LIVE CONFIG ===` banner — eyeball `GENERATIONS` / `STEPS_PER_ROLE` / reward flags / seed.

---

## 3. GPU layout (8 cards) and why it matters

Every card runs a vLLM rollout engine; judges and trainers are pinned to specific cards. The split is
explicit in **`psro.env`** (source of truth), cards 0–7:

| vars (psro.env) | value |
|---|---|
| `ONEJOB_JUDGE_GPUS` (WildGuard) | `0,1,2` |
| `ONEJOB_FAITHFUL_JUDGE_GPU` (Qwen3.5-9B) | `3` |
| `ONEJOB_TOPIC_GPU` | `7` |
| `ONEJOB_TRAIN_CARDS_ATTACKER` / `_DEFENDER` | `6,7` / `4,5,6,7` |
| `ONEJOB_SERVER_UTILS` (per-card rollout mem frac) | `0.68×3, 0.92×5` (0–2 leave headroom for WildGuard) |

**Why not re-pack casually:** vLLM's `gpu_memory_utilization` is a *whole-card* fraction reduced by
whatever else sits on the card. **Two heavy judges (or judge + too-greedy rollout) on one card → the
second can't load its weights → it dies at step 0 with `No available memory for the cache blocks`.** If
you change the map, re-check the per-card budget; defaults are sized for 8B judges + the base on 141 GB.

---

## 4. What it does / what to expect

- **Init:** A0/D0 cold from the abliterated base (served directly; each oracle a fresh LoRA r64).
- **Loop:** alternately train a best-response attacker, then defender, scoring one payoff cell each, for
  `GENERATIONS=8` iterations (A1/D1 … A8/D8). Reward = `v8-table-r2-seedskip-faithful0.8`.
- **Throughput:** ~**1–2 h per oracle** on 8× H200; the full 8 gens ≈ most of a day.
- **One oracle = one role best-response**, up to `STEPS_PER_ROLE=600`, cut short by early stop:
  **Attacker** (`ATK_EARLY_STOP=1`): after ≥ `ATK_MIN_STEP=200` steps, stop when harmful-win hasn't
  improved by `≥ ATK_MIN_DELTA=0.01` over its best for `ATK_PATIENCE=5` windows. Early stop waits for the
  triggering step's adapter to finish writing, then uses **that** checkpoint (no best-on-train pick).
- **Monitor:** `tail -f …/psro/$RUN_TAG/driver.log` — `next action` → `training A1 …` → `completed action`.

---

## 5. Results, resume, cleanup

```text
…/psro/$RUN_TAG/
  state.json                               # durable progress: pools, payoff cells, completed flag
  driver.log                               # top-level driver log
  game_dump.jsonl                          # sampled games
  matrices/                                # payoff matrix (meta-game) + Nash mixtures
  cells/                                   # per-cell transcripts
  training/<ORACLE>/training.log           # per-oracle log (A1, D2, …)
  training/<ORACLE>/ckpt/global_step*_hf/  # the trained LoRA adapter (HF/PEFT)
…/psro/${RUN_TAG}_raw/                     # raw payoff rollouts
```
- **Done:** `state.json` has `completed: true`.
- **Resume is automatic + idempotent:** `state.json` is the only durable record — completed cells/oracles
  are skipped on re-entry; the entrypoint wraps the driver in a crash-retry loop
  (`DRIVER_MAX_ATTEMPTS=8`). To resume, relaunch with the **same `RUN_TAG`, same output mount, same settings**.
- **Don't delete a checkpoint still referenced as a historical opponent** (pollutes later cells). On an
  error, read `training/<ORACLE>/training.log`; don't "fix" it by wiping the output dir.
- Changing algorithm settings? Use a **new `RUN_TAG`**.
- **Offline:** run once on a networked box to fill the HF cache mount, copy it to the target, set
  `HF_HUB_OFFLINE=1`. The build itself still needs network (or `docker save`/`load` the image).

---

## 6. Share results back (please do this)

1. **wandb run** — if you passed `WANDB_API_KEY`, it's online under `coldpsro-singleturn-baseline`
   (override with `-e WANDB_PROJECT=…`); send the URL. Offline → `wandb sync …/psro/$RUN_TAG/wandb/offline-run-*`.
2. **LoRA adapters → HuggingFace** — each oracle's adapter is at
   `…/psro/$RUN_TAG/training/<ORACLE>/ckpt/global_step*_hf`. With `HF_PUSH_REPO`+`HF_TOKEN` they auto-push
   at the end; else `./push_checkpoints.sh $RUN_TAG`. Share `hf.co/$HF_PUSH_REPO` (also uploads `state.json`).

These adapters are the actual trained attacker/defender policies — the deliverable we need.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `docker build` fails on pip / flash-attn wheel | requirements are cp310 — the Dockerfile uses a py3.10 base; don't swap to a py3.12 base image. |
| Qwen3.5-9B judge fails to load / serve | the 2nd-venv vLLM pin is too old — rebuild with `--build-arg VLLM_JUDGE_VERSION=<supported>` (§1.5.3). |
| `No available memory for the cache blocks` at step 0 | two judges/engines on one card, or a card's rollout `UTIL` too high — keep the default card map (§3). |
| OOM / Ray or vLLM dies early, `/dev/shm` errors | `--shm-size` too small, or you passed `--ipc=host`. Use `--shm-size=32g`, drop `--ipc=host`. |
| `FATAL: no GPUs` / `not writable` | forgot `--gpus all`, or the output volume isn't writable. |
| `FATAL: base policy not found …abliterated…` | base download failed — check network / HF cache mount writable. |
| A model path "does not exist" after download | HF served a different revision; `fetch_models.sh` pins exact revisions — update the revision there if a repo moved. |
| `rc=1` driver retry in log | entrypoint retries + resumes on crash (`DRIVER_MAX_ATTEMPTS=8`); if it loops, read the last `driver.log` error. |
| Slow first step (~3–4 min) | one-time Triton/CUDA compile; later steps much faster. |

---

## 8. Knobs

All explicit in **`psro.env`** (no inherited defaults). Most-touched: `GENERATIONS=8`,
`STEPS_PER_ROLE=600`, `MATRIX_EPISODES=4096`, `GPUS=8` + the `ONEJOB_*` card split (§3), `WANDB_PROJECT`,
`HF_PUSH_REPO`, `DRIVER_MAX_ATTEMPTS`. The algorithm block (GRPO group size, KL targets, LRs, reward
`V8_*` flags, seeds) is copied verbatim from our live single-turn contract — **leave it as-is** to
reproduce the baseline; if you change it, use a new `RUN_TAG`.

---

## 9. Divergence from our internal run

Our gpu2 baseline used an SFT'd base (`pi0`, served via `ONEJOB_SHARED_BASE=1` = abliterated + a pi0
LoRA). To keep this fully off-the-shelf, the base here is the **raw abliterated Llama** served directly
(`ONEJOB_SHARED_BASE=0`). The pipeline code is identical (`ssp_v10xm_g2`); only the base weights differ.
Expect the attacker to be *weaker* than our internal run (which itself plateaued after a couple of
generations) — a clean open baseline, not a bit-for-bit reproduction.

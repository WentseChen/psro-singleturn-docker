# Cold PSRO (single-turn) — Docker handoff

Attacker/defender self-play safety RL (Self-RedTeam style), **single-turn**, **8 PSRO generations**,
on a single **8× H200** node. One driver process spawns everything inside the container — Ray, the
per-card vLLM rollout servers, the three judges (WildGuard / topic / faithful-rewrite), and the GRPO
training phase. **No Slurm**, no multi-node, no external orchestration.

- **Policy model:** public abliterated Llama-3.1-8B (`mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated`).
- **Judges:** WildGuard (harm), the same abliterated Llama (topic), Qwen3.5-9B (faithful-rewrite).
- **Reward:** `v8-table-r2-seedskip-faithful0.8`.

**Everything is off-the-shelf and open-source — no checkpoints of ours are used or shipped.** You
provide the **8× H200** (and, to share results back, wandb + HF credentials). Cold PSRO trains from
generation 0 on the open base; each oracle is a fresh LoRA rank-64 adapter.

**Please share two things back with us** when done (§6): the **wandb run** and the **LoRA adapters
pushed to a HuggingFace repo**.

> Docker here only packages the *software* (code + Python/CUDA user-space deps + launch). GPUs are
> still the host's — the image does not turn a non-GPU box into an H200 server. The container shares
> the host kernel and uses the host NVIDIA driver.

---

## 0a. Get the code + the env tarballs

Source (this repo) is on GitHub; the **18 GB `ctx/` tarballs are NOT in git** (two conda
envs + SFT data — too large, and path-pinned). Get both:
```
git clone git@github.com:WentseChen/psro-singleturn-docker.git   # source only
```
Then obtain `ctx/` (`vllm30.tar`, `srt.tar`, `sft_data.tar`, `role_phase.tar`,
`ssp_v10xm_g2.tar`, `work_v9zp_g2.tar`) **out-of-band** from us (shared drive / scp) and drop
it into the repo root as `ctx/`. You cannot regenerate `ctx/` yourself — `stage.sbatch` only
works on the GPU box that already has our pinned conda envs. With `ctx/` in place, go to §1 step 2.

---

## 0. Host prerequisites

- Linux x86_64, a single machine with **8× H200** (~140 GB each). Not for training on a laptop/Mac.
- NVIDIA driver working on the host, plus **Docker Engine** and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (so `docker run --gpus all` sees the GPUs). The toolkit does **not** replace the GPU driver.
- Network on first run to pull the public models (or pre-populate the HF cache volume, §5).
- Disk: **~70 GB** for the image + **~55 GB** HF cache (3 judges + base) + room for outputs (per-oracle
  LoRA + PSRO raw games). The Docker storage dir and your output volume may be on **different disks** —
  check both.
- If the box is Slurm-managed, start the container on an **allocated GPU node**; you do not run
  `sbatch` inside the container.

Sanity-check the host before building:
```
uname -m                                   # x86_64
docker version                             # client must reach the Server
nvidia-smi -L                              # should list 8 H200
docker run --rm --gpus all ubuntu:22.04 nvidia-smi   # GPUs visible inside a container
docker info --format '{{.DockerRootDir}}'  # then df -h that path AND your output dir
```

---

## 1. Build (two steps)

The image embeds two conda envs that are **path-pinned** (shebangs/RPATHs hardcode absolute paths),
and the `vllm30` rollout env is **node-local** — so the build context is produced on a GPU node,
then the image is built from it.

**Step 1 — stage the context (on the GPU box that has the envs):**
```
sbatch stage.sbatch      # or just run it on the GPU node
```
Fills `ctx/` with tarballs of the two envs + code + SFT data (~18 GB). Already run here, so if you
received a populated `ctx/` you can skip to step 2. Only redo this if you rebuild the envs.

**Step 2 — build the image (on your Docker host):**
```
docker build -t psro:single-turn .
```
The image recreates the pinned absolute paths (`/scratch/wentsec/envs/vllm30`,
`/zfsauton/scratch/wentsec/...`). **Do not relocate them** — the envs will break.

**Pinned stack** (bundled in the envs; CUDA base image `nvidia/cuda:12.6.3-devel`):

| env | role | key versions |
|---|---|---|
| `srt` | training (OpenRLHF / DeepSpeed) | torch 2.6.0+cu124, transformers 4.50.0, peft 0.15.2, deepspeed 0.16.5, ray 2.44.0 |
| `vllm30` | rollout + judge serving | vLLM 0.30 (node-local; bundles its own torch+CUDA) |

**Pipeline code** = `ssp_v10xm_g2` + `work_v9zp_g2` — the exact single-turn pipeline we validated on
gpu2 — plus `role_phase_v9sb_flame.sh`. **The code is `COPY`/`ADD`ed into the image, not mounted.**
So **after any code change you must rebuild the image** (step 2); a stale image runs stale code.

---

## 2. Run

Keep secrets out of your shell history and out of `docker inspect` / `ps`: read them with `read -rsp`
into host env vars, then pass the **names only** to `docker run` (`-e NAME`, no `=value`).

```bash
# secrets -> host env (not echoed, not in history)
read -rsp 'W&B API key (blank to run offline): ' WANDB_API_KEY; echo
read -rp  'W&B entity (blank for default): '     WANDB_ENTITY
read -rsp 'HF push token (blank to skip push): ' HF_TOKEN; echo
export WANDB_API_KEY WANDB_ENTITY HF_TOKEN
export HF_PUSH_REPO=your-org/coldpsro-singleturn   # where to push the LoRA adapters

docker run --rm --gpus all --shm-size=32g \
  -e RUN_TAG=run1 \
  -e WANDB_API_KEY -e WANDB_ENTITY -e WANDB_PROJECT=coldpsro-singleturn-baseline \
  -e HF_PUSH_REPO -e HF_TOKEN \
  -v /ABS/path/to/psro_out:/zfsauton/scratch/wentsec/coldpsro_slurm/psro \
  -v /ABS/path/to/hf_cache:/zfsauton/scratch/wentsec/hf_srt \
  psro:single-turn
```

**Mounts:**
1. **output (required)** → `/zfsauton/scratch/wentsec/coldpsro_slurm/psro`. Results land in
   `…/psro/$RUN_TAG/`; mount a host dir so they survive the container. (Image runs as root, so outputs
   may be root-owned — keep the dir read/writable when handing it over.)
2. **HF cache (recommended)** → `/zfsauton/scratch/wentsec/hf_srt`. Models download here on first
   start; mounting it means the ~55 GB download happens once and is reused.

**Env (nothing is required to *run*; the sharing ones are how you send results back):**
- `RUN_TAG` — names the run; fresh tag = clean run, **reuse the same tag = resume** (§5).
- `WANDB_API_KEY` (+ optional `WANDB_ENTITY`, `WANDB_PROJECT`) — logs metrics to wandb. **Without it the
  run still works** and wandb falls back to `offline` (local files under `…/psro/$RUN_TAG/wandb`; you
  can `wandb sync` later).
- `HF_PUSH_REPO` (+ `HF_TOKEN`) — at the end, the LoRA adapters are auto-pushed there. Without them the
  run still works; push later with `./push_checkpoints.sh $RUN_TAG`.

**Detach / stop:**
```bash
docker run -d --name psro-run1 --gpus all --shm-size=32g … psro:single-turn   # add -d + --name
docker logs -f psro-run1
docker stop -t 180 psro-run1          # give the driver time to flush checkpoints
```

**Notes:**
- **`--shm-size=32g` is required** (Ray/vLLM IPC + LoRA staging use `/dev/shm`). Do **not** also pass
  `--ipc=host` — that makes Docker ignore `--shm-size` and fall back to the host's `/dev/shm`, which is
  often tiny and OOMs mid-run. If 32 GB is tight, raise it; don't switch to `--ipc=host`.
- First start runs `fetch_models.sh` (pulls base + 3 judges **at pinned revisions**), then the driver.
- The driver prints a `=== LIVE CONFIG ===` banner at start — eyeball it to confirm `GENERATIONS`,
  `STEPS_PER_ROLE`, reward flags and seed before letting it run for a day.

---

## 2.5. Smoke test first (recommended)

One tiny run end-to-end before the real thing. It **still uses all 8 cards and the real judges** (so
it's a true plumbing test, not instant), but finishes in minutes, not a day:
```bash
docker run --rm --gpus all --shm-size=32g \
  -e RUN_TAG=smoke1 -e GENERATIONS=1 -e STEPS_PER_ROLE=4 \
  -v /ABS/psro_out:/zfsauton/scratch/wentsec/coldpsro_slurm/psro \
  -v /ABS/hf_cache:/zfsauton/scratch/wentsec/hf_srt \
  psro:single-turn
```
Healthy = models fetch, driver prints LIVE CONFIG, `next action` → `training A1` → a `global_step*_hf`
appears under `…/psro/smoke1/training/A1/ckpt/`, no `No available memory for the cache blocks` / OOM.
Use a **separate** `RUN_TAG` from the real run.

---

## 3. GPU layout (8 cards) and why it matters

Every card runs a vLLM rollout engine; the judges and trainers are pinned to specific cards. The split
is explicit in **`psro.env`** (the source of truth); defaults assume cards indexed 0–7:

| vars (psro.env) | value |
|---|---|
| `ONEJOB_JUDGE_GPUS` (WildGuard) | `0,1,2` |
| `ONEJOB_FAITHFUL_JUDGE_GPU` (Qwen3.5-9B) | `3` |
| `ONEJOB_TOPIC_GPU` | `7` |
| `ONEJOB_TRAIN_CARDS_ATTACKER` / `_DEFENDER` | `6,7` / `4,5,6,7` |
| `ONEJOB_SERVER_UTILS` (per-card rollout mem frac) | `0.68×3, 0.92×5` (cards 0–2 leave headroom for WildGuard) |

**Why you shouldn't casually re-pack this:** vLLM's `gpu_memory_utilization` is a *whole-card* fraction
and is reduced by whatever else already sits on that card. **Put two heavy judges (or a judge + a
too-greedy rollout engine) on one card and the second one can't even load its weights — it dies at step 0
with `No available memory for the cache blocks`.** If you change the card map, re-check the per-card
memory budget; the defaults are sized for 8B judges + the base policy on 141 GB cards.

---

## 4. What it does / what to expect

- **Init:** A0/D0 start cold from the abliterated base (served directly; each oracle a fresh LoRA r64).
- **Loop:** alternately train a best-response attacker, then defender, scoring one payoff cell each
  time, for `GENERATIONS=8` iterations (A1/D1 … A8/D8). Reward = `v8-table-r2-seedskip-faithful0.8`.
- **Throughput:** roughly **1–2 h per oracle** on 8× H200; the full 8 generations ≈ most of a day.
- **One oracle = one role best-response**, up to `STEPS_PER_ROLE=600` steps, cut short by early stop:
  - **Attacker early stop** (`ATK_EARLY_STOP=1`): after at least `ATK_MIN_STEP=200` steps, stop when the
    harmful-win rate hasn't improved by `≥ ATK_MIN_DELTA=0.01` over its best for `ATK_PATIENCE=5`
    consecutive windows (plateau floor/ceiling guards `ATK_PLATEAU_FLOOR`/`ATK_CEILING_MAX=0.05`).
  - Early stop waits for the triggering step's adapter to finish writing, then uses **that** checkpoint
    (no "best-on-train" selection). `training/<ORACLE>/` records the step actually trained.
- **Monitor:** `tail -f …/psro/$RUN_TAG/driver.log`. Healthy progress: `next action: {...}` →
  `training A1 …` → `completed action`, repeating. Per-oracle logs under
  `…/psro/$RUN_TAG/training/<ORACLE>/training.log`.

---

## 5. Results, resume and cleanup

Output tree under `…/psro/$RUN_TAG/`:
```text
state.json                                   # durable progress: pools, payoff cells, completed flag
driver.log                                   # top-level driver log
game_dump.jsonl                              # sampled games
matrices/                                    # payoff matrix (meta-game) + Nash mixtures
cells/                                       # per-cell transcripts
training/<ORACLE>/training.log               # e.g. A1, D2 — per-oracle training log
training/<ORACLE>/ckpt/global_step*_hf/      # the trained LoRA adapter (HF/PEFT format)
```
Plus a sibling `…/psro/${RUN_TAG}_raw/` with raw payoff rollouts.

- **Done:** `state.json` has `completed: true`, matrices in `matrices/`, adapters under `training/*/ckpt`.
- **Resume is automatic and idempotent:** `state.json` is the only durable record — every completed cell
  and oracle is **skipped on re-entry**. The entrypoint also wraps the driver in a crash-retry loop
  (`DRIVER_MAX_ATTEMPTS=8`): on any crash it kills stray GPU procs and re-runs, resuming from `state.json`.
  **To resume a run, just launch again with the same `RUN_TAG`, same output mount, same settings.**
- **Do not delete a checkpoint that is still referenced as a historical opponent** in the pool — it
  pollutes later payoff cells. If something errors, read the relevant `training/<ORACLE>/training.log`;
  don't try to "fix" it by wiping the output dir.
- A fresh `RUN_TAG` starts clean. Changing algorithm settings? Use a **new `RUN_TAG`** so you don't mix
  two configs in one state.

**Offline / air-gapped:** on a networked box, run once so `fetch_models.sh` fills the HF cache mount,
then copy that whole cache dir to the target and set `HF_HUB_OFFLINE=1`. The build itself still needs
network (or build on a networked box and `docker save` / `docker load` the image).

---

## 6. Share results back (please do this)

1. **wandb run** — if you passed `WANDB_API_KEY`, the run is online under project
   `coldpsro-singleturn-baseline` (override with `-e WANDB_PROJECT=…`); send us the run/report URL. If it
   ran offline, `wandb sync …/psro/$RUN_TAG/wandb/offline-run-*` then share.
2. **LoRA adapters → HuggingFace** — each oracle's final adapter is at
   `…/psro/$RUN_TAG/training/<ORACLE>/ckpt/global_step*_hf` (HF/PEFT). With `HF_PUSH_REPO` + `HF_TOKEN`
   set they're pushed automatically at the end; otherwise run (inside the container or with the same env)
   `./push_checkpoints.sh $RUN_TAG`. Then share `hf.co/$HF_PUSH_REPO` (it also uploads `state.json` so the
   meta-game is reproducible).

These adapters are the actual trained attacker/defender policies — that's the deliverable we need.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FATAL: no GPUs` | forgot `--gpus all`. |
| `FATAL: … not writable` | the output volume isn't writable — mount a writable host dir. |
| `No available memory for the cache blocks` at step 0 | two judges/engines packed on one card, or a card's rollout `UTIL` too high — see §3; keep the default card map. |
| OOM / Ray or vLLM dies early, `/dev/shm` errors | `--shm-size` too small, or you passed `--ipc=host` (which disables it). Use `--shm-size=32g`, drop `--ipc=host`. |
| `FATAL: base policy not found …abliterated…` | base download failed — check network / that the HF cache mount is writable. |
| A model path "does not exist" after download | HF served a different revision than the pinned one. `fetch_models.sh` pins exact revisions; if a repo was yanked/moved, update the revision there to match the hash in `psro.env`. |
| wandb says it's offline | no `WANDB_API_KEY` — pass it, or `wandb sync` the local run afterward. |
| `rc=1` driver retry in the log | the entrypoint retries + resumes on any crash (up to `DRIVER_MAX_ATTEMPTS=8`). On dedicated GPUs this should be rare; if it loops, read the last `driver.log` error. |
| Slow first step (~3–4 min) | one-time Triton/CUDA kernel compile; later steps are much faster. |

---

## 8. Knobs

Everything is explicit in **`psro.env`** (no inherited defaults). Most likely to touch:
- `GENERATIONS=8` — number of PSRO iterations.
- `STEPS_PER_ROLE=600`, `MATRIX_EPISODES=4096` — training length / payoff-cell size.
- `GPUS=8` and the `ONEJOB_*` card split / judge placement (§3; assume cards 0–7).
- `WANDB_PROJECT`, `HF_PUSH_REPO` — results sharing.
- `DRIVER_MAX_ATTEMPTS` (default 8) — crash-resume retry budget.

The algorithm block (GRPO group size, KL targets, LRs, reward `V8_*` flags, seeds) is copied verbatim
from our live single-turn contract — **leave it as-is** to reproduce the baseline. If you do change
reward/algorithm settings, change them deliberately and use a **new `RUN_TAG`**.

---

## 9. What is / isn't shipped

- **Shipped (in the image):** both conda envs (`srt` = training, `vllm30` = rollout), the code
  (`ssp_v10xm_g2`, `work_v9zp_g2` — the exact single-turn pipeline validated on gpu2),
  `role_phase_v9sb_flame.sh`, and the SFT prompt data. All at the pinned absolute layout. **No API keys
  are baked.**
- **Pulled on first run (all public, off-the-shelf):** the abliterated Llama-3.1-8B (base **and**
  topic-judge base), WildGuard (harm judge), Qwen3.5-9B (faithful-rewrite judge) — each at a pinned
  revision.
- **No checkpoints of ours** are baked, shipped or required. The run trains from scratch on the open
  base. You supply the 8× H200 and (to share back) wandb + HF credentials.

**Divergence from our internal run:** our gpu2 baseline used an SFT'd base (`pi0`, served via
`ONEJOB_SHARED_BASE=1` = abliterated + a pi0 LoRA). To keep this fully off-the-shelf, the base here is
the **raw abliterated Llama** served directly (`ONEJOB_SHARED_BASE=0`). The pipeline code is identical
(`ssp_v10xm_g2` / `work_v9zp_g2`); only the base weights differ. Expect the attacker to be *weaker* than
our internal run (which itself plateaued after a couple of generations) — this is a clean open baseline,
not a bit-for-bit reproduction.

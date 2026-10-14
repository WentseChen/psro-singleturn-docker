# Cold PSRO (single-turn) — Docker handoff

Attacker/defender self-play safety RL (Self-RedTeam style), **single-turn**, **8 PSRO generations**,
on a single **8× H200** node. One driver process spawns everything inside the container — Ray, the
per-card vLLM rollout servers, the three judges (WildGuard / topic / faithful), and the GRPO
training phase. **No Slurm**, no multi-node, no external orchestration.

**Everything is off-the-shelf and open-source — no checkpoints of ours are used or shipped.** You
provide only the **GPUs**. The base policy is the public abliterated Llama-3.1-8B (pulled from
HuggingFace along with the judges on first start); cold PSRO trains from generation 0 on top of it,
each oracle a fresh LoRA rank-64 adapter.

Please **share two things back with us** when done (details in §4): the **wandb run** and the
**LoRA adapters pushed to a HuggingFace repo**.

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

## 0. Prerequisites

- A single machine with **8× H200** (≈140 GB each), NVIDIA driver + `nvidia-container-toolkit`
  (so `docker run --gpus all` works).
- **~70 GB** free disk for the image, plus space for the output volume and the HF model cache
  (~55 GB: 3 judges + the abliterated base).
- Network access on first run to pull the public models (or pre-populate the HF cache volume and
  run offline).
- For sharing results back: a **wandb** account (API key) and a **HuggingFace** repo + write token.

---

## 1. Build (two steps)

The image embeds two conda envs that are **path-pinned** (their shebangs/RPATHs hardcode absolute
paths), and the `vllm30` rollout env is **node-local** — so the build context must be produced on
a GPU node, then the image is built from it.

**Step 1 — stage the context (on the GPU box that has the envs):**
```
sbatch stage.sbatch      # or just run it on the GPU node
```
This fills `ctx/` with tarballs of the two envs + the code + SFT data (~10 GB). It's already been
run here, so `ctx/` is populated — you only need to redo this if you rebuild the envs.

**Step 2 — build the image (on your Docker host):**
Copy this whole directory (including `ctx/`) to the Docker host, then:
```
docker build -t psro:single-turn .
```
The image recreates the pinned absolute paths (`/scratch/wentsec/envs/vllm30`,
`/zfsauton/scratch/wentsec/...`). **Do not try to relocate them** — the envs will break.

---

## 2. Run
```
docker run --rm --gpus all --shm-size=32g \
  -e RUN_TAG=run1 \
  -e WANDB_API_KEY=xxxxxxxx  -e WANDB_ENTITY=your_wandb_entity \
  -e HF_PUSH_REPO=your-org/coldpsro-singleturn  -e HF_TOKEN=hf_xxxxxxxx \
  -v /ABS/path/to/psro_out:/zfsauton/scratch/wentsec/coldpsro_slurm/psro \
  -v /ABS/path/to/hf_cache:/zfsauton/scratch/wentsec/hf_srt \
  psro:single-turn
```

Mounts:
1. **output (required)** → `/zfsauton/scratch/wentsec/coldpsro_slurm/psro`. Results land in
   `…/psro/$RUN_TAG/`; mount a host dir so they survive the container.
2. **HF cache (recommended)** → `/zfsauton/scratch/wentsec/hf_srt`. Models download here on first
   start; mounting it means the ~55 GB download happens once and is reused across runs.

Env (all optional except nothing is required to *run*; the two sharing ones are how you send results back):
- `RUN_TAG` — names the run; fresh tag = clean run, reuse = resume from last checkpoint.
- `WANDB_API_KEY` (+ optional `WANDB_ENTITY`) — logs training metrics to wandb. **Without it the run
  still works** but wandb goes `offline` (local files under `…/psro/$RUN_TAG/wandb`, which you can
  `wandb sync` later).
- `HF_PUSH_REPO` (+ `HF_TOKEN`) — at the end, the LoRA adapters are auto-pushed there to share back.
  Without them the run still works; you can push later with `./push_checkpoints.sh $RUN_TAG`.

Notes:
- **`--shm-size=32g` is required** (Ray/vLLM IPC + LoRA staging use `/dev/shm`). Do **not** also
  pass `--ipc=host` — that makes Docker ignore `--shm-size` and fall back to the host's `/dev/shm`,
  which is often tiny and will OOM mid-run. If 32 GB turns out tight, raise it; don't switch to
  `--ipc=host`.
- First start runs `fetch_models.sh` (pulls the base + 3 judges at pinned revisions), then the driver.

---

## 3. What it does / what to expect

- **Init:** A0/D0 start cold from the abliterated Llama base (served directly; each oracle is a
  fresh LoRA rank-64 adapter).
- **Loop:** alternately train a best-response attacker, then defender, scoring one payoff cell each
  time, for `GENERATIONS=8` iterations. Reward = `v8-table-r2-seedskip-faithful0.8`.
- **Throughput:** roughly **1–2 h per oracle** on 8× H200; the full 8 generations ≈ most of a day.
- **Monitor:** `tail -f …/psro/$RUN_TAG/driver.log`. Healthy progress shows
  `next action: {...}` → `training A1 …` → `completed action`, repeating. Per-oracle training logs
  are under `…/psro/$RUN_TAG/training/<ORACLE>/training.log`.
- **Done:** `…/psro/$RUN_TAG/state.json` has `completed: true`, with the payoff matrix in
  `matrices/`, per-cell transcripts in `cells/`, and sampled games in `game_dump.jsonl`.

---

## 4. Share results back (please do this)

1. **wandb run** — if you passed `WANDB_API_KEY`, the run is already online under project
   `coldpsro-singleturn-baseline` (override with `-e WANDB_PROJECT=…`); send us the run/report URL.
   If it ran offline, `wandb sync …/psro/$RUN_TAG/wandb/offline-run-*` then share.
2. **LoRA adapters → HuggingFace** — each oracle's final adapter is at
   `…/psro/$RUN_TAG/training/<ORACLE>/ckpt/global_step*_hf` (HF/PEFT format). With `HF_PUSH_REPO` +
   `HF_TOKEN` set, they're pushed automatically at the end; otherwise run inside the container (or
   with the same env) `./push_checkpoints.sh $RUN_TAG`. Then share the repo `hf.co/$HF_PUSH_REPO`
   (it also uploads `state.json` so the meta-game is reproducible).

These adapters are the actual trained attacker/defender policies — that's the deliverable we need.

---

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FATAL: no GPUs` | forgot `--gpus all`. |
| `FATAL: … not writable` | the output volume isn't writable — mount a writable host dir. |
| OOM / Ray or vLLM dies early, `/dev/shm` errors | `--shm-size` too small, or you passed `--ipc=host` (which disables it). Use `--shm-size=32g` and drop `--ipc=host`. |
| `FATAL: base policy not found …abliterated…` | the base download failed — check network / the HF cache mount is writable. |
| A model path "does not exist" after download | HF served a different revision than the pinned one. `fetch_models.sh` pins exact revisions; if a repo was yanked/moved, update the revision there to match the hash in `psro.env`. |
| wandb says it's offline | no `WANDB_API_KEY` — pass it, or `wandb sync` the local run afterward. |
| `rc=1` driver retry in the log | the entrypoint retries + resumes on any crash (up to `DRIVER_MAX_ATTEMPTS=8`). On dedicated GPUs this should be rare; if it loops, read the last `driver.log` error. |
| Slow first step (~3–4 min) | one-time Triton/CUDA kernel compile; later steps are much faster. |

---

## 6. Knobs

Everything is explicit in **`psro.env`** (no inherited defaults). The ones you're most likely to
touch:
- `GENERATIONS=8` — number of PSRO iterations.
- `STEPS_PER_ROLE=600`, `MATRIX_EPISODES=4096` — training length / payoff-cell size.
- `GPUS=8` and the `ONEJOB_*` card split / judge placement (assume 8 cards indexed 0–7).
- `WANDB_PROJECT`, `HF_PUSH_REPO` — results sharing.
- `DRIVER_MAX_ATTEMPTS` (default 8) — crash-resume retry budget.

The algorithm block (GRPO group size, KL targets, LRs, reward `V8_*` flags, seeds) is copied
verbatim from our live single-turn contract — **leave it as-is** to reproduce the baseline.

---

## 7. What is / isn't shipped

- **Shipped (in the image):** both conda envs (`srt` = training, `vllm30` = rollout), the code
  (`ssp_v10xm_g2`, `work_v9zp_g2` — the exact single-turn pipeline we validated on gpu2),
  `role_phase_v9sb_flame.sh`, and the SFT prompt data. All paths are the pinned absolute layout.
- **Pulled on first run (all public, off-the-shelf):** the abliterated Llama-3.1-8B (base **and**
  topic-judge base), WildGuard (harm judge), Qwen3.5-9B (faithful-rewrite judge).
- **No checkpoints of ours** are baked, shipped, or required. The run trains from scratch on the
  open base. You supply the 8× H200 and (to share back) wandb + HF credentials.

**Note on divergence from our internal run:** our gpu2 baseline used an SFT'd base (`pi0`, served
via `ONEJOB_SHARED_BASE=1` = abliterated + a pi0 LoRA). To keep this fully off-the-shelf, the base
here is the **raw abliterated Llama** served directly (`ONEJOB_SHARED_BASE=0`). The pipeline code is
identical (`ssp_v10xm_g2` / `work_v9zp_g2`); only the base weights differ. Expect the attacker to be
*weaker* than our internal run (which itself plateaued after a couple of generations) — this is a
clean open baseline, not a bit-for-bit reproduction.

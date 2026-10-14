# PSRO invalid-move run: what changed and why (worklog)

Branch `wentse/v6d-invalid-move-reward` (released code, images v2/v3) and
`wentse/v6d-invalid-move-reward-dev` (this file + unreleased speed work).
Last updated 2026-09-24.

## 1. The bug: remote PSRO stalled after iteration 1

- Symptom (W&B, remote 4xH200): A1 trained, then the A1xD0 payoff cell never finished.
  34/2000 benign episodes after 9,856 candidates (cap 16,000).
- Cause: `GAME_REWARD_MODE=shared_binary` dropped drift games from the reward, so the
  attacker learned to rewrite every *benign* seed into a harmful prompt at no cost.
  The cell keeps only benign episodes that stay benign, so the benign stratum could
  never fill. Naive self-play has the same reward but no payoff cell, so it didn't stall.

## 2. Reward fix: flat invalid-move floor (`ATTACKER_INVALID_MOVE=1`, commit ec69903)

On top of the v6d legacy (Self-RedTeam v2) reward:

| attacker move | reward |
| --- | --- |
| valid (label matches seed type, on topic) | unchanged v6d reward |
| one violation (drift either way, or off-topic) | -3 |
| two violations (drift + off-topic) | -4 |
| + CoT format violation | extra -2 |
| judge label missing / unclassifiable | min(r, 0) |

- Checked: 216-case sweep (`tests/test_invalid_move_reward.py`); no move beats a valid
  one, and drift is never cheaper than an honest failure.
- Payoff matrix does the same: invalid move = -1 (the bound; worst valid is -0.75).
  Matrix version `role-lora-psro-terminal-projection-v4+invalid-move-v1` (2eeb331).
- Locally tested (2xH200, 200 steps): benign count stable (34 -> 34), attacker success
  0.33 -> 0.96, invalid moves 0.38 -> 0.02; the cell filled with 1.8% drift.

## 3. Robustness fixes

| commit | fix |
| --- | --- |
| 9ba1751 | a locked topic-label cache counts as a miss instead of failing the step |
| e8817d4 | invalid-move reward reaches Docker runs and the matrix |
| d261fae | W&B upload failures no longer kill training; topic judge for Slurm cells |
| 3ef76c6 | Slurm path sets `TOPIC_API_*` for the run-scoped topic judge (KeyError fix) |
| 1aaa113 (image v3) | cell judge frees GPU 0 before training (SIGTERM, then SIGKILL the group, then wait until >=85% free); driver retries up to 5x. Fixes seed 8889 stuck at `training_A1` |
| d3baab2 (image v4) | same hole on the training side: `role_phase.sh` now kills its WildGuard judge's Ray actor (by `/tmp/ssp-judge-PORT`); the driver kills training stragglers after every phase (early stop included) and confirms all 4 cards are free after training and before it. Streaming engines die with their parent (PDEATHSIG) |

## 4. Defaults and Docker

- Image defaults: `GAME_REWARD_MODE=legacy ATTACKER_INVALID_MOVE=1 RACE_CELLS=1`.
  Asking for `shared_binary` together with the invalid-move flag is an error.
- Racing: 256-episode pilot per cell; only Nash-relevant rows/columns go to 4000.
  The newest attacker/defender cells always get the full budget.
- Images: v3 `…-invalidmove-20260923-v3` (commit 1aaa113); **v4
  `ghcr.io/wentsechen/safeselfplay:llama8b-h200-invalidmove-20260924-v4`** for the full rerun
  (digest `sha256:dcd49b1b…`, commit b815827; PSRO s8888/s8889, uniform, naive; new suffixes `*_inv4_*`). Runbook: `docker/RUN_PSRO_INVALIDMOVE.md`.
- v4 ships the streaming-cell code with `CELL_STREAMING=0`; switch on after the H200 timing.
- GPU layout stays 4xH200 in the repo; 1-2 GPU test configs were only in local sbatch files.

## 5. GPU utilisation of payoff cells (dev branch, not in any image yet)

Measured on gpu2 during a cell (old code): the cell GPU was at 22% util and the other
three cards were idle. A 4000-episode cell took 73 min.

Causes:
1. `_generate` called vLLM in 64-prompt chunks, so each chunk waited for its longest
   reply (up to 2048 tokens; defender average ~750).
2. Attacker generation, then defender generation, then judging ran in lockstep for
   the whole wave, so the judge GPU idled during generation and vice versa.
3. 64-pair waves (dozens per cell), eager mode (no CUDA graphs), one GPU per cell.

The only barrier the code needs is the wave boundary: valid episodes (after drift/parse
drops) are kept in `candidate_index` order, so the kept set does not favour short replies.

Fix (commits 702b310, d899c6c; `roll/utils/stream_engine_pool.py`):
- `EnginePool`: one vLLM engine process per GPU, continuous batching, a constant number
  of requests in flight (`CELL_MAX_NUM_SEQS`, default 256), CUDA graphs on.
- `_stream_wave`: each episode moves on independently. Its defence is submitted the
  moment its attack finishes, and it goes to WildGuard/topic judge threads as soon as
  a judge batch is full.
- Waves are sized from deficit / observed retention (`CELL_STREAM_WAVE_PAIRS`, default
  1024), so a cell is 1-2 waves. The candidate list is fixed by index, so the retained
  prefix is the same estimand.
- Driver: a single pending cell gets all worker GPUs, plus the spare memory on the
  judge cards (`CELL_SHARE_JUDGE_GPUS=1`). The pool sizes each engine to what is free
  and skips a card with <30% usable.
- Safety: if an engine fails, the cell falls back to the old batched path.
  `CELL_STREAMING=0` turns streaming off.
- Seeds: every request keeps its own seed. Outputs match the old path in distribution,
  not token for token (batch numerics differ; also true of the old code).
- Tests: `tests/test_stream_wave.py` (pipeline order, cache hits, topic filter);
  2x2080Ti smoke test with LoRA: 2 engines + graphs 9.6 s vs 327 s for 1 eager engine
  at concurrency 4 on the same 160 requests.
- Still to do: time a full cell on 4xH200 against the 73 min baseline.

## 6. Job chaining (local Auton runs)

`psro_full8` runs as a Slurm chain 62428 -> 62429 -> 62430 (`afterany`). Each job
resumes from `state.json`, so the node stays busy without resubmitting by hand.

## 7. Open items

- Remote s8888 (v2 image) was not stuck: its A1xD0 cell took ~2h15m (vs 73 min locally), A1 payoff
  +0.118 vs A0's -0.475, and D1 started 02:06 EDT. Remote cells are slow, which is what the streaming work targets.
- Stale test fixture (`InterleavedEpisodePlanTest` pools of 5/7 prompts vs no-replacement selection) enlarged; the
  production benign pool (20k) exceeds the candidate cap's max benign draws (8k).
- Put the streaming cells into an image (v4) after the H200 timing.
- Baselines (naive/uniform) with the new reward; step-budget parity; seeds.

#!/usr/bin/env python3
"""Slurm-native driver for the cold role-LoRA zero-sum PSRO continuation.

On Modal the controller re-detaches itself after every durable action
(``advance_cold_psro.spawn``).  Here the same function is called in an ordinary
loop: ``spawn`` is a no-op (see cold_psro_common), so one call advances exactly
one cell / oracle and returns, and this process decides what runs next.

Two things Modal provided as separate containers are provided here instead:

* the WildGuard judge, run on GPU 0 for the payoff-cell phases and stopped
  before a training phase, because scripts/slurm/role_phase.sh starts its own
  judge alongside the four-GPU trainer;
* the payoff cells themselves, which are prefetched three at a time on GPUs 1-3
  before re-entering the coordinator.  The coordinator then finds each raw cell
  already complete and finishes it on CPU, so the schedule it walks is
  unchanged -- only the wall-clock is.

The whole loop is idempotent: state.json is the only durable progress record,
every completed cell and oracle is skipped on re-entry, and an interrupted
oracle restarts from the base adapter exactly as it would on Modal (the run
contract pins save_steps to the final step, so there is no mid-role checkpoint).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cold_psro_common as common  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from roll.utils.shared_game_reward import shared_game_reward_enabled, shared_game_reward_contract, SHARED_GAME_REWARD_VERSION
from roll.utils.topic_api import configure_local_topic_backend




SLURM_DIR = Path(__file__).resolve().parent
# env knobs a role may override with a _ATTACKER/_DEFENDER suffix
PER_ROLE_KNOBS = frozenset({
    "ROLLOUT_BATCH", "TRAIN_BATCH", "GROUP_SIZE", "MICRO_TRAIN", "MICRO_ROLLOUT",
    "ASYNC_DEPTH", "ASYNC_STALE_K", "ASYNC_TAKE", "ASYNC_JUDGE_WORKERS", "ASYNC_JUDGE_CHUNK",
    "ADAPTIVE_EPOCHS", "ADAPTIVE_EPOCHS_MAX", "INIT_KL_COEF", "GENERATE_MAX_LEN",
    "LR_WARMUP_RATIO", "SAVE_STEPS", "CELL_SERVER_INFLIGHT",
    "USE_LIGER",  # 10-11: Liger helps the attacker trainer (+12% tok/s), not the defender
})
ROLE_PHASE = Path(os.environ.get("ROLE_PHASE_SCRIPT") or (SLURM_DIR / "role_phase.sh"))
# ONEJOB=1: long-lived judges + vLLM servers shared by cells and training
ONEJOB = os.environ.get("ONEJOB", "0") == "1"
SERVICES = None  # onejob_services.OneJobServices when ONEJOB
CELL_WORKER = SLURM_DIR / "payoff_cell_worker.py"


def log(message: str) -> None:
    print(f"[driver {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


# --- judge -------------------------------------------------------------------


class Judge:
    """WildGuard on GPU 0, shared by every concurrent payoff-cell worker."""

    def __init__(self, cfg: "Config") -> None:
        self.cfg = cfg
        self.process: subprocess.Popen[bytes] | None = None
        self.url = f"http://127.0.0.1:{cfg.cell_judge_port}/classify"
        self.gpu = "0"
        self.ray_tmpdir = f"{os.environ.get('SSP_TMP', '/tmp')}/coldpsro-judge-{cfg.cell_judge_port}"

    def _healthy(self) -> bool:
        payload = json.dumps(
            {
                "queries": [{"prompt": "hi", "response": "hello", "game_idx": 0}],
                "prompts": None,
            }
        ).encode()
        request = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return b"labels" in response.read()
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def start(self) -> str:
        if self.process is not None and self.process.poll() is None:
            return self.url
        self.cfg.judge_log.parent.mkdir(parents=True, exist_ok=True)
        environment = self.cfg.base_env()
        environment["CUDA_VISIBLE_DEVICES"] = self.gpu
        environment["RAY_TMPDIR"] = self.ray_tmpdir
        command = [
            self.cfg.python, "-m", "openrlhf.cli.serve_wildguard",
            "--model_path", self.cfg.judge_model,
            "--port", str(self.cfg.cell_judge_port),
            "--host", "127.0.0.1",
            "--bf16", "--num_gpus", "1", "--tensor_parallel_size", "1",
            "--enforce_eager", "--seed", "42",
            # 8192 is the upstream value; a long transcript returns 500 at 4096
            # and takes the whole server down mid-cell.
            "--max_len", "8192",
            "--max_num_seqs", "64", "--batch_size", "64",
            "--gpu_memory_utilization", str(self.cfg.cell_judge_util),
        ]
        log(f"starting cell judge on GPU 0 -> {self.url}")
        handle = self.cfg.judge_log.open("ab")
        self.process = subprocess.Popen(
            command, cwd=str(self.cfg.work), env=environment,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
        for attempt in range(180):
            if self._healthy():
                log(f"cell judge up after ~{attempt * 10}s")
                return self.url
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"cell judge exited rc={self.process.returncode}; "
                    f"see {self.cfg.judge_log}"
                )
            time.sleep(10)
        raise RuntimeError("cell judge did not become healthy within 30 minutes")

    def stop(self) -> None:
        if self.process is None:
            return
        log("stopping cell judge")
        try:
            pgid = os.getpgid(self.process.pid)
        except ProcessLookupError:
            pgid = None
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            self.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            pass
        # The judge's model lives in a Ray actor, not in the process waited on
        # above, so the leader can exit while the actor still holds GPU 0. The
        # next training phase then puts its GPU-0 engine on a card that is still
        # ~0.45 occupied and dies with "No available memory for the cache
        # blocks" before W&B starts (seed 8889, 2026-09-23). Kill the whole
        # group unconditionally, then anything of this judge's private Ray
        # cluster that escaped the group, and wait for the card to be free.
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        _kill_processes_marked(self.ray_tmpdir)
        self.process = None
        _wait_for_free_gpu(self.cfg.python, self.gpu, log)


def _kill_processes_marked(marker: str) -> list[int]:
    """SIGKILL this user's processes whose command line or environment names ``marker``.

    Ray starts its daemons and actor workers for a private cluster under the
    driver's RAY_TMPDIR, and some of them leave the driver's process group, so a
    group kill alone can miss the actor that holds the GPU.
    """
    me = {os.getpid(), os.getppid()}
    uid = os.getuid()
    needle = marker.encode()
    killed = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in me:
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            blob = (entry / "cmdline").read_bytes() + b"\0" + (entry / "environ").read_bytes()
        except OSError:
            continue
        if needle in blob:
            try:
                os.kill(int(entry.name), signal.SIGKILL)
                killed.append(int(entry.name))
            except (ProcessLookupError, PermissionError):
                pass
    if killed:
        log(f"  killed {len(killed)} leftover process(es) of {marker}")
    return killed


def _gpu_free_fraction(python: str, gpu: str) -> float:
    probe = ("import torch; f, t = torch.cuda.mem_get_info(0); print(f / t)")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    out = subprocess.run([python, "-c", probe], env=env, capture_output=True,
                         text=True, timeout=120)
    return float(out.stdout.strip().splitlines()[-1])


def _wait_for_free_gpu(python: str, gpu: str, log=log, need: float | None = None) -> bool:
    """Block until a card is actually free, instead of a fixed sleep."""
    if need is None:
        need = float(os.environ.get("CELL_JUDGE_GPU_FREE_FRACTION", "0.85"))
    deadline = time.time() + float(os.environ.get("CELL_JUDGE_RELEASE_TIMEOUT", "300"))
    free = 0.0
    while True:
        try:
            free = _gpu_free_fraction(python, gpu)
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            free = 0.0
        if free >= need:
            log(f"  GPU {gpu} released ({free:.0%} free)")
            return True
        if time.time() >= deadline:
            log(f"  WARNING: GPU {gpu} still only {free:.0%} free after its previous tenant "
                f"stopped; the next phase may fail to allocate and will be retried")
            return False
        time.sleep(10)


def _release_training_gpus(cfg: "Config") -> None:
    """Leave no training process on any card before the judge and cells start.

    role_phase.sh stops its Ray cluster and judges on exit, but its WildGuard
    judge runs a private Ray cluster whose vLLM actor sits outside the judge's
    process group, and an early stop can SIGKILL the phase before its trap
    finishes. Either way a stray engine keeps its memory, and the next vLLM on
    that card (the cell judge on GPU 0, a cell on 1-2) cannot allocate. Kill by
    temp-dir marker, then confirm every card is free.
    """
    for marker in (f"{os.environ.get('SSP_TMP', '/tmp')}/ssp-ray-{cfg.ray_port}", f"{os.environ.get('SSP_TMP', '/tmp')}/ssp-judge-{cfg.train_judge_port}"):
        _kill_processes_marked(marker)
    if ONEJOB:
        return  # the persistent servers hold their cards on purpose
    _confirm_gpus_free(cfg)


def _confirm_gpus_free(cfg: "Config") -> None:
    """Wait until every card is free of the previous phase (bounded, non-fatal)."""
    topic_gpu = None
    if os.environ.get("TOPIC_API_BASE_URL", "").startswith("http://127.0.0.1"):
        # The run-scoped topic judge stays up between phases (~13% of a card).
        topic_gpu = os.environ.get("TOPIC_JUDGE_GPU", str(max(cfg.gpus - 1, 0)))
    for gpu in range(cfg.gpus):
        _wait_for_free_gpu(cfg.python, str(gpu), log,
                           need=0.70 if str(gpu) == topic_gpu else None)


# --- configuration ------------------------------------------------------------


class Config:
    def __init__(self, args: argparse.Namespace) -> None:
        self.workspace: Path = args.workspace.resolve()
        self.work: Path = args.work.resolve()
        self.repo: Path = SLURM_DIR.parents[1]
        self.hf_home: Path = args.hf_home.resolve()
        self.sft_dir: Path = args.sft_dir.resolve()
        self.python: str = args.python
        self.suffix: str = args.suffix
        self.psro_root: Path = args.psro_root.resolve()
        self.raw_payoff_root: Path = args.raw_payoff_root.resolve()
        self.run_root: Path = self.psro_root / self.suffix
        self.gpus: int = args.gpus
        self.cell_workers: int = args.cell_workers
        self.cell_judge_port: int = args.cell_judge_port
        self.cell_judge_util: float = args.cell_judge_util
        self.train_judge_port: int = args.train_judge_port
        self.ray_port: int = args.ray_port
        self.judge_log: Path = self.run_root / "logs" / "cell_judge.log"
        self.cell_log_dir: Path = self.run_root / "logs" / "cells"
        self.judge_model: str = str(args.judge_model)
        self.coordinator: str = args.coordinator
        self.init_mode: str = args.init_mode
        self.stop_before_evaluation: bool = not args.run_official_evaluation

    def apply_to_process(self) -> None:
        """Put the child environment on this process too.

        The coordinator evaluates a cell in-process whenever a prefetch worker
        is missing, and the Slurm job inherits the login profile's HF_HOME and
        TRANSFORMERS_CACHE, which point at a different, incomplete cache.
        """

        os.environ.pop("TRANSFORMERS_CACHE", None)
        os.environ.update(self.base_env())

    def base_env(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment.pop("TRANSFORMERS_CACHE", None)
        environment.update(
            {
                "HF_HOME": str(self.hf_home),
                "HF_HUB_CACHE": str(self.hf_home / "hub"),
                "HF_HUB_DISABLE_XET": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "NCCL_CUMEM_ENABLE": "0",
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                # Console scripts (ray, deepspeed) must come from the same
                # environment as --python; a stale ~/.local/bin copy shadows
                # them otherwise.
                "PATH": ":".join(
                    part for part in (
                        str(Path(self.python).parent),
                        os.environ.get("PATH", ""),
                    ) if part
                ),
                "PYTHONPATH": ":".join(
                    part for part in (
                        str(self.work), str(self.repo),
                        os.environ.get("PYTHONPATH", ""),
                    ) if part
                ),
            }
        )
        return environment


# --- local oracle trainer ------------------------------------------------------


def _build_criterion(role: str):
    """Early-stopping criterion for one oracle, or None to run the full budget.

    Defenders use the absolute rule: their own objective saturates, so stop once
    the attacker's success rate stays low, guarded by benign compliance and a
    CoT-format ceiling (a degenerate defender drives attack success to zero by
    emitting unparseable text, which looks identical to a win).

    Attackers use a plateau rule instead. No attacker in either arm exceeded
    ~0.57 success against a trained defender, so an absolute threshold would
    never fire and every attacker would burn the whole budget; stopping on "no
    improvement for `patience` windows" is the rule that actually binds.
    """
    if os.environ.get("LADDER_EARLY_STOP", "1") == "0":
        return None
    from early_stop_monitor import EarlyStop

    window = int(os.environ.get("EARLY_STOP_WINDOW", "20"))
    if role == "attacker":
        if os.environ.get("ATK_EARLY_STOP", "1") == "0":
            return None  # attackers run the full budget; only defenders stop at 100% pass
        return EarlyStop(
            metric="attacker/info/successful_harmful_request_rate",
            guard_metric="",
            ceiling_metric="attacker/cot_format_violation",
            ceiling_max=float(os.environ.get("ATK_CEILING_MAX", "0.30")),
            abort_max=0.80, abort_patience=2,
            mode="plateau",
            min_delta=float(os.environ.get("ATK_MIN_DELTA", "0.01")),
            window=window,
            patience=int(os.environ.get("ATK_PATIENCE", "3")),
            min_step=int(os.environ.get("ATK_MIN_STEP", "60")),
        )
    return EarlyStop(
        metric="attacker/info/successful_harmful_request_rate",
        guard_metric="defender/wildguard_actual_benign_compliance_acc",
        threshold=float(os.environ.get("DEF_THRESHOLD", "0.01")),
        guard_min=float(os.environ.get("DEF_GUARD_MIN", "0.912")),
        ceiling_metric="defender/cot_format_violation",
        ceiling_max=0.10, abort_max=0.80, abort_patience=2,
        window=window,
        patience=int(os.environ.get("DEF_PATIENCE", "3")),
        min_step=int(os.environ.get("DEF_MIN_STEP", "60")),
    )


def early_stop_configuration() -> dict[str, Any]:
    return {role: criterion.configuration() if criterion else None
            for role in ("attacker", "defender")
            for criterion in [_build_criterion(role)]}


def _checkpoint_ready(path: Path) -> bool:
    """Wait for complete adapter contents, not just mkdir during a save."""
    try:
        json.loads((path / "adapter_config.json").read_text())
        weights = path / "adapter_model.safetensors"
        if weights.is_file():
            from safetensors import safe_open
            with safe_open(str(weights), framework="numpy") as adapter:
                return bool(list(adapter.keys()))
        import zipfile
        with zipfile.ZipFile(path / "adapter_model.bin") as archive:
            return bool(archive.namelist()) and archive.testzip() is None
    except Exception:
        # safetensors raises its own extension exception for incomplete writes.
        return False


def _terminate_training(process) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=180)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


def _supervise_training(environment, training_log: Path, criterion, ckpt_dir: Path):
    """Run role_phase.sh, tail its log, and stop the run when the criterion fires.

    The trainer is started in its own process group so the whole tree can be
    signalled; on a trigger we wait for the checkpoint at the stopping step to
    appear before terminating, so the run always leaves a usable adapter.
    """
    import signal
    from early_stop_monitor import metric_re, parse_line

    names = [criterion.metric]
    if criterion.guard_metric:
        names.append(criterion.guard_metric)
    if criterion.ceiling_metric:
        names.append(criterion.ceiling_metric)
    patterns = {n: metric_re(n) for n in names}

    handle = training_log.open("ab")
    process = subprocess.Popen(
        ["bash", str(ROLE_PHASE)], cwd=str(environment.get("SSP_WORK", ".")),
        env=environment, stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    offset, remainder, stop_step = 0, "", None
    try:
        # stop_step in the condition is load-bearing: without it the loop runs
        # to the trainer's natural exit and the terminate branch below never
        # gets to cut the run short.
        while process.poll() is None and stop_step is None:
            time.sleep(5)
            if not training_log.is_file():
                continue
            with training_log.open("r", errors="ignore") as reader:
                reader.seek(offset)
                chunk = reader.read()
                offset = reader.tell()
            if not chunk:
                continue
            parts = re.split(r"[\r\n]+", remainder + chunk)
            remainder = parts.pop() if parts else ""
            for line in parts:
                got = parse_line(line, patterns)
                if got is None:
                    continue
                step, values = got
                criterion.observe(step, values)
                if (criterion.triggered or criterion.collapsed) and stop_step is None:
                    stop_step = step
                    break
        if stop_step is None:
            return process.wait(), "ran the full budget"
        # let the checkpoint at the stopping step land before killing the tree
        deadline = time.time() + float(os.environ.get("EARLY_STOP_CKPT_WAIT", "900"))
        target = ckpt_dir / f"global_step{stop_step}_hf"
        while time.time() < deadline and not _checkpoint_ready(target) and process.poll() is None:
            time.sleep(10)
        if not _checkpoint_ready(target):
            _terminate_training(process)
            return 1, f"early stop at step {stop_step}, but checkpoint is incomplete: {target}"
        if process.poll() not in {None, 0}:
            return process.returncode, f"trainer failed while saving step {stop_step}"
        _terminate_training(process)
        kind = "COLLAPSE" if criterion.collapsed else "converged"
        return (2 if criterion.collapsed else 0), f"{kind} at step {stop_step} ({criterion.reason or criterion.mode})"
    finally:
        _terminate_training(process)
        handle.close()


def make_local_oracle_trainer(psro: Any, cfg: Config):
    """Replace the Modal oracle function with a role_phase.sh invocation.

    Everything the coordinator relies on is preserved: the pool is normalized
    (which re-verifies every frozen opponent's weight SHA), an already-finished
    target is returned without retraining, and the manifest carries the three
    fields the idempotence check compares.
    """

    def train_cold_psro_oracle(
        *,
        run_root: str,
        target: str,
        role: str,
        opponent_pool: list[dict[str, object]],
        steps: int,
        lora_rank: int,
        lora_alpha: int,
        learning_rate: float,
        sft_stop_after_step: int,
        sft_batches_per_step: int,
        save_steps: int,
        actor_lr_scheduler: str,
        lr_warmup_ratio: float,
        training_seed: int,
        wandb_identity: str,
        cold_start: bool | None = None,
        role_start_label: str | None = None,
        role_start_adapter: str | None = None,
        role_start_sha256: str | None = None,
    ) -> dict[str, Any]:
        expected_prefix = "A" if role == "attacker" else "D"
        if not re.fullmatch(rf"{expected_prefix}[1-9][0-9]*", target):
            raise ValueError(f"invalid cold PSRO target: {target!r}")
        root = Path(run_root)
        if root == psro.PSRO_OUTPUT_ROOT or psro.PSRO_OUTPUT_ROOT not in root.parents:
            raise ValueError("run_root must be below the PSRO output root")
        normalized_pool = psro._normalize_fixed_opponent_pool(opponent_pool)
        run_dir = root / "training" / target
        checkpoint = run_dir / "ckpt" / f"global_step{steps}_hf"
        manifest_path = run_dir / "manifest.json"

        # Cold start: every oracle begins at the base model with a fresh
        # adapter. Hot start changes exactly this one thing -- the oracle
        # continues the previous same-role adapter, and only the first
        # generation of each role starts from base. The opponent rule, the
        # payoff matrix and the meta-game solve are untouched.
        requested_start = role_start_adapter
        role_start_adapter = None
        generation = int(target[1:])
        if cfg.init_mode == "hot" and generation > 1:
            parent = f"{target[0]}{generation - 1}"
            state = psro._read_json(root / "state.json")
            parent_record = state["population"].get(parent)
            if not parent_record or not parent_record.get("path"):
                raise RuntimeError(f"hot start for {target} needs {parent}")
            role_start_adapter = str(parent_record["path"])
            if not psro._adapter_checkpoint(Path(role_start_adapter)):
                raise FileNotFoundError(
                    f"missing hot-start adapter for {target}: {role_start_adapter}"
                )
        expected_role_start = role_start_adapter or psro.BASE_MODEL
        # The current coordinator supplies explicit initialization metadata;
        # older local/hot callers derive it from cfg. Keep both call forms.
        if cold_start is not None:
            expected_cold = cfg.init_mode == "cold"
            expected_label = f"{target[0]}{0 if expected_cold else generation - 1}"
            if cold_start != expected_cold or role_start_label != expected_label:
                raise RuntimeError(f"{target}: coordinator/local initialization mismatch")
            if requested_start != role_start_adapter:
                raise RuntimeError(f"{target}: coordinator/local start adapter mismatch")
            actual_sha = (
                psro._adapter_identity(Path(role_start_adapter))["adapter_sha256"]
                if role_start_adapter else f"base:{psro.BASE_MODEL}"
            )
            if role_start_sha256 != actual_sha:
                raise RuntimeError(f"{target}: role-start checkpoint identity changed")

        # An early-stopped oracle has a checkpoint below the step budget.
        # Reuse its completed manifest if the coordinator was interrupted
        # between the manifest write and population update.
        if manifest_path.is_file():
            manifest = psro._read_json(manifest_path)
            if (
                manifest.get("role_start") != expected_role_start
                or manifest.get("fixed_opponent_pool") != normalized_pool
                or manifest.get("steps") != steps
            ):
                raise RuntimeError(f"existing {target} checkpoint contract changed")
            recorded = Path(manifest.get("checkpoint", str(checkpoint)))
            if recorded.parent != checkpoint.parent or not psro._adapter_checkpoint(recorded):
                raise RuntimeError(f"completed {target} checkpoint is missing: {recorded}")
            log(f"{target} already trained; reusing {recorded}")
            return psro._adapter_identity(recorded)

        # A preempted attempt leaves a partial ckpt tree. gpu2 is the only node
        # available and its project QOS is preemptible by qos_radio_freq, so an
        # oracle that needs ~5h is unlikely to survive a single allocation.
        # Rather than discard the work, continue from the newest retained
        # checkpoint and train only the steps that remain. The seam costs the
        # optimizer state and restarts the LR schedule, so a resumed oracle is
        # not bit-identical to an uninterrupted one; the manifest records it.
        # Mid-oracle resume trades a clean optimizer state and LR schedule for
        # not losing work to preemption. D7 was first trained across a seam and
        # came out anomalously weak, so a retrain that must be clean sets
        # PSRO_MIDRUN_RESUME=0 and restarts the oracle from step 0 instead.
        # Match role_phase.sh's default. A base-KL run must restart a partial
        # oracle from base; inheriting its adapter also changes the reference.
        kl_coef = float(os.environ.get("INIT_KL_COEF") or "0.01")
        if kl_coef > 0 and role_start_adapter:
            raise RuntimeError(
                f"{target}: INIT_KL_COEF={kl_coef} with a start adapter would "
                "anchor KL to that adapter, not the base. Set INIT_KL_COEF=0 "
                "for hot-start training."
            )
        # KL_REF=adapter_off takes the reference by switching the LoRA off in
        # the actor, so it is the base model whatever adapter the trainer
        # starts from: a resumed segment keeps the same base anchor.
        adapter_off_kl = os.environ.get("KL_REF") == "adapter_off"
        allow_resume = os.environ.get(
            "PSRO_MIDRUN_RESUME", "0" if kl_coef > 0 and not adapter_off_kl else "1"
        ) != "0"
        resume_from, resume_step = None, 0
        if run_dir.exists():
            # A resumed segment counts its checkpoints from zero again;
            # resume_state.json holds the oracle step it started at and the
            # checkpoint it started from.
            state_file = run_dir / "resume_state.json"
            prior = psro._read_json(state_file) if state_file.is_file() else {}
            base_step = int(prior.get("resume_step", 0))
            done = sorted(
                (base_step + int(m.group(1)), path)
                for path in (run_dir / "ckpt").glob("global_step*_hf")
                for m in [re.fullmatch(r"global_step(\d+)_hf", path.name)]
                if m and psro._adapter_checkpoint(path)
            ) if allow_resume else []
            if done and done[-1][0] < steps:
                if kl_coef > 0 and not adapter_off_kl:
                    raise RuntimeError(
                        f"{target}: base-KL training cannot resume a partial "
                        "adapter; set PSRO_MIDRUN_RESUME=0 or INIT_KL_COEF=0. "
                        "The existing checkpoint has been left in place."
                    )
                resume_step, newest = done[-1]
                keep = run_dir.with_name(f"{target}.segment_{resume_step}")
                if keep.exists():
                    shutil.rmtree(keep, ignore_errors=True)
                run_dir.rename(keep)
                resume_from = str(keep / "ckpt" / newest.name)
                log(f"{target} resuming from step {resume_step} ({resume_from})")
            elif allow_resume and not done and prior.get("resume_from") \
                    and psro._adapter_checkpoint(Path(prior["resume_from"])):
                # died before this segment's first checkpoint: start the same
                # seam again; the earlier segment directory holds the adapter
                resume_step, resume_from = base_step, prior["resume_from"]
                discarded = run_dir.with_name(f".discarded_{target}_{os.getpid()}")
                run_dir.rename(discarded)
                shutil.rmtree(discarded, ignore_errors=True)
                log(f"{target} resuming again from step {resume_step} ({resume_from})")
            else:
                log(f"{target} restarting; clearing partial {run_dir}")
                # rmtree races with NFS silly-rename (.nfsXXXX) when a preempted
                # process still holds a log handle. Rename first -- that always
                # succeeds -- then delete best-effort.
                discarded = run_dir.with_name(f".discarded_{target}_{os.getpid()}")
                run_dir.rename(discarded)
                shutil.rmtree(discarded, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        train_steps = steps
        if resume_from:
            # Only the TRAINER sees the resumed start and the reduced budget.
            # expected_role_start and `steps` stay at their contract values so
            # the manifest still describes the oracle as specified and the
            # idempotence check on a later re-entry still matches.
            role_start_adapter = resume_from
            train_steps = steps - resume_step
            # The trainer counts from zero again, so the budget checkpoint of a
            # resumed segment is global_step{train_steps}, not global_step{steps}.
            checkpoint = run_dir / "ckpt" / f"global_step{train_steps}_hf"
            psro._write_json_atomic(run_dir / "resume_state.json",
                                    {"resume_step": resume_step, "resume_from": resume_from})

        criterion = _build_criterion(role)
        effective_save_steps = math.gcd(save_steps, train_steps)
        if criterion is not None:
            effective_save_steps = math.gcd(effective_save_steps, criterion.window)
        environment = cfg.base_env()
        environment.update(
            {
                "SSP_ROOT": str(cfg.workspace),
                "SSP_WORK": str(cfg.work),
                "SSP_HF_HOME": str(cfg.hf_home),
                "SSP_SFT_DIR": str(cfg.sft_dir),
                "SSP_OUT": str(root / "training"),
                "SSP_PYTHON": cfg.python,
                "ROLE": role,
                "TAG": target,
                "GEN": target[1:],
                "STEPS": str(train_steps),
                "GPUS": str(cfg.gpus),
                "LORA_RANK": str(lora_rank),
                "LORA_ALPHA": str(lora_alpha),
                "LR": repr(float(learning_rate)),
                "SFT_STOP": str(sft_stop_after_step),
                "SFT_BATCHES_PER_STEP": str(sft_batches_per_step),
                "SAVE_STEPS": str(effective_save_steps),
                # role_phase.sh keeps only ONE checkpoint by default, which
                # leaves held-out selection nothing to choose between. Retain
                # every checkpoint this oracle writes.
                "MAX_CKPT_NUM": os.environ.get(
                    "MAX_CKPT_NUM",
                    str(max(steps // effective_save_steps + 2, 4)),
                ),
                "ACTOR_LR_SCHEDULER": actor_lr_scheduler,
                "LR_WARMUP_RATIO": repr(float(lr_warmup_ratio)),
                # A resumed segment restarts the prompt sampler; offsetting the
                # seed keeps it from replaying the first segment's prompts.
                "TRAINING_SEED": str(training_seed + resume_step),
                "OPPONENT_POOL_JSON": json.dumps(normalized_pool, sort_keys=True),
                # role_lora_v2.py sets both on every cold-PSRO oracle.
                "EXACT_LABEL_BALANCE": "1",
                "LABEL_DRIFT_POLICY": psro.TRAINING_LABEL_DRIFT_POLICY,
                # Score the attacker against its assigned prompt_type instead
                # of the label WildGuard observed. Empty keeps upstream v4.
                "ATTACKER_OBJECTIVE": os.environ.get(
                    "ATTACKER_OBJECTIVE", ""
                ),
                # Same in-training evaluation cadence as the archived runs, so
                # the W&B panels line up with A1/A2/D1/D2.
                "EVAL_STEPS": "10",
                "EVAL_START_STEPS": "50",
                "DIVERSITY_STEPS": "5",
                "WANDB_RUN_NAME": wandb_identity,
                "JUDGE_PORT": str(cfg.train_judge_port),
                "RAY_PORT": str(cfg.ray_port),
                # Keep scripts/slurm/README.md's 18.7s/step layout: no sleep
                # flags, 4 engines at TP=1, prefix caching, CUDA graphs. With
                # the judge colocated the sleep flags cost ~110s/step, because
                # --deepspeed_enable_sleep offloads optimizer state every step.
                "VLLM_SLEEP": os.environ.get("VLLM_SLEEP", "0"),
                "DS_SLEEP": os.environ.get("DS_SLEEP", "0"),
                "EAGER": os.environ.get("EAGER", "0"),
                # That layout leaves the ZeRO-3 actor ~22G of headroom, and a
                # PSRO oracle asked for 22.9G in one allocation. The traceback
                # puts it in make_experience -> actor.forward, i.e. the logprob
                # pass, which is batched by micro_rollout_batch_size. The
                # ladder's 32 is fine against a single frozen adapter, but a
                # trained Nash mixture answers at length and 32 packed
                # sequences reach ~48k tokens, whose fp32 logits are that 22.9G.
                # The archived Modal runs used 8; restoring it caps the pass at
                # ~6G and is pure batching, so the gradients are unchanged.
                "MICRO_ROLLOUT": os.environ.get("MICRO_ROLLOUT", "8"),
                # per-role micro-batch: attacker sequences are ~2.5x shorter
                "MICRO_TRAIN": os.environ.get(f"MICRO_TRAIN_{role.upper()}") or os.environ.get("MICRO_TRAIN", "8"),
                # Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
                # here. It pushes the LoRA weight sync onto the CUDA-IPC path,
                # which passes a memory fd with pidfd_getfd; this host runs
                # kernel.yama.ptrace_scope=1, so that call is denied between
                # sibling Ray actors and every engine update fails with
                # "pidfd_getfd: Operation not permitted".
            }
        )
        if role_start_adapter:
            environment["START_ADAPTER"] = role_start_adapter
        else:
            environment.pop("START_ADAPTER", None)
        environment.pop("OPPONENT", None)
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        # Per-role config: <KNOB>_ATTACKER / <KNOB>_DEFENDER replaces <KNOB>
        # for that role's oracles only (attacker phases are rollout-bound,
        # defender phases trainer-bound, so they want different settings).
        suffix = f"_{role.upper()}"
        per_role = {k[: -len(suffix)]: v for k, v in os.environ.items()
                    if k.endswith(suffix) and k[: -len(suffix)] in PER_ROLE_KNOBS}
        environment.update(per_role)
        if per_role:
            log(f"  per-role config ({role}): " + ", ".join(f"{k}={v}" for k, v in sorted(per_role.items())))

        training_log = run_dir / "training.log"
        log(
            f"training {target} ({role}) [{cfg.init_mode} start from "
            f"{role_start_adapter or 'base'}] vs frozen mixture "
            + ", ".join(
                f"{entry['id']}={float(entry['probability']):.4f}"
                for entry in normalized_pool
            )
        )
        log(f"  wandb run name: {wandb_identity}")
        log(f"  log: {training_log}")
        note = "ran the full budget"
        if ONEJOB:
            # Per-role split: attacker phases are rollout-bound (short
            # attacks, long frozen-defender replies; trainer idles ~10 s of a
            # 26 s step), defender phases trainer-bound (long replies to train
            # on).  ONEJOB_TRAIN_CARDS_<ROLE> overrides the shared default.
            train_cards = [int(c) for c in (os.environ.get(f"ONEJOB_TRAIN_CARDS_{role.upper()}")
                                            or os.environ.get("ONEJOB_TRAIN_CARDS", "2,3")).split(",")]
            # ONEJOB_SHARE_TRAIN_CARDS=1 (10-08): keep the train cards' vLLM servers awake and in the rollout pool.
            # With them asleep the trainer card idled ~43% of every step (flame dmon: cards 3/7 mean 52-55% util)
            # waiting for rollouts while the 3 other cards were pegged; needs a low ONEJOB_SERVER_UTILS on train cards.
            # "1" shares every train card; a card list ("3") shares only those (10-08: util 0.30 left no KV room on
            # an 80GB card, and card 2 is a pure rollout card in attacker phases, so it must stay at its full util).
            share = os.environ.get("ONEJOB_SHARE_TRAIN_CARDS", "0").strip()
            shared = list(train_cards) if share == "1" else [int(x) for x in share.split(",") if x.strip() and share != "0"]
            asleep = [c for c in train_cards if c not in shared]
            rollout_cards = [c for c in range(cfg.gpus) if c not in asleep]
            if asleep:
                SERVICES.sleep(asleep)
            environment.update({
                "ROLLOUT_SERVER": "1",
                # ASYNC_INFLIGHT_CAP_<ROLE> (10-08): defender phases have only the 2 judge-hosting cards for rollout;
                # a 768 cap queued 334 requests on a KV-full server and games aged past ASYNC_STALE_K (9% replayed).
                "ASYNC_INFLIGHT_CAP": os.environ.get(f"ASYNC_INFLIGHT_CAP_{role.upper()}",
                                                     os.environ.get("ASYNC_INFLIGHT_CAP", "0")),
                # GC_OFF_<ROLE>=1 (10-08): no gradient checkpointing for that role (defender trainer used 22 of 80 GB);
                # role_phase reads ${GC_FLAGS-<gc on>}, so an empty value disables it.
                **({"GC_FLAGS": ""} if os.environ.get(f"GC_OFF_{role.upper()}") == "1" else {}),
                # <KNOB>_<ROLE> (10-08, user): attacker KL drifts far (0.75 at the cap), defender barely moves (0.01-0.03,
                # coef at the floor) under one per-token target -> per-role KL target / coef bounds.
                **{k: os.environ[f"{k}_{role.upper()}"] for k in ("KL_TARGET", "KL_COEF_MIN", "KL_COEF_MAX", "ASYNC_STALE_K", "ASYNC_DEPTH", "ATTACK_AHEAD")
                   if os.environ.get(f"{k}_{role.upper()}")},
                # role_phase sets TRAIN_GPUS = GPUS - this; with shared train cards it is no longer len(rollout_cards)
                "DISAGG_ROLLOUT_GPUS": str(cfg.gpus - len(train_cards)),
                "EXTERNAL_ROLLOUT_SERVERS": ",".join(SERVICES.server_urls[c] for c in rollout_cards),
                "EXTERNAL_JUDGE_URL": SERVICES.url,
                "TRAIN_CARDS": __import__("onejob_services").phys_cards(",".join(str(c) for c in train_cards)),
                # diversity metrics (wanli/sbert/self-BLEU) are logging-only and
                # were where a trainer segfaulted; off unless asked for
                "DIVERSITY_STEPS": os.environ.get("ONEJOB_DIVERSITY_STEPS", "100000"),
                "ROLLOUT_TRAIN_PREFIX": f"{target}_",
                "RUNNER_LOG_DIR": str(run_dir),
            })
        if criterion is None:
            with training_log.open("ab") as handle:
                completed = subprocess.run(
                    ["bash", str(ROLE_PHASE)],
                    cwd=str(cfg.work), env=environment,
                    stdout=handle, stderr=subprocess.STDOUT,
                )
            returncode = completed.returncode
        else:
            returncode, note = _supervise_training(
                environment, training_log, criterion, checkpoint.parent
            )
            log(f"  early stop: {note}")
        _release_training_gpus(cfg)
        if ONEJOB:
            SERVICES.wake()
        stop_record = {"enabled": criterion is not None, "outcome": note,
                       "returncode": returncode,
                       **(criterion.report() if criterion else {})}
        psro._write_json_atomic(run_dir / "early_stop.json", stop_record)
        if returncode != 0:
            raise RuntimeError(
                f"{target} training failed rc={returncode}; "
                f"see {training_log}"
            )
        # Use the triggering checkpoint, even if training advanced another
        # step before the monitor signalled it to stop.
        if criterion is not None and criterion.triggered:
            stopping_step = criterion.measurements[-1].window_end
            checkpoint = checkpoint.parent / f"global_step{stopping_step}_hf"
            log(f"  stopped early; final checkpoint is {checkpoint.name}")
        if not psro._adapter_checkpoint(checkpoint):
            raise RuntimeError(f"{target} finished without {checkpoint}")
        # Publishing is a side effect of a finished role, never a precondition:
        # publish_role_checkpoint reports and swallows its own failures.
        from roll.utils.checkpoint_publish import publish_role_checkpoint

        # history_selfplay builds its own config without `suffix` or
        # `run_root`; run_dir is <root>/<suffix>/training/<target> in every
        # caller, so its grandparent name is the suffix.
        run_suffix = (getattr(cfg, "suffix", None)
                      or getattr(getattr(cfg, "run_root", None), "name", None)
                      or run_dir.parent.parent.name)
        publish_role_checkpoint(
            checkpoint, run_suffix=run_suffix, target=target, role=role,
            extra={"early_stop": stop_record.get("outcome"),
                   "steps_per_role": environment.get("STEPS_PER_ROLE"),
                   "reward_mode": environment.get("GAME_REWARD_MODE", "legacy"),
                   "topic_penalty": environment.get("TOPIC_PENALTY")},
            log=log,
        )

        import hashlib

        run_name = wandb_identity
        psro._write_json_atomic(
            manifest_path,
            {
                "method": "Self-RedTeam role-specific PEFT LoRA v2 (Slurm port)",
                "role": role,
                "ppo_reward_type": (__import__("roll.utils.game_v7", fromlist=["x"]).GAME_V7_VERSION
                                    if environment.get("GAME_REWARD_MODE") == "v7"
                                    else SHARED_GAME_REWARD_VERSION if shared_game_reward_enabled() else "general_sum"),
                # every knob that changes what the oracle optimises; a resume
                # with different values is a different experiment
                "training_knobs": {k: environment.get(k) for k in (
                    "GAME_REWARD_MODE", "KL_REF", "KL_EXACT", "INIT_KL_COEF", "FORMAT_GRAMMAR",
                    "FORMAT_RESCUE_RESERVE", "SFT_STOP", "ASYNC_STALE_K", "ASYNC_DEPTH",
                    "ENGINE_OLD_LOGP", "ASYNC_TIS_CAP", "ADAPTIVE_EPOCHS", "ADAPTIVE_EPOCHS_MAX", "MICRO_TRAIN",
                    "TRAIN_BATCH", "ROLLOUT_BATCH", "BASE_MODEL_PATH")},
                "strict_zero_sum_ppo_reward": shared_game_reward_enabled(),
                "shared_game_reward": shared_game_reward_contract() if shared_game_reward_enabled() else None,
                "label_drift_training_policy": "drop benign-to-harmful as in matrix" if shared_game_reward_enabled() else psro.TRAINING_LABEL_DRIFT_POLICY,
                "prompt_label_balance": (
                    "deterministic exact 50/50 harmful/benign finite prefix"
                ),
                "base_model": psro.BASE_MODEL,
                "init_mode": cfg.init_mode,
                "role_start": expected_role_start,
                "role_start_base": psro.BASE_MODEL,
                "fixed_opponent_base": psro.BASE_MODEL,
                "fixed_opponent_adapter": None,
                "fixed_opponent_pool": normalized_pool,
                "fixed_opponent_sampling": (
                    "deterministic request-level categorical draw; one frozen "
                    "opponent action per two-turn episode"
                ),
                "steps": steps,
                "save_steps": save_steps,
                "effective_save_steps": effective_save_steps,
                "early_stopping": stop_record,
                "resumed_from_step": resume_step or None,
                "resumed_from_checkpoint": resume_from,
                # Cumulative across resume seams: the trainer's local step
                # count restarts at zero in each segment.
                "steps_completed": resume_step
                + int(re.fullmatch(r"global_step(\d+)_hf", checkpoint.name).group(1)),
                "rollout_batch_size": int(environment.get("ROLLOUT_BATCH", "128")),
                "train_batch_size": int(environment.get("TRAIN_BATCH", "32")),
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "learning_rate": learning_rate,
                "actor_lr_scheduler": actor_lr_scheduler,
                "lr_warmup_ratio": lr_warmup_ratio,
                "seed": training_seed,
                "kl_penalty": kl_coef,
                "reference_kl_monitored": True,
                "sft_stop_after_step": sft_stop_after_step,
                "sft_batches_per_step": sft_batches_per_step,
                "wandb_identity": run_name,
                "wandb_run_id": hashlib.sha1(run_name.encode()).hexdigest()[:8],
                "wandb_entity": environment.get("WANDB_ENTITY", ""),
                "wandb_project": environment.get("WANDB_PROJECT", "self-play"),
                "wandb_group": environment.get("WANDB_GROUP", "upstream-selfredteam-role-lora-v2"),
                "checkpoint_retention": "all_hf_checkpoints_for_selection",
                "checkpoint": str(checkpoint),
                "executor": "slurm/role_phase.sh",
            },
        )
        # The oracle IS the final checkpoint of its training run. Held-out
        # checkpoint selection was removed: the pipeline has no disjoint
        # prompt set (training consumes rows 0-19,199 and the shipped
        # "holdout" file is 1000/1000 inside it), and the selector's vLLM
        # LoRA cache reused one lora_int_id across checkpoints, so every
        # curve it produced was the first checkpoint measured repeatedly.
        selected = checkpoint
        return psro._adapter_identity(selected)

    return train_cold_psro_oracle


def finalize_local_training(psro: Any, root: Path, state: dict[str, Any]) -> None:
    """Save the final meta-game without claiming external benchmarks ran."""
    generations = int(state["contract"]["generations"])
    attackers, defenders, matrix = psro.build_psro_payoff_matrix(
        state["population"], state["cells"]
    )
    if len(attackers) != generations + 1 or len(defenders) != generations + 1:
        raise RuntimeError("Cannot finalize an incomplete PSRO population")
    solution = psro.solve_zero_sum_meta_game(matrix)
    snapshot = psro._persist_matrix_snapshot(
        root, state, snapshot_name=f"final_g{generations}",
        attackers=attackers, defenders=defenders, matrix=matrix, solution=solution,
    )
    latest = root / "matrices/final.json"
    psro._write_json_atomic(latest, psro._read_json(Path(snapshot["path"])))
    state["matrix_snapshots"]["latest_final"] = {
        "path": str(latest), "sha256": psro._sha256_file(latest),
        "generation": generations, "immutable_snapshot": snapshot,
    }
    final = {"attackers": attackers, "defenders": defenders, "payoff": matrix,
             "meta_solution": solution, "snapshot": snapshot}
    state.setdefault("final_matrices", {})[str(generations)] = final
    state.update(final_matrix=final, training_completed=True,
                 stage="training_complete_evaluation_pending")
    psro._persist_psro_inventory(root, state)
    psro._write_json_atomic(root / "state.json", state)


def _base_model_path(cfg) -> str:
    """Resolve the base snapshot the same way role_phase.sh does."""
    override = os.environ.get("BASE_MODEL_PATH")
    if override:
        return override
    hub = cfg.hf_home / "hub" / "models--mlabonne--Meta-Llama-3.1-8B-Instruct-abliterated" / "snapshots"
    snapshots = sorted(p for p in hub.glob("*") if p.is_dir())
    if not snapshots:
        raise RuntimeError(f"no base-model snapshot under {hub}")
    return str(snapshots[0])


def initialize_state(psro: Any, cfg: Config, advance_kwargs: dict[str, Any]) -> None:
    """Write a fresh run's contract and A0/D0 population, and nothing else.

    ``advance_cold_psro`` persists the contract and the base population before
    it dispatches an action, but it has no initialize-only mode. Stubbing both
    workers makes the first call stop right after the durable write, so the
    judge and the cell workers are still set up by the normal loop below rather
    than an action running in-process with no judge.
    """

    class _Initialized(Exception):
        pass

    def _stop(**_kwargs: Any) -> None:
        raise _Initialized

    cell, oracle = psro.evaluate_zero_sum_psro_cell, psro.train_cold_psro_oracle
    psro.evaluate_zero_sum_psro_cell = common._bind_local(_stop)
    psro.train_cold_psro_oracle = common._bind_local(_stop)
    try:
        psro.advance_cold_psro(**advance_kwargs)
    except _Initialized:
        pass
    finally:
        psro.evaluate_zero_sum_psro_cell = cell
        psro.train_cold_psro_oracle = oracle
    state_path = cfg.run_root / "state.json"
    if not state_path.is_file():
        raise RuntimeError(f"coordinator did not initialize {state_path}")
    log(f"initialized {state_path}")


def _racing_plan(
    psro: Any, cfg: Config, state: dict[str, Any], pending: list[tuple[str, str]]
) -> dict[str, int]:
    """Decide a per-cell episode budget from the pilot estimates.

    Keeps the top-``m`` attacker rows and defender columns by point estimate
    against the current meta-strategy, plus anything already carrying Nash
    weight, and gives the survivors the full budget. Everything else stops at the
    pilot. Pruning is deliberately allowed to be wrong: the offline sweep on the
    completed 4x4 matrix put this at 18.9x less wall clock than the uniform
    4,000-per-cell schedule while still recovering the defender Nash support 62%
    of the time, against 92% for the full schedule.
    """
    import numpy as np
    from roll.utils.role_lora_zero_sum_psro import solve_zero_sum_meta_game

    contract = state["contract"]
    full = int(os.environ.get("RACE_FULL_EPISODES") or contract["matrix_episodes_per_cell"])
    attackers = sorted(
        (k for k, v in state["population"].items() if k.startswith("A")),
        key=lambda k: int(k[1:]),
    )
    defenders = sorted(
        (k for k, v in state["population"].items() if k.startswith("D")),
        key=lambda k: int(k[1:]),
    )
    cells = state["cells"]
    matrix = np.array(
        [
            [cells.get(f"{a}__{d}", {}).get("value", np.nan) for d in defenders]
            for a in attackers
        ],
        dtype=float,
    )
    if np.isnan(matrix).all():
        return {f"{a}__{d}": full for a, d in pending}
    # fill gaps with row/column means so the solver sees a complete matrix
    grand = float(np.nanmean(matrix))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                row = np.nanmean(matrix[i]); col = np.nanmean(matrix[:, j])
                cand = [x for x in (row, col) if not np.isnan(x)] or [grand]
                matrix[i, j] = float(np.mean(cand))
    try:
        sol = solve_zero_sum_meta_game(matrix, max_exploitability=1.0)
        p = np.array(sol["attacker_strategy"]); q = np.array(sol["defender_strategy"])
    except Exception as exc:                       # a poor pilot solve is not fatal
        log(f"  racing: meta-solve failed ({exc}); giving every cell the full budget")
        return {f"{a}__{d}": full for a, d in pending}
    row_value = matrix @ q
    col_value = p @ matrix
    m_row = int(os.environ.get("RACE_M_ROW", "2"))
    m_col = int(os.environ.get("RACE_M_COL", "2"))
    tau = float(os.environ.get("RACE_TAU", "0.01"))
    keep_a = {attackers[i] for i in np.argsort(-row_value)[:m_row]} | {
        attackers[i] for i in range(len(attackers)) if p[i] > tau
    }
    keep_d = {defenders[j] for j in np.argsort(col_value)[:m_col]} | {
        defenders[j] for j in range(len(defenders)) if q[j] > tau
    }
    # The newest attacker and defender are the best responses this generation was
    # run to produce, and they have no measured row/column to rank on, so they are
    # never allowed to be pruned on an imputed estimate.
    if attackers:
        keep_a.add(attackers[-1])
    if defenders:
        keep_d.add(defenders[-1])
    pilot = int(os.environ.get("RACE_PILOT_EPISODES", "256"))
    plan: dict[str, int] = {}
    for a, d in pending:
        key = f"{a}__{d}"
        # A cell that already finished on disk keeps the budget it was actually
        # run with. The meta-strategy moves as the population grows, so a
        # restart can recompute a different budget for a cell that is complete
        # but not yet recorded in state.json; the coordinator then compares the
        # new budget against the stored contract and rejects the cell as
        # changed, which wedges the run on every retry.
        done = cfg.run_root / "cells" / key / "summary.json"
        if done.is_file():
            try:
                plan[key] = int(json.loads(done.read_text())["episodes"])
                continue
            except (ValueError, KeyError, OSError):
                pass
        plan[key] = full if (a in keep_a and d in keep_d) else pilot
    log(f"  racing: keeping attackers {sorted(keep_a)}, defenders {sorted(keep_d)}")
    log(f"  racing: {sum(v == full for v in plan.values())}/{len(plan)} cells at the "
        f"full {full}, the rest capped at the {pilot}-episode pilot")
    return plan


# --- in-pipeline evaluation (ONEJOB_INPIPE_EVAL=1) + phase profile (ONEJOB_PHASE_PROFILE=1), 10-09 -------------
# While a new oracle's payoff cells run, an eval worker (inpipe_eval.py, own process group) evaluates that oracle
# through the running rollout servers (training format grammar) and the running WildGuard: attacker -> local victim
# on cards whose rollout server is put to sleep for the cell phase; defender -> the official suite's task code.
# The driver waits at most INPIPE_EVAL_MAX_WAIT seconds after the cells before killing the worker; it never raises.
def _inpipe_new_members(pending):
    attackers, defenders = {a for a, _ in pending}, {d for _, d in pending}
    new = ([next(iter(attackers))] if len(attackers) == 1 else []) + ([next(iter(defenders))] if len(defenders) == 1 else [])
    return [label for label in new if label[1:] != "0"]  # A0/D0 baselines are computed (once) inside the worker


def _inpipe_victim_cards(cfg):
    if os.environ.get("INPIPE_VICTIM_CARDS") is not None:
        return [int(c) for c in os.environ["INPIPE_VICTIM_CARDS"].split(",") if c.strip()]
    judge = {int(c) for c in os.environ.get("ONEJOB_JUDGE_GPUS", "0").split(",") if c.strip()}
    topic = set() if os.environ.get("ONEJOB_SHARED_BASE", "0") == "1" else {int(os.environ.get("ONEJOB_TOPIC_GPU", cfg.gpus - 1))}
    asleep = set(getattr(SERVICES, "asleep", set()))
    candidates = [c for c in range(cfg.gpus - 1, -1, -1) if c not in judge | topic | asleep]
    chosen = candidates[: int(os.environ.get("INPIPE_VICTIM_N", "1"))]
    awake_left = [c for c in range(cfg.gpus) if c not in asleep and c not in chosen]
    return chosen if awake_left else []


def _inpipe_launch(ctx, cfg, state, labels, judge_urls, cell_env=None):
    """Start one eval worker per new member; victim cards (attacker evals) get their rollout server put to sleep."""
    phys = __import__("onejob_services").phys_cards
    out = ctx["out"]
    if any(label.startswith("A") for label in labels):
        cards = _inpipe_victim_cards(cfg)
        if cards:
            SERVICES.sleep(cards)
            ctx["victim_cards"] = cards
            if cell_env is not None:
                cell_env["CELL_SERVERS"] = ",".join(SERVICES.awake_urls())
            log(f"  in-pipeline eval: victim card(s) {cards} (rollout server asleep meanwhile)")
    servers = ",".join(SERVICES.awake_urls())
    eval_py = os.environ.get("INPIPE_EVAL_PY") or (
        "/tmp/eval_cache/srteval/bin/python" if Path("/tmp/eval_cache/srteval/bin/python").exists() else cfg.python)
    max_new = int(state["contract"]["matrix_generation"]["max_new_tokens"])
    for label in labels:
        kind = "attacker" if label.startswith("A") else "defender"
        environment = cfg.base_env()
        environment["HF_HUB_CACHE"] = os.environ.get("INPIPE_EVAL_HF_HUB_CACHE", "/tmp/eval_cache/hf/hub")
        victim_cards = phys(",".join(str(c) for c in ctx["victim_cards"])) if ctx["victim_cards"] else ""
        command = [eval_py, str(SLURM_DIR / "inpipe_eval.py"), "--kind", kind, "--label", label,
                   "--adapter", str(state["population"][label].get("path")),
                   "--servers", servers, "--judge-urls", ",".join(judge_urls),
                   "--base-model", os.environ["BASE_MODEL_PATH"], "--out-dir", str(out),
                   "--max-new-tokens", str(max_new), "--victim-cards", victim_cards]
        handle = (out / f"worker_{label}.log").open("ab")
        proc = subprocess.Popen(command, cwd=str(cfg.work), env=environment, stdout=handle,
                                stderr=subprocess.STDOUT, start_new_session=True)
        ctx["procs"].append((label, proc, handle))
        ctx["prof"].add_status(out / f"status_{label}.json")
        ctx["prof"].event(f"launch {label}")
        log(f"  in-pipeline eval: {kind} {label} started (pid {proc.pid}, log {out / f'worker_{label}.log'})")


def _inpipe_start(cfg, state, pending, judge_urls, cell_env, cell_progress):
    """INPIPE_EVAL_MODE=inpipe (default): workers run alongside the cells; =separate: launched after the cells
    (the next oracle waits for them, i.e. the 'stop and evaluate' fallback without a relaunch)."""
    inpipe = os.environ.get("ONEJOB_INPIPE_EVAL", "0") == "1"
    if not inpipe and os.environ.get("ONEJOB_PHASE_PROFILE", "0") != "1":
        return None
    try:
        from inpipe_profile import PhaseProfiler
        phys = __import__("onejob_services").phys_cards
        out = cfg.run_root / "evals"
        out.mkdir(parents=True, exist_ok=True)
        labels = [label for label in (_inpipe_new_members(pending) if inpipe else [])
                  if not (out / f"{label}.json").exists()]
        tag = "_".join(labels) or f"cells_{len(state['cells'])}"
        cards_phys = [int(c) for c in phys(",".join(str(c) for c in range(cfg.gpus))).split(",")]
        ctx = {"t0": time.time(), "out": out, "procs": [], "victim_cards": [], "cfg": cfg, "state": state,
               "labels": labels, "judge_urls": list(judge_urls), "mode": os.environ.get("INPIPE_EVAL_MODE", "inpipe"),
               "prof": PhaseProfiler(out, tag, cards_phys, list(SERVICES.server_urls),
                                     progress=lambda: dict(cell_progress))}
        if labels and ctx["mode"] != "separate":
            _inpipe_launch(ctx, cfg, state, labels, judge_urls, cell_env)
        return ctx
    except Exception as exc:  # noqa: BLE001  (evaluation must never break the run)
        log(f"  in-pipeline eval: start failed ({type(exc).__name__}: {exc}); continuing without it")
        return None


def _inpipe_finish(ctx, cell_progress):
    if not ctx:
        return
    try:
        cells_s = time.time() - ctx["t0"]
        ctx["prof"].event("cells_done")
        if ctx["mode"] == "separate" and ctx["labels"]:
            _inpipe_launch(ctx, ctx["cfg"], ctx["state"], ctx["labels"], ctx["judge_urls"])
        budget = float(os.environ.get("INPIPE_EVAL_MAX_WAIT", "1800" if ctx["mode"] != "separate" else "7200"))
        deadline = time.time() + budget
        results = {}
        for label, proc, handle in ctx["procs"]:
            while proc.poll() is None and time.time() < deadline:
                time.sleep(5)
            if proc.poll() is None:
                log(f"  in-pipeline eval {label}: still running {budget:.0f}s after the cells; killed "
                    f"(partial results: {ctx['out'] / f'status_{label}.json'})")
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(os.getpgid(proc.pid), sig)
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=30)
                        break
            handle.close()
            path = ctx["out"] / f"{label}.json"
            results[label] = json.loads(path.read_text()) if path.exists() else None
            if results[label] is None:
                log(f"  in-pipeline eval {label}: no result (rc={proc.returncode}); see worker_{label}.log")
            elif "victims" in results[label]:
                for victim, arms in results[label]["victims"].items():
                    log(f"  in-pipeline eval {label} vs {victim}: " + ", ".join(
                        f"{arm} ASR {v['asr']:.3f}" for arm, v in sorted(arms.items())))
            else:
                log(f"  in-pipeline eval {label}: " + json.dumps(results[label].get("metrics")) +
                    " | D0 " + json.dumps(results[label].get("baseline_metrics")))
        if ctx["victim_cards"]:
            SERVICES.wake(ctx["victim_cards"])
        tail = time.time() - ctx["t0"] - cells_s
        timings = {label: (r or {}).get("timings") for label, r in results.items()}
        load = sum(v for t in timings.values() if t for k, v in t.items() if k.endswith("_load_s"))
        wall = sum((r or {}).get("wall_s", 0) for r in results.values())
        summary = ctx["prof"].stop({"cells_s": round(cells_s, 1), "eval_tail_after_cells_s": round(tail, 1),
                                    "cells": dict(cell_progress), "eval_timings": timings,
                                    "victim_load_s": round(load, 1),
                                    "victim_load_frac_of_eval": round(load / wall, 3) if wall else None})
        log(f"  {summary}; cells {cells_s / 60:.1f} min, eval tail after cells {tail / 60:.1f} min, "
            f"victim model load {load:.0f}s of {wall:.0f}s eval wall")
        if wall and load / wall > float(os.environ.get("INPIPE_LOAD_FRAC_WARN", "0.5")):
            log("  in-pipeline eval: model loading dominates the eval; consider INPIPE_EVAL_MODE=separate")
    except Exception as exc:  # noqa: BLE001
        log(f"  in-pipeline eval: finish failed ({type(exc).__name__}: {exc})")
        with contextlib.suppress(Exception):
            if ctx.get("victim_cards"):
                SERVICES.wake(ctx["victim_cards"])


def prefetch_cells(psro: Any, cfg: Config, state: dict[str, Any], judge: Judge) -> None:
    from roll.utils.role_lora_cold_psro import missing_matrix_cells

    pending = missing_matrix_cells(state["population"], state["cells"])
    if not pending:
        return
    contract = state["contract"]
    generation = contract["matrix_generation"]
    race = os.environ.get("RACE_CELLS", "0") == "1"
    plan: dict[str, int] = {}
    if race:
        plan = _racing_plan(psro, cfg, state, list(pending))
        plan_path = cfg.run_root / "cell_episode_plan.json"
        payload = {
            "default": int(os.environ.get("RACE_FULL_EPISODES")
                           or contract["matrix_episodes_per_cell"]),
            "pilot": int(os.environ.get("RACE_PILOT_EPISODES", "256")),
            "m_row": int(os.environ.get("RACE_M_ROW", "2")),
            "m_col": int(os.environ.get("RACE_M_COL", "2")),
            "cells": plan,
            "note": (
                "Per-cell episode budgets chosen by the racing scheduler. A reduced "
                "budget is an exact prefix of the full-budget episode sequence, so a "
                "capped cell is the same estimand stopped early."
            ),
        }
        plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log(f"  racing: wrote {plan_path}")

    def episode_budget(attacker: str, defender: str) -> int:
        return plan.get(f"{attacker}__{defender}", int(contract["matrix_episodes_per_cell"]))

    # advance_cold_psro records one cell per call, so after a prefetch the
    # loop comes back here with cells already finished on disk. Relaunching a
    # worker just to hit "Reusing completed payoff cell" costs a process start
    # each; the coordinator reuses them in-process instead.
    raw_root = Path(getattr(psro, "RAW_PAYOFF_ROOT"))

    def raw_done(attacker: str, defender: str) -> bool:
        name = f"{cfg.suffix}__{attacker}__{defender}__n{episode_budget(attacker, defender)}__zs_v4"
        return (raw_root / name / "summary.json").exists()

    pending = [(a, d) for a, d in pending if not raw_done(a, d)]
    if not pending:
        return
    judge_url = judge.start()
    cfg.cell_log_dir.mkdir(parents=True, exist_ok=True)
    log(f"prefetching {len(pending)} payoff cell(s) on GPUs 1-{cfg.cell_workers}"
        + (f" (+ shared judge cards)" if os.environ.get("CELL_SHARE_JUDGE_GPUS", "1") == "1"
           and os.environ.get("CELL_STREAMING", "1") != "0" else ""))

    queue = list(pending)
    running: dict[str, tuple[str, str, subprocess.Popen[bytes], Any]] = {}
    failures: list[str] = []
    worker_gpus = list(range(1, cfg.cell_workers + 1))
    cell_judge_urls = [judge_url]
    cell_env = {}
    if ONEJOB:
        # every card serves; a cell worker is just a client of the servers
        SERVICES.wake()
        free_gpus = [[] for _ in pending]
        n_wg = int(os.environ.get("ONEJOB_CELL_WG_REPLICAS", "0"))
        if n_wg > 0:
            # ONEJOB_CELL_WG_REPLICAS (10-09): extra WildGuard servers on the last n cards (or
            # ONEJOB_CELL_WG_CARDS) for the cell phase; their rollout servers sleep meanwhile and
            # cells generate on the remaining awake servers.
            judge_cards = {int(c) for c in os.environ.get("ONEJOB_JUDGE_GPUS", "0").split(",") if c.strip()}
            wg_cards = ([int(c) for c in os.environ["ONEJOB_CELL_WG_CARDS"].split(",")]
                        if os.environ.get("ONEJOB_CELL_WG_CARDS")
                        else [c for c in range(cfg.gpus - 1, -1, -1) if c not in judge_cards][:n_wg])
            cell_judge_urls += SERVICES.start_cell_judges(wg_cards)
            cell_env["CELL_SERVERS"] = ",".join(SERVICES.awake_urls())
            log(f"  cell judges: {cell_judge_urls}; cell servers: {cell_env['CELL_SERVERS']}")
    elif os.environ.get("CELL_STREAMING", "1") != "0":
        # A streaming cell runs one engine per GPU it is given, so never leave
        # a card idle: split the worker cards over the cells that can run now,
        # and hand out the judge cards too (the pool sizes each engine to what
        # the judge left free, and skips a card with too little).
        slots = max(1, min(len(pending), cfg.cell_workers))
        free_gpus = [worker_gpus[i::slots] for i in range(slots)]
        if os.environ.get("CELL_SHARE_JUDGE_GPUS", "1") == "1":
            spare = [g for g in range(cfg.gpus) if g not in worker_gpus]
            for index, gpu in enumerate(spare):
                free_gpus[index % slots].append(gpu)
    else:
        free_gpus = [[gpu] for gpu in worker_gpus]

    cell_progress = {"total": len(pending), "done": 0}
    inpipe = _inpipe_start(cfg, state, pending, cell_judge_urls, cell_env, cell_progress) if ONEJOB else None
    launched = [0]
    def launch(gpus: list[int], attacker: str, defender: str) -> None:
        attacker_record = state["population"][attacker]
        defender_record = state["population"][defender]
        environment = cfg.base_env()
        environment.update(cell_env)
        cell_judge = cell_judge_urls[launched[0] % len(cell_judge_urls)]
        launched[0] += 1
        gpu = ",".join(str(g) for g in gpus) or f"srv{attacker}{defender}"
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        environment["CELL_ENGINE_GPUS"] = ",".join(str(g) for g in gpus)
        command = [
            cfg.python, str(CELL_WORKER),
            "--coordinator", cfg.coordinator,
            "--psro-root", str(cfg.psro_root),
            "--raw-payoff-root", str(cfg.raw_payoff_root),
            "--upstream-source", str(cfg.work),
            "--judge-url", cell_judge,
            "--run-suffix", cfg.suffix,
            "--attacker-label", attacker,
            "--defender-label", defender,
            "--attacker-adapter", attacker_record.get("path") or "base",
            "--defender-adapter", defender_record.get("path") or "base",
            "--attacker-sha256", attacker_record["adapter_sha256"],
            "--defender-sha256", defender_record["adapter_sha256"],
            "--episodes", str(episode_budget(attacker, defender)),
            "--seed-base", str(contract["seed_base"]),
            "--max-candidate-multiplier", str(generation["max_candidate_multiplier"]),
            "--candidate-wave-pairs", str(generation["candidate_wave_pairs"]),
            *(
                ["--adaptive-stratum-sampling"]
                if os.environ.get("ADAPTIVE_STRATUM_SAMPLING") == "1"
                else []
            ),
            "--generation-batch-size", str(generation["generation_batch_size"]),
            "--judge-batch-size", str(generation["judge_batch_size"]),
            "--max-new-tokens", str(generation["max_new_tokens"]),
        ]
        cell_log = cfg.cell_log_dir / f"{attacker}__{defender}.log"
        handle = cell_log.open("ab")
        process = subprocess.Popen(
            command, cwd=str(cfg.work), env=environment,
            stdout=handle, stderr=subprocess.STDOUT,
        )
        running[gpu] = (attacker, defender, process, handle)
        log(f"  cell {attacker} vs {defender} -> GPU {gpu}, judge {cell_judge}, log {cell_log}")

    while queue or running:
        while queue and free_gpus:
            launch(free_gpus.pop(0), *queue.pop(0))
        time.sleep(float(os.environ.get("CELL_POLL_SECONDS", "2")))
        for gpu, (attacker, defender, process, handle) in list(running.items()):
            if process.poll() is None:
                continue
            handle.close()
            del running[gpu]
            cell_progress["done"] += 1
            free_gpus.append([] if ONEJOB else [int(g) for g in gpu.split(",")])
            if process.returncode == 0:
                log(f"  cell {attacker} vs {defender} done")
            else:
                failures.append(f"{attacker}__{defender} rc={process.returncode}")
                log(f"  cell {attacker} vs {defender} FAILED rc={process.returncode}")
    _inpipe_finish(inpipe, cell_progress)  # before the replica stop wakes every server
    if ONEJOB and getattr(SERVICES, "cell_judge_procs", None) is not None and int(os.environ.get("ONEJOB_CELL_WG_REPLICAS", "0")) > 0:
        SERVICES.stop_cell_judges()
        SERVICES.wake()  # training re-sleeps its own cards; a rollout card must not stay asleep
    if failures:
        # Not fatal: the coordinator re-runs any cell it still needs inline, on
        # the GPUs the workers just released. Surface it loudly either way.
        log(f"prefetch failures (coordinator will retry inline): {failures}")


# --- main loop -----------------------------------------------------------------


def main() -> int:
    # Before anything reads the reward contract. advance_cold_psro writes it
    # during initialize_state, and that call lives in a contract-hashed source
    # we must not edit, so the configuration has to be in place beforehand.
    configure_local_topic_backend()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--hf-home", required=True, type=Path)
    parser.add_argument("--sft-dir", required=True, type=Path)
    parser.add_argument("--psro-root", required=True, type=Path)
    parser.add_argument("--raw-payoff-root", required=True, type=Path)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--cell-workers", type=int, default=3)
    parser.add_argument("--cell-judge-port", type=int, default=5310)
    parser.add_argument("--cell-judge-util", type=float, default=0.45)
    parser.add_argument("--train-judge-port", type=int, default=5311)
    parser.add_argument("--ray-port", type=int, default=6311)
    parser.add_argument("--run-official-evaluation", action="store_true")
    parser.add_argument("--coordinator", default=common.COLD_COORDINATOR)
    parser.add_argument("--init-mode", choices=("cold", "hot"), default="cold")
    parser.add_argument("--max-actions", type=int, default=0)
    parser.add_argument(
        "--fresh-cold-run", action="store_true",
        help="initialize a new cold run instead of requiring a seeded state",
    )
    parser.add_argument("--generations", type=int, default=0)
    args = parser.parse_args()

    cfg = Config(args)
    cfg.apply_to_process()
    state_path = cfg.run_root / "state.json"
    if not state_path.is_file():
        if cfg.init_mode != "hot" and not args.fresh_cold_run:
            raise SystemExit(
                f"no seeded run at {state_path}; run seed_cold_psro_run.py first"
            )
        # Both a hot run and an explicitly requested fresh cold run start at
        # generation one, so there is nothing to seed: the coordinator's own
        # initializer writes A0/D0 and the contract.
        log(f"no state at {state_path}; initializing a fresh {cfg.init_mode} run")

    psro, payoff = common.load_modules(cfg.coordinator)
    common.retarget(
        psro, payoff,
        psro_root=cfg.psro_root,
        raw_payoff_root=cfg.raw_payoff_root,
        upstream_source=cfg.work,
        judge_url="http://127.0.0.1:0/classify",
    )
    if ONEJOB:
        global SERVICES
        from onejob_services import OneJobServices

        os.environ.setdefault("BASE_MODEL_PATH", _base_model_path(cfg))
        SERVICES = OneJobServices(cfg, log)
        SERVICES.launch()
        import atexit

        atexit.register(SERVICES.shutdown)
        os.environ.update(SERVICES.client_env())
        os.environ["CELL_SERVERS"] = ",".join(SERVICES.server_urls)
        judge = SERVICES
    else:
        judge = Judge(cfg)
    psro.train_cold_psro_oracle = common._bind_local(
        make_local_oracle_trainer(psro, cfg)
    )
    psro._stable_wildguard_rm_url = lambda *a, **k: judge.url
    payoff._stable_wildguard_rm_url = lambda *a, **k: judge.url

    from roll.utils.role_lora_cold_psro import next_action

    if state_path.is_file():
        contract = psro._read_json(state_path)["contract"]
    else:
        contract = None
    advance_kwargs = (
        psro._advance_kwargs(contract, cfg.suffix)
        if contract
        else {"run_suffix": cfg.suffix}
    )
    if args.generations:
        # Also honoured on an EXISTING contract: that is the append-only
        # generation extension. _advance_kwargs rebuilds every field from the
        # stored contract, so without this the flag is silently dropped and the
        # driver reports "run already complete" instead of extending. The
        # coordinator still refuses a bump that is not a clean extension of a
        # completed run, so this cannot corrupt a run in progress.
        advance_kwargs["generations"] = args.generations
    if not contract:
        # A fresh run writes its contract on the first advance, so the two
        # behaviour changes have to be supplied here to be recorded in it.
        objective = os.environ.get("ATTACKER_OBJECTIVE", "")
        if objective:
            advance_kwargs["training_attacker_objective"] = objective
        if os.environ.get("ADAPTIVE_STRATUM_SAMPLING") == "1":
            advance_kwargs["matrix_adaptive_stratum_sampling"] = True
        # Training budget and checkpoint cadence are contract fields, so they can
        # only be chosen when the contract is first written. Every attacker in
        # both arms was still improving at step 100, and 5 of 8 defenders peaked
        # before it, so the budget is raised and checkpoints are kept densely
        # enough to select on afterwards.
        # Cell generation shape. generation_batch_size >= 2 * candidate_wave_pairs
        # makes each wave ONE vLLM call per role, so continuous batching refills
        # finished slots instead of every 64-prompt chunk waiting for its longest
        # reply (a 2048-token defender straggler gated each chunk). Contract
        # fields: chosen when the run starts, recorded, never changed mid-run.
        for key, env in (("steps_per_role", "STEPS_PER_ROLE"),
                         ("save_steps", "SAVE_STEPS_CONTRACT"),
                         ("training_seed", "TRAINING_SEED"),
                         ("matrix_episodes", "MATRIX_EPISODES"),
                         ("candidate_wave_pairs", "MATRIX_WAVE_PAIRS"),
                         ("generation_batch_size", "MATRIX_GENERATION_BATCH"),
                         ("judge_batch_size", "MATRIX_JUDGE_BATCH")):
            raw = os.environ.get(env)
            if raw:
                advance_kwargs[key] = int(raw)
        budget = advance_kwargs.get("steps_per_role", 100)
        # SFT_STOP in the caller's environment overrides both roles (0 = no
        # online SFT term); it used to be silently replaced by these literals.
        override = os.environ.get("SFT_STOP")
        advance_kwargs["attacker_sft_stop_after_step"] = int(override) if override else min(30, budget)
        advance_kwargs["defender_sft_stop_after_step"] = int(override) if override else min(10, budget)
        # per-role learning rates (coordinator defaults: attacker 1e-5, defender 4e-5)
        for role_ in ("attacker", "defender"):
            lr_env = os.environ.get(f"{role_.upper()}_LR")
            if lr_env:
                advance_kwargs[f"{role_}_learning_rate"] = float(lr_env)
    log(f"coordinator: {cfg.coordinator} (init_mode={cfg.init_mode})")
    log(f"run root: {cfg.run_root}")
    if contract:
        log(f"contract generations={contract['generations']} "
            f"steps_per_role={contract['steps_per_role']} "
            f"episodes/cell={contract['matrix_episodes_per_cell']}")

    # PSRO cells call the topic API directly and nothing else keeps a judge
    # alive between phases, so with the local backend the matrix stage died
    # with "Topic API request failed or timed out". Own one judge for the whole
    # run, as the container's entrypoint already does.
    judge_context = contextlib.nullcontext()
    from roll.utils.role_lora_training_reward import invalid_move_reward_enabled
    if ((shared_game_reward_enabled() or invalid_move_reward_enabled())
            and os.environ.get("TOPIC_JUDGE_BACKEND", "local") == "local"
            and os.environ.get("TOPIC_JUDGE_MANAGED") != "1"):
        from roll.utils.local_topic_runtime import local_topic_service

        service_env = dict(os.environ)
        service_env.setdefault("SSP_PYTHON", cfg.python)
        service_env.setdefault("SSP_ROOT", str(cfg.workspace))
        service_env.setdefault("TOPIC_JUDGE_PORT",
                               str(int(os.environ.get("JUDGE_PORT", cfg.train_judge_port)) + 300))
        service_env.setdefault("TOPIC_JUDGE_MODEL", os.environ.get("TOPIC_JUDGE_BASE_MODEL") or os.environ.get("BASE_MODEL_PATH", ""))
        # serve_topic_judge.py defaults to 0.60 of the card when run standalone,
        # which is right for a dedicated engine and catastrophic here: it starves
        # the trainable engine, the opponent and the ZeRO-3 shard sharing that
        # card, and the failure surfaces as an opaque NCCL "unhandled cuda
        # error" at init. Mirror role_phase.sh's measured budget instead.
        service_env.setdefault("TOPIC_JUDGE_UTIL", "0.13")
        service_env.setdefault("TOPIC_JUDGE_MAX_LEN", "4096")
        # Cell workers occupy GPUs 1..cell_workers and the cell judge takes GPU
        # 0, so the run-scoped judge has to come off the end and the worker
        # count has to shrink by one -- otherwise the judge's ~18 GiB sits on a
        # cell worker's card and the cell dies with a plain CUDA OOM. The
        # container's entrypoint already does exactly this (cell-workers 2 when
        # a judge is managed, judge on GPU 3); the bare path did not.
        service_env.setdefault("TOPIC_JUDGE_GPU", str(max(cfg.gpus - 1, 0)))
        if cfg.cell_workers > cfg.gpus - 2:
            log(f"run-scoped judge on GPU {cfg.gpus - 1}: cell workers "
                f"{cfg.cell_workers} -> {cfg.gpus - 2}")
            cfg.cell_workers = max(cfg.gpus - 2, 1)
        # Point the judge and every client at it, exactly as the container's
        # entrypoint does for its managed judge: the service serves under
        # TOPIC_API_MODEL, and payoff cells and role_phase.sh (which reuses a
        # managed judge) read TOPIC_API_*. Local mode overrides any leftover
        # hosted-API settings so nothing is sent to a cloud endpoint.
        local_api = {
            "TOPIC_API_PROVIDER": "openai_compatible",
            "TOPIC_API_MODEL": "local-topic-judge",
            "TOPIC_API_BASE_URL": f"http://127.0.0.1:{service_env['TOPIC_JUDGE_PORT']}/v1",
            "TOPIC_API_KEY": "local",
        }
        service_env.update(local_api)
        service_env.pop("TOPIC_API_KEY_FILE", None)
        os.environ.update(local_api)
        os.environ.pop("TOPIC_API_KEY_FILE", None)
        judge_context = local_topic_service(service_env, SLURM_DIR.parent.parent)
        log(f"run-scoped topic judge on :{service_env['TOPIC_JUDGE_PORT']} "
            f"(GPU {service_env.get('TOPIC_JUDGE_GPU', '3')})")

    actions = 0
    try:
      with judge_context:
        while True:
            if not state_path.is_file():
                initialize_state(psro, cfg, advance_kwargs)
                contract = psro._read_json(state_path)["contract"]
                advance_kwargs = psro._advance_kwargs(contract, cfg.suffix)
                continue
            state = psro._read_json(state_path)
            if state.get("completed") is True:
                current = int(state["contract"]["generations"])
                requested = int(args.generations or 0)
                if requested > current:
                    # Append-only generation extension. It has to go straight
                    # through advance_cold_psro: next_action() below reads the
                    # OLD generation count off the stored contract and would
                    # just report the run finished again. The coordinator
                    # rewrites the contract, clears `completed`, and keeps the
                    # population and every measured cell; it refuses the bump
                    # outright if the run is not a clean extension.
                    log(f"extending generations {current} -> {requested}")
                    psro.advance_cold_psro(**advance_kwargs)
                    contract = psro._read_json(state_path)["contract"]
                    advance_kwargs = psro._advance_kwargs(contract, cfg.suffix)
                    actions += 1
                    continue
                log("run already complete")
                break
            action = next_action(
                state["population"], state["cells"], state["evaluations"],
                generations=state["contract"]["generations"],
            )
            log(f"next action: {action}")
            if action["kind"] == "evaluation" and cfg.stop_before_evaluation:
                finalize_local_training(psro, cfg.run_root, state)
                generations = int(state["contract"]["generations"])
                log(
                    f"All {generations} rounds and the {generations + 1}x"
                    f"{generations + 1} matrix are complete; saved matrices/final.json. "
                    "Official defender benchmarks are a separate evaluation job."
                )
                break
            if action["kind"] == "cell" and ONEJOB:
                prefetch_cells(psro, cfg, state, judge)
            elif action["kind"] == "cell":
                # If a worker dies the coordinator evaluates that cell inline,
                # in this process. Keep GPU 0 out of its view so the fallback
                # never lands a 0.90-utilization engine on the judge's card.
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                    str(index) for index in range(1, cfg.cell_workers + 1)
                )
                prefetch_cells(psro, cfg, state, judge)
            else:
                judge.stop()
                # Cell workers and their engines must be gone before training
                # claims the cards; judge.stop() only confirms GPU 0.
                if not ONEJOB:
                    _confirm_gpus_free(cfg)
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            result = psro.advance_cold_psro(**advance_kwargs)
            actions += 1
            if isinstance(result, dict) and result.get("completed") is True:
                log("PSRO run complete")
                break
            log(f"completed action: {result.get('completed_action', result)}")
            if args.max_actions and actions >= args.max_actions:
                log(f"stopping after {actions} action(s) as requested")
                break
    finally:
        judge.stop()

    final = psro._read_json(state_path)
    log(f"stage={final.get('stage')} completed={final.get('completed')}")
    log(f"population={sorted(final.get('population', {}))}")
    matrix_size = (int(final["contract"]["generations"]) + 1) ** 2
    log(f"cells={len(final.get('cells', {}))}/{matrix_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

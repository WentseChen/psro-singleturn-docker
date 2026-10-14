"""Container entrypoint for the existing single-node Llama 8B LoRA workflow."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


BASE_MODEL = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
JUDGE_MODEL = "allenai/wildguard"
REPO = Path(__file__).resolve().parents[1]


def valid_run_name(name: str | None) -> bool:
    return bool(name and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?", name))


def configure_wandb(env: dict[str, str]) -> None:
    # This private teacher handoff intentionally includes a fixed credential.
    # Never print it or include it in the public run-settings record.
    path = REPO / "docker/wandb.fixed.json"
    if path.is_file():
        try:
            config = json.loads(path.read_text())
            fields = ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT", "WANDB_GROUP")
            if any(not isinstance(config.get(key), str) or not config[key].strip() for key in fields):
                raise ValueError
            env.update({key: config[key] for key in fields})
            env["WANDB_MODE"] = "online"
        except (ValueError, OSError):
            raise RuntimeError("Invalid private W&B configuration in docker/wandb.fixed.json") from None


def environment() -> dict[str, str]:
    env = dict(os.environ)
    workspace = Path(env.get("SSP_ROOT", "/workspace"))
    env.update(
        SSP_ROOT=str(workspace),
        SSP_PYTHON=sys.executable,
        HF_HOME=env.get("SSP_HF_HOME", str(workspace / "hf")),
        PSRO_ROOT=env.get("PSRO_ROOT", str(workspace / "psro/role_lora_zero_sum_psro")),
        RAW_PAYOFF_ROOT=env.get("RAW_PAYOFF_ROOT", str(workspace / "psro/raw_payoff_v2")),
        GPUS="4",
    )
    env["HF_HUB_CACHE"] = str(Path(env["HF_HOME"]) / "hub")
    # Default reward: the v6d (legacy) reward with the flat invalid-move floor
    # for the attacker. The flag is defined on the legacy reward only.
    # An explicit GAME_REWARD_MODE=shared_binary without the flag keeps it off.
    env.setdefault("ATTACKER_INVALID_MOVE",
                   "0" if env.get("GAME_REWARD_MODE") == "shared_binary" else "1")
    if env["ATTACKER_INVALID_MOVE"] not in {"0", "1"}:
        raise ValueError("ATTACKER_INVALID_MOVE must be 0 or 1")
    if env["ATTACKER_INVALID_MOVE"] == "1":
        env.setdefault("GAME_REWARD_MODE", "legacy")
        if env["GAME_REWARD_MODE"] != "legacy":
            raise ValueError("ATTACKER_INVALID_MOVE=1 requires GAME_REWARD_MODE=legacy; "
                             "set ATTACKER_INVALID_MOVE=0 to run shared_binary")
    for name, default in (("INIT_KL_COEF", "0"), ("OFFTOPIC_ENABLE", "0"),
                          ("GAME_REWARD_MODE", "shared_binary"), ("TOPIC_PENALTY", "2"),
                          ("TOPIC_JUDGE_BACKEND", "local"), ("TOPIC_API_PROVIDER", "anthropic"),
                          ("TOPIC_API_BASE_URL", "https://api.anthropic.com"),
                          ("TOPIC_API_MODEL", "claude-opus-5"),
                          ("PSRO_MIDRUN_RESUME", "0"), ("LADDER_EARLY_STOP", "1"),
                          # Racing payoff estimation, as the v6d run: every pending cell
                          # gets a 256-episode pilot and only the Nash-relevant rows and
                          # columns are topped up to MATRIX_EPISODES (~3x fewer episodes).
                          ("RACE_CELLS", "1")):
        env.setdefault(name, default)
    if env["TOPIC_JUDGE_BACKEND"] not in {"local", "anthropic"}:
        raise ValueError("TOPIC_JUDGE_BACKEND must be local or anthropic")
    needs_topic_judge = (env["GAME_REWARD_MODE"] == "shared_binary"
                         or env.get("ATTACKER_INVALID_MOVE", "0") == "1")
    if needs_topic_judge and env["TOPIC_JUDGE_BACKEND"] == "local":
        port = int(env.get("TOPIC_JUDGE_PORT", "5410"))
        if not 1024 <= port <= 65535:
            raise ValueError("TOPIC_JUDGE_PORT must be between 1024 and 65535")
        model = env.get("TOPIC_LOCAL_MODEL") or BASE_MODEL
        env.update(TOPIC_JUDGE_MANAGED="1", TOPIC_JUDGE_PORT=str(port),
                   TOPIC_API_PROVIDER="openai_compatible",
                   TOPIC_API_BASE_URL=f"http://127.0.0.1:{port}/v1",
                   TOPIC_API_MODEL=model, TOPIC_API_KEY="local")
        env.pop("TOPIC_API_KEY_FILE", None)
        # GPU 3 hosts the label model throughout training/matrix estimation.
        # Reduce colocated rollout reservations; PSRO cells use GPUs 1 and 2.
        #
        # TOPIC_JUDGE_GPU is what actually places it.  Until it existed the
        # comment above was aspirational: role_phase.sh launched every judge on
        # GPU 0, so the topic judge landed on top of WildGuard.  vLLM measures
        # gpu_memory_utilization against the whole card and subtracts what other
        # processes already hold, so the second engine was left less memory than
        # its own weights and died with "No available memory for the cache
        # blocks".  Keeping the two judges on separate cards is what makes four
        # GPUs enough.  Measured per judge on H200: ~17 GiB, 0.118 of a card.
        #
        # These are the settings of the validated 100-step A3 run on 4xH200, not
        # estimates. With the judge on its own card the trainable and opponent
        # engines keep their proven 0.28 / 0.26 and the rollout engines stay
        # resident: the earlier 0.24 / 0.22 + VLLM_SLEEP=1 existed only to make
        # room for a judge believed to be stacked on GPU 0, and the sleep costs
        # ~110s/step because deepspeed offloads optimizer state every step. The
        # measured run held ~50s/step with VLLM_SLEEP=0 and never OOMed, on the
        # same PSRO frozen-mixture shape this pipeline uses.
        for name, default in (("TOPIC_JUDGE_GPU", "3"), ("TOPIC_JUDGE_UTIL", "0.13"),
                              ("TOPIC_JUDGE_MAX_LEN", "4096"),
                              ("UTIL", "0.28"), ("OPP_UTIL", "0.26"),
                              ("VLLM_SLEEP", "0")):
            env.setdefault(name, default)
    else:
        env.pop("TOPIC_JUDGE_MANAGED", None)
    env.pop("TRANSFORMERS_CACHE", None)
    configure_wandb(env)
    return env


def model_paths(env: dict[str, str], *, download: bool = False) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    paths = {}
    for key, model in (("BASE_MODEL_PATH", BASE_MODEL), ("JUDGE_MODEL_PATH", JUDGE_MODEL)):
        override = env.get(key)
        if override:
            if not (Path(override) / "config.json").is_file():
                raise RuntimeError(f"Missing model config in {key}={override}")
            paths[key] = override
        else:
            try:
                paths[key] = snapshot_download(
                    model, cache_dir=env["HF_HUB_CACHE"], local_files_only=not download,
                    token=env.get("HF_TOKEN") or None,
                )
            except Exception as exc:
                # Do not echo request objects or credential-bearing environment.
                raise RuntimeError(
                    f"Cannot {'download' if download else 'find cached'} {model} "
                    f"({type(exc).__name__}). Run prepare with an authorized HF_TOKEN "
                    "on a machine with network access."
                ) from None
    paths["COLD_PSRO_BASE_MODEL_DIR"] = paths["BASE_MODEL_PATH"]
    if env.get("TOPIC_JUDGE_MANAGED") == "1":
        local_model = env.get("TOPIC_LOCAL_MODEL") or BASE_MODEL
        if local_model == BASE_MODEL:
            paths["TOPIC_JUDGE_MODEL"] = paths["BASE_MODEL_PATH"]
        elif Path(local_model).is_absolute():
            if not (Path(local_model) / "config.json").is_file():
                raise RuntimeError("TOPIC_LOCAL_MODEL directory is not visible inside the container")
            paths["TOPIC_JUDGE_MODEL"] = local_model
        else:
            paths["TOPIC_JUDGE_MODEL"] = snapshot_download(
                local_model, cache_dir=env["HF_HUB_CACHE"], local_files_only=not download,
                token=env.get("HF_TOKEN") or None,
            )
    return paths


def check_gpus() -> None:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("This preset needs exactly 4 visible GPUs; select them with GPU_DEVICES=0,1,2,3.")
    for index in range(4):
        gpu = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {gpu.name}, {gpu.total_memory / 2**30:.1f} GiB", flush=True)


def report_build_provenance(env: dict[str, str]) -> None:
    """Say which commit is baked into this image, and refuse a stale one.

    The repo is COPYed into the image, so a container's code is frozen at build
    time while /workspace persists across images. An image built before the
    judges were placed on separate cards puts both on GPU 0 and dies at step 0
    with "No available memory for the cache blocks" -- an expensive way to find
    out you forgot to rebuild.

    The test is on the baked source, not on a version string: a stale image is
    stale whatever it claims.
    """
    info = REPO / "BUILD_INFO"
    stamp = info.read_text().strip().replace("\n", " ") if info.is_file() else ""
    print(f"image build: {stamp or 'unstamped (built before provenance was added)'}", flush=True)
    phase = (REPO / "scripts/slurm/role_phase.sh").read_text()
    missing = [knob for knob in ("TOPIC_JUDGE_GPU", "TOPIC_JUDGE_MAX_LEN")
               if knob not in phase]
    if missing:
        raise RuntimeError(
            f"This image predates the judge-placement fix ({', '.join(missing)} absent from "
            f"role_phase.sh), so both judges would land on one card and training would die "
            f"with 'No available memory for the cache blocks'. Rebuild it: "
            f"bash docker/run_llama8b.sh build"
        )
    print(f"judge placement: WildGuard on GPU {env.get('JUDGE_GPU', '0')}, "
          f"topic judge on GPU {env.get('TOPIC_JUDGE_GPU', '(resolved at launch)')}", flush=True)


def training_command(args, env: dict[str, str]) -> list[str]:
    workspace = Path(env.get("SSP_ROOT", "/workspace"))
    if args.command == "role":
        env.update(ROLE=args.role, GEN=str(args.generation), STEPS=str(args.steps),
                   TAG=f"{'A' if args.role == 'attacker' else 'D'}{args.generation}",
                   SSP_OUT=env.get("SSP_OUT", str(workspace / "roles")))
        return ["bash", str(REPO / "scripts/slurm/role_phase.sh")]

    if not valid_run_name(args.run_suffix):
        raise ValueError("Set RUN_SUFFIX or --run-suffix to a run name without directory separators.")
    if args.command in {"uniform", "naive"}:
        return [
            sys.executable, str(REPO / "scripts/slurm/history_selfplay.py"),
            "--method", args.command, "--run-suffix", args.run_suffix,
            "--rounds", str(args.generations), "--steps", str(args.steps_per_role),
        ]
    env["STEPS_PER_ROLE"] = str(args.steps_per_role)
    env["MATRIX_EPISODES"] = str(args.matrix_episodes)
    return [
        sys.executable, str(REPO / "scripts/slurm/cold_psro_local.py"),
        "--workspace", str(workspace), "--work", env["SSP_WORK"],
        "--hf-home", env["HF_HOME"], "--sft-dir", env["SSP_SFT_DIR"],
        "--psro-root", env["PSRO_ROOT"], "--raw-payoff-root", env["RAW_PAYOFF_ROOT"],
        "--suffix", args.run_suffix, "--python", sys.executable,
        "--judge-model", env["JUDGE_MODEL_PATH"],
        "--gpus", "4", "--cell-workers", "2" if env.get("TOPIC_JUDGE_MANAGED") == "1" else "3", "--fresh-cold-run",
        "--generations", str(args.generations),
    ]


def record_run_settings(args, env: dict[str, str]) -> None:
    """Prevent accidental recipe changes when a container is recreated."""
    if args.command == "role":
        return
    workspace = Path(env.get("SSP_ROOT", "/workspace"))
    root = (Path(env["PSRO_ROOT"]) if args.command == "psro" else
            workspace / "selfplay_baselines" / args.command) / args.run_suffix
    defaults = {
        "INIT_KL_COEF": "0", "OFFTOPIC_ENABLE": "0", "PSRO_MIDRUN_RESUME": "0",
        "OFFTOPIC_PENALTY": "0.5", "ATTACKER_OBJECTIVE": "",
        "GAME_REWARD_MODE": "legacy", "ATTACKER_INVALID_MOVE": "1",
        "TRAINING_SEED": "8888", "PROMPT_MAX_LEN": "2048", "GENERATE_MAX_LEN": "2048",
        "MICRO_TRAIN": "8", "MICRO_ROLLOUT": "8", "TRAIN_BATCH": "32", "ROLLOUT_BATCH": "128",
        "SAVE_STEPS_CONTRACT": "10", "RACE_CELLS": "1", "RACE_FULL_EPISODES": "",
        "LADDER_EARLY_STOP": "1", "EARLY_STOP_WINDOW": "20",
        "ATK_MIN_DELTA": "0.01", "ATK_PATIENCE": "3", "ATK_MIN_STEP": "60",
        "ATK_CEILING_MAX": "0.30", "DEF_THRESHOLD": "0.01", "DEF_GUARD_MIN": "0.912",
        "DEF_PATIENCE": "3", "DEF_MIN_STEP": "60",
        "TOPIC_JUDGE_BACKEND": "local", "TOPIC_LOCAL_MODEL": "",
        "TOPIC_JUDGE_UTIL": "0.13", "TOPIC_JUDGE_GPU": "3", "TOPIC_JUDGE_MAX_LEN": "4096",
        "UTIL": "0.28", "OPP_UTIL": "0.26",
        "VLLM_SLEEP": "0",
    }
    recipe = {
        "method": args.command, "rounds": args.generations, "steps_per_role": args.steps_per_role,
        "base_model": BASE_MODEL, "judge_model": JUDGE_MODEL,
        "environment": {key: env.get(key, default) for key, default in defaults.items()},
    }
    from roll.utils.shared_game_reward import shared_game_reward_enabled, shared_game_reward_contract
    if shared_game_reward_enabled():
        recipe["shared_game_reward"] = shared_game_reward_contract()
    if args.command == "psro":
        recipe["matrix_episodes"] = args.matrix_episodes
    root.mkdir(parents=True, exist_ok=True)
    path = root / "docker_run_settings.json"
    if path.exists():
        if json.loads(path.read_text()) != recipe:
            raise ValueError(f"Training settings changed for {root}; use a new --run-suffix.")
    else:
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(recipe, indent=2) + "\n")
        temp.replace(path)


def lock_workspace(env: dict[str, str]):
    root = Path(env.get("SSP_ROOT", "/workspace"))
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".llama8b-training.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Another training job is using this workspace; run the three experiments sequentially.")
    # Keep the lock when exec replaces this entrypoint with the coordinator.
    os.set_inheritable(lock.fileno(), True)
    return lock


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="download base/judge weights into the mounted cache; no GPU needed")
    sub.add_parser("check", help="check four visible GPUs and cached model paths without starting training")
    for method, rounds in (("psro", 8), ("uniform", 8), ("naive", 6)):
        training = sub.add_parser(method, help=f"run/resume {method}, default {rounds} rounds")
        training.add_argument("--run-suffix", default=os.environ.get("RUN_SUFFIX"))
        training.add_argument("--rounds", "--generations", dest="generations", type=int,
                              default=int(os.environ.get("GENERATIONS", str(rounds))))
        budget = "300" if method in {"psro", "uniform"} else "100"
        training.add_argument("--steps-per-role", type=int, default=int(os.environ.get("STEPS_PER_ROLE", budget)))
        training.add_argument("--matrix-episodes", type=int, default=int(os.environ.get("MATRIX_EPISODES", "4000")))
    role = sub.add_parser("role", help="train one attacker or defender LoRA")
    role.add_argument("--role", choices=("attacker", "defender"), default="attacker")
    role.add_argument("--generation", type=int, default=1)
    role.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", "100")))
    args = parser.parse_args(argv)
    if args.command in {"psro", "uniform", "naive"}:
        if not valid_run_name(args.run_suffix):
            parser.error("Set --run-suffix to one directory name, e.g. psro8_run1")
        if args.generations < 1 or args.steps_per_role < 1:
            parser.error("rounds and steps-per-role must be positive")
        if args.command == "psro" and (args.matrix_episodes < 256 or args.matrix_episodes % 2):
            parser.error("matrix-episodes must be even and at least 256")
    if args.command == "role" and (args.generation < 1 or args.steps < 1):
        parser.error("generation and steps must be positive")
    env = environment()
    if args.command == "naive" and float(env["INIT_KL_COEF"]) > 0:
        parser.error("naive warm-start training requires INIT_KL_COEF=0")
    os.environ.update(env)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    if args.command != "prepare" and env.get("DRY_RUN") != "1":
        # Fail before allocating GPUs or loading weights when a required judge is missing.
        from roll.utils.shared_game_reward import shared_game_reward_enabled, shared_game_reward_contract
        if shared_game_reward_enabled():
            shared_game_reward_contract()
    env.update(model_paths(env, download=args.command == "prepare"))
    if args.command == "prepare":
        for name in ("BASE_MODEL_PATH", "JUDGE_MODEL_PATH"):
            print(f"{name}={env[name]}")
        if "TOPIC_JUDGE_MODEL" in env:
            print(f"TOPIC_JUDGE_MODEL={env['TOPIC_JUDGE_MODEL']}")
        return 0
    if env.get("DRY_RUN") != "1" or args.command == "check":
        check_gpus()
    if args.command == "check":
        report_build_provenance(env)
        print("Four GPUs and cached model paths found; no training started.")
        return 0
    command = training_command(args, env)
    # PSRO dry-run stops before the coordinator. Role dry-run also exercises
    # the existing shell launcher's full command assembly.
    if env.get("DRY_RUN") == "1" and args.command in {"psro", "uniform", "naive"}:
        import shlex
        print(f"rounds={args.generations}; steps-per-role={args.steps_per_role}; "
              f"early-stop={env['LADDER_EARLY_STOP']}")
        print(shlex.join(command))
        return 0
    lock = None
    if env.get("DRY_RUN") != "1":
        lock = lock_workspace(env)
        record_run_settings(args, env)
    print(f"Starting {args.command}: {BASE_MODEL}; reward mode={env['GAME_REWARD_MODE']}", flush=True)
    # The drivers resume from state.json, so a transient failure (a judge that
    # has not released its card yet, a node hiccup) is recoverable by simply
    # re-entering. Without this a container dies on the first one and, run
    # with --rm, takes its logs with it. Stopping the container still stops at
    # once: a signal raises SystemExit and is never retried.
    retries = args.command in {"psro", "uniform", "naive"} and env.get("DRY_RUN") != "1"
    return run_with_retries(command, env, retries)


def run_with_retries(command: list[str], env: dict[str, str], retries: bool) -> int:
    attempts = int(env.get("DRIVER_MAX_ATTEMPTS", "5")) if retries else 1
    wait = float(env.get("DRIVER_RETRY_WAIT", "60"))
    rc = 1
    for attempt in range(1, attempts + 1):
        if attempts > 1:
            print(f"=== driver attempt {attempt}/{attempts} ===", flush=True)
        rc = run_driver_once(command, env)
        if rc == 0:
            return 0
        if attempt < attempts:
            print(f"driver exited rc={rc}; resuming from saved state in {wait:.0f}s", flush=True)
            time.sleep(wait)
    return rc


def run_driver_once(command: list[str], env: dict[str, str]) -> int:
    if env.get("TOPIC_JUDGE_MANAGED") == "1":
        from roll.utils.local_topic_runtime import run_with_local_topic
        return run_with_local_topic(command, env, REPO)
    child = subprocess.Popen(command, env=env, start_new_session=True)

    def forward(signum, frame):
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass
        raise SystemExit(128 + signum)

    prior = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        return child.wait()
    finally:
        for sig, handler in prior.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

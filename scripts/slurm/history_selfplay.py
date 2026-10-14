#!/usr/bin/env python3
"""Run uniform-history or naive self-play using the existing role-LoRA trainer.

No Slurm or Modal service is required. One round produces A_i followed by D_i.
Uniform-history starts each oracle from base, matching cold PSRO, and samples
all available opposite-role policies (including A0/D0) equally per episode.
Naive inherits the previous same-role weights and faces only the newest opponent.
"""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cold_psro_common as common
from cold_psro_local import Config, early_stop_configuration, make_local_oracle_trainer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from roll.utils.history_selfplay import history_schedule
from roll.utils.shared_game_reward import shared_game_reward_enabled, shared_game_reward_contract
from roll.utils.topic_api import configure_local_topic_backend
from roll.utils.role_lora_cold_psro import opponent_pool


def run_baseline(psro: Any, cfg: Any, train: Any, *, method: str, rounds: int, steps: int) -> dict:
    schedule = history_schedule(method, rounds)
    if steps < 1:
        raise ValueError("steps must be positive")
    root = cfg.run_root
    root.mkdir(parents=True, exist_ok=True)
    path = root / "state.json"
    settings = {
        "lora_rank": 64, "lora_alpha": 64, "steps": steps,
        "sft_batches_per_step": 1,
        "save_steps": int(os.environ.get("SAVE_STEPS_CONTRACT", "10")),
        "actor_lr_scheduler": "constant_with_warmup", "lr_warmup_ratio": 0.05,
        "training_seed": int(os.environ.get("TRAINING_SEED", "8888")),
    }
    if settings["save_steps"] < 1:
        raise ValueError("SAVE_STEPS_CONTRACT must be positive")
    # Capture environment settings that affect this training path. A different
    # recipe gets a new run suffix; source-code hashes are not startup gates.
    recipe_defaults = {
        "INIT_KL_COEF": "0.01", "PSRO_MIDRUN_RESUME": "0",
        "OFFTOPIC_ENABLE": "1", "OFFTOPIC_PENALTY": "0.5",
        "ATTACKER_OBJECTIVE": "", "GENERATE_MAX_LEN": "2048", "PROMPT_MAX_LEN": "2048",
        "MICRO_TRAIN": "8", "MICRO_ROLLOUT": "8",
        "TRAIN_BATCH": "32", "ROLLOUT_BATCH": "128",
    }
    recipe = {key: os.environ.get(key, default) for key, default in recipe_defaults.items()}
    if shared_game_reward_enabled():
        recipe["shared_game_reward"] = shared_game_reward_contract()
    recipe["PSRO_MIDRUN_RESUME"] = os.environ.get(
        "PSRO_MIDRUN_RESUME", "0" if float(recipe["INIT_KL_COEF"] or "0.01") > 0 else "1"
    )
    if method == "naive" and float(recipe["INIT_KL_COEF"] or "0.01") > 0:
        raise ValueError("Naive warm-start training requires INIT_KL_COEF=0 in the current trainer.")
    contract = {
        "schema_version": "history-selfplay-v1", "method": method, "rounds": rounds,
        "generations": rounds,
        "base_model": psro.BASE_MODEL, "include_base_in_history": method == "uniform",
        "initialization": "cold" if method == "uniform" else "warm",
        "settings": settings, "recipe": recipe,
        "early_stopping": early_stop_configuration(),
        "attacker_learning_rate": 1e-5, "defender_learning_rate": 4e-5,
        "attacker_sft_cutoff": min(30, steps), "defender_sft_cutoff": min(10, steps),
        "matrix_or_nash_solver_during_training": False,
    }
    state = psro._read_json(path) if path.is_file() else {
        "contract": contract, "completed": False, "stage": "initialized",
        "population": {label: psro._base_population_identity(label) for label in ("A0", "D0")},
        "phases": {}, "evaluations": {},
    }
    if state.get("contract") != contract:
        raise ValueError(f"Run settings changed; use a different run suffix: {root}")
    for label, record in state["population"].items():
        if label in {"A0", "D0"}:
            if record != psro._base_population_identity(label):
                raise RuntimeError(f"base identity changed: {label}")
        else:
            psro._verify_population_identity(label, record, lora_rank=64, lora_alpha=64)
    psro._write_json_atomic(path, state)

    for phase in schedule:
        label, role = phase["target"], phase["role"]
        if label in state["population"]:
            continue
        pool = opponent_pool(phase["opponents"], phase["probabilities"], state["population"])
        spec = {**phase, "opponent_pool": pool}
        if label in state["phases"] and state["phases"][label] != spec:
            raise RuntimeError(f"Recorded opponents changed for {label}")
        state["phases"][label] = spec
        state["stage"] = f"training_{label}"
        psro._write_json_atomic(path, state)
        print(f"{method} {label}: start={phase['initialized_from']}, opponents={phase['opponents']}, "
              f"probabilities={phase['probabilities']}", flush=True)
        record = train(
            run_root=str(root), target=label, role=role, opponent_pool=pool,
            **settings,
            learning_rate=contract[f"{role}_learning_rate"],
            sft_stop_after_step=contract[f"{role}_sft_cutoff"],
            wandb_identity=f"role_lora_{method}__{cfg.suffix}__{label}",
        )
        state["population"][label] = record
        state["stage"] = f"completed_{label}"
        psro._write_json_atomic(path, state)
        psro._write_json_atomic(root / "checkpoint_inventory.json", state["population"])
    state.update(completed=True, stage="training_complete")
    psro._write_json_atomic(path, state)
    return state


def main(argv=None) -> int:
    # Before anything reads the reward contract; role_phase.sh only configures
    # the local judge later, inside the training phase.
    configure_local_topic_backend()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("uniform", "naive"), required=True)
    parser.add_argument("--run-suffix", required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args(argv)
    if args.steps is None:
        args.steps = 300 if args.method == "uniform" else 100
    if not args.run_suffix or Path(args.run_suffix).name != args.run_suffix or args.run_suffix in {".", ".."}:
        parser.error("run-suffix must be a single directory name")
    workspace = Path(os.environ.get("SSP_ROOT", "/workspace"))
    options = SimpleNamespace(
        workspace=workspace, work=Path(os.environ["SSP_WORK"]),
        hf_home=Path(os.environ["HF_HOME"]), sft_dir=Path(os.environ["SSP_SFT_DIR"]),
        psro_root=workspace / "selfplay_baselines" / args.method,
        raw_payoff_root=workspace / "psro/raw_payoff_v2", suffix=args.run_suffix,
        python=sys.executable, gpus=4, cell_workers=3, cell_judge_port=5310,
        cell_judge_util=0.45, train_judge_port=5311, ray_port=6311,
        judge_model=os.environ["JUDGE_MODEL_PATH"], coordinator=common.COLD_COORDINATOR,
        init_mode="cold" if args.method == "uniform" else "hot", run_official_evaluation=False,
    )
    cfg = Config(options)
    cfg.apply_to_process()
    psro, payoff = common.load_modules()
    common.retarget(psro, payoff, psro_root=cfg.psro_root,
                    raw_payoff_root=cfg.raw_payoff_root, upstream_source=cfg.work,
                    judge_url="http://127.0.0.1:5311/classify")
    train = make_local_oracle_trainer(psro, cfg)
    cfg.run_root.mkdir(parents=True, exist_ok=True)
    with (cfg.run_root / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another process is already training {cfg.run_root}")
        run_baseline(psro, cfg, train, method=args.method, rounds=args.rounds, steps=args.steps)
    print(f"Training complete: {cfg.run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

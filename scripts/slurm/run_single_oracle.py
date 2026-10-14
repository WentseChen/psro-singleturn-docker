#!/usr/bin/env python3
"""Train one PSRO oracle standalone, outside the coordinator's state machine.

Used for follow-up runs on a finished PSRO run: retraining an oracle that
collapsed, or continuing one for a larger step budget.  Every hyperparameter
and the frozen opponent mixture are copied verbatim from the source oracle's
manifest.json, so the only thing that changes is what the caller asks to
change (step count, warm start, checkpoint retention).

Output goes to its own directory tree; the source run is read-only here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cold_psro_common as common  # noqa: E402
from early_stop_monitor import EarlyStop, parse_line  # noqa: E402

ROLE_PHASE = Path(__file__).resolve().parent / "role_phase.sh"


def log(message: str) -> None:
    print(f"[oracle {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _supervise(environment, work, training_log, criterion, ckpt_dir,
               ckpt_wait, log_fn):
    """Run role_phase.sh, optionally stopping early on a met criterion.

    Returns (returncode, stopped_early, note).  The trainer runs in its own
    process group so the whole Ray/vLLM tree goes down with it.
    """
    with training_log.open("ab") as handle:
        proc = subprocess.Popen(
            ["bash", str(ROLE_PHASE)],
            cwd=str(work), env=environment,
            stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if criterion is None:
            return proc.wait(), False, ""

        patterns = criterion.patterns()
        offset = 0
        remainder = ""
        note = ""
        while True:
            code = proc.poll()
            try:
                with training_log.open("r", errors="ignore") as reader:
                    reader.seek(offset)
                    chunk = reader.read()
                    offset = reader.tell()
            except OSError:
                chunk = ""
            if chunk:
                remainder += chunk
                pieces = re.split(r"[\r\n]+", remainder)
                remainder = pieces.pop() if pieces else ""
                for piece in pieces:
                    parsed = parse_line(piece, patterns)
                    if parsed is None:
                        continue
                    measurement = criterion.observe(*parsed)
                    if measurement is not None:
                        log_fn(f"  early-stop check {measurement.describe()}")
                    if criterion.triggered or criterion.collapsed:
                        break
            if criterion.triggered or criterion.collapsed:
                step = criterion.measurements[-1].window_end
                log_fn(f"early stop: {criterion.reason}")
                target = ckpt_dir / f"global_step{step}_hf"
                deadline = time.time() + ckpt_wait
                while not target.is_dir() and time.time() < deadline:
                    if proc.poll() is not None:
                        break
                    time.sleep(5)
                if target.is_dir():
                    log_fn(f"  triggering checkpoint present: {target}")
                else:
                    log_fn(f"  WARNING: {target} did not appear within "
                           f"{ckpt_wait}s; stopping anyway")
                note = criterion.reason
                if proc.poll() is None:
                    log_fn("  terminating trainer process group")
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        log_fn("  trainer did not exit; sending SIGKILL")
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.wait()
                return 0, True, note
            if code is not None:
                return code, False, note
            time.sleep(5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path,
                        help="finished run root holding training/<source>/manifest.json")
    parser.add_argument("--source-oracle", required=True,
                        help="oracle whose manifest supplies hyperparameters and mixture")
    parser.add_argument("--tag", required=True, help="output directory name / W&B suffix")
    parser.add_argument("--out-root", required=True, type=Path,
                        help="directory that will hold <tag>/")
    parser.add_argument("--steps", required=True, type=int,
                        help="optimizer steps THIS run performs")
    parser.add_argument("--start-adapter", default="",
                        help="warm-start adapter; empty means cold start from base")
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--max-ckpt-num", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--judge-port", type=int, default=5321)
    parser.add_argument("--ray-port", type=int, default=6321)
    parser.add_argument("--workspace", type=Path,
                        default=Path("/zfsauton/scratch/wentsec/coldpsro_slurm"))
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--hf-home", type=Path,
                        default=Path("/zfsauton/scratch/wentsec/hf_srt"))
    parser.add_argument("--sft-dir", type=Path,
                        default=Path("/zfsauton/scratch/wentsec/sft_data"))
    parser.add_argument("--python", default="/zfsauton/scratch/wentsec/envs/srt/bin/python")
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--micro-train", type=int, default=8,
                        help="micro train batch; lower trades throughput for memory "
                             "without changing the effective train_batch_size")
    parser.add_argument("--micro-rollout", type=int, default=8)
    parser.add_argument("--generate-max-len", type=int, default=0,
                        help="defender/attacker generation cap; 0 keeps role_phase.sh's default")
    parser.add_argument("--prompt-max-len", type=int, default=0,
                        help="prompt cap; 0 keeps role_phase.sh's default")
    parser.add_argument("--attacker-objective", default="",
                        help="training attacker objective; empty keeps the upstream default")
    parser.add_argument("--early-stop", action="store_true",
                        help="stop once the primary metric holds below threshold")
    parser.add_argument("--early-stop-metric",
                        default="attacker/info/successful_harmful_request_rate")
    parser.add_argument("--early-stop-threshold", type=float, default=0.02)
    parser.add_argument("--early-stop-guard-metric",
                        default="defender/wildguard_actual_benign_compliance_acc")
    parser.add_argument("--early-stop-guard-min", type=float, default=0.912)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-window", type=int, default=0,
                        help="steps per measurement; 0 means use --save-steps")
    parser.add_argument("--early-stop-min-step", type=int, default=30)
    parser.add_argument("--early-stop-ceiling-metric",
                        default="defender/cot_format_violation",
                        help="metric that must stay BELOW --early-stop-ceiling-max "
                             "for a window to count as success")
    parser.add_argument("--early-stop-ceiling-max", type=float, default=0.10)
    parser.add_argument("--early-stop-abort-max", type=float, default=0.80,
                        help="ceiling-metric level treated as model collapse")
    parser.add_argument("--early-stop-abort-patience", type=int, default=2)
    parser.add_argument("--early-stop-ckpt-wait", type=int, default=900,
                        help="seconds to wait for the triggering checkpoint before stopping")
    args = parser.parse_args()

    work = args.work or (args.workspace / "work")
    manifest_path = args.source_run / "training" / args.source_oracle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool = manifest["fixed_opponent_pool"]
    role = manifest["role"]

    # Every frozen opponent adapter must still be on disk and unchanged; the
    # coordinator normally re-verifies this and a silent swap would invalidate
    # the comparison against the original oracle.
    for entry in pool:
        adapter = entry.get("adapter")
        if adapter and not Path(adapter).is_dir():
            raise SystemExit(f"missing frozen opponent adapter: {adapter}")

    out_dir = args.out_root / args.tag
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing {out_dir}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.start_adapter and not Path(args.start_adapter).is_dir():
        raise SystemExit(f"missing start adapter: {args.start_adapter}")

    environment = dict(os.environ)
    environment.pop("TRANSFORMERS_CACHE", None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update({
        "HF_HOME": str(args.hf_home),
        "HF_HUB_CACHE": str(args.hf_home / "hub"),
        "HF_HUB_DISABLE_XET": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "NCCL_CUMEM_ENABLE": "0",
        # NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here.
        # Expandable segments cannot be shared over CUDA IPC, and the LoRA
        # weight broadcast to vLLM (update_lora_weight_cuda_ipc) then dies with
        # "pidfd_getfd: Operation not permitted".
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "PATH": ":".join(p for p in (str(Path(args.python).parent),
                                     os.environ.get("PATH", "")) if p),
        "PYTHONPATH": ":".join(p for p in (str(work), str(common.REPO_ROOT),
                                           os.environ.get("PYTHONPATH", "")) if p),
        "SSP_ROOT": str(args.workspace),
        "SSP_WORK": str(work),
        "SSP_HF_HOME": str(args.hf_home),
        "SSP_SFT_DIR": str(args.sft_dir),
        "SSP_OUT": str(args.out_root),
        "SSP_PYTHON": args.python,
        "ROLE": role,
        "TAG": args.tag,
        "GEN": str(args.source_oracle[1:]),
        "STEPS": str(args.steps),
        "GPUS": str(args.gpus),
        "LORA_RANK": str(manifest["lora_rank"]),
        "LORA_ALPHA": str(manifest["lora_alpha"]),
        "LR": repr(float(manifest["learning_rate"])),
        "SFT_STOP": str(manifest["sft_stop_after_step"]),
        "SFT_BATCHES_PER_STEP": str(manifest["sft_batches_per_step"]),
        "SAVE_STEPS": str(args.save_steps),
        "MAX_CKPT_NUM": str(args.max_ckpt_num),
        "ACTOR_LR_SCHEDULER": manifest["actor_lr_scheduler"],
        "LR_WARMUP_RATIO": repr(float(manifest["lr_warmup_ratio"])),
        "TRAINING_SEED": str(manifest["seed"]),
        "OPPONENT_POOL_JSON": json.dumps(pool, sort_keys=True),
        "EXACT_LABEL_BALANCE": "1",
        "LABEL_DRIFT_POLICY": manifest["label_drift_training_policy"],
        "EVAL_STEPS": "10",
        "EVAL_START_STEPS": "50",
        "DIVERSITY_STEPS": "5",
        "WANDB_RUN_NAME": args.wandb_run_name,
        "JUDGE_PORT": str(args.judge_port),
        "RAY_PORT": str(args.ray_port),
        "VLLM_SLEEP": "0",
        "DS_SLEEP": "0",
        "EAGER": "0",
        "MICRO_ROLLOUT": str(args.micro_rollout),
        "MICRO_TRAIN": str(args.micro_train),
    })
    if args.start_adapter:
        environment["START_ADAPTER"] = args.start_adapter
    if args.attacker_objective:
        environment["ATTACKER_OBJECTIVE"] = args.attacker_objective
    if args.generate_max_len:
        environment["GENERATE_MAX_LEN"] = str(args.generate_max_len)
    if args.prompt_max_len:
        environment["PROMPT_MAX_LEN"] = str(args.prompt_max_len)

    mixture = ", ".join(
        f"{e['id']}={float(e['probability']):.4f}"
        for e in pool if float(e["probability"]) > 0.005
    )
    start = args.start_adapter or "base (cold)"
    log(f"training {args.tag} ({role}) from {start}")
    log(f"  steps={args.steps} save_steps={args.save_steps} "
        f"max_ckpt_num={args.max_ckpt_num}")
    log(f"  frozen mixture (from {args.source_oracle}): {mixture}")
    log(f"  out: {out_dir}")
    log(f"  wandb run name: {args.wandb_run_name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    training_log = out_dir / "training.log"

    criterion = None
    if args.early_stop:
        criterion = EarlyStop(
            metric=args.early_stop_metric,
            guard_metric=args.early_stop_guard_metric,
            threshold=args.early_stop_threshold,
            guard_min=args.early_stop_guard_min,
            window=args.early_stop_window or args.save_steps,
            patience=args.early_stop_patience,
            min_step=args.early_stop_min_step,
            ceiling_metric=args.early_stop_ceiling_metric,
            ceiling_max=args.early_stop_ceiling_max,
            abort_max=args.early_stop_abort_max,
            abort_patience=args.early_stop_abort_patience,
        )
        log(f"  early stop: {criterion.metric} <= {criterion.threshold} on "
            f"{criterion.patience} consecutive {criterion.window}-step windows, "
            f"guarded by {criterion.guard_metric} >= {criterion.guard_min}"
            + (f" and {criterion.ceiling_metric} <= {criterion.ceiling_max}"
               if criterion.ceiling_metric else ""))
        if criterion.ceiling_metric:
            log(f"  collapse abort: {criterion.ceiling_metric} >= "
                f"{criterion.abort_max} on {criterion.abort_patience} "
                f"consecutive windows")

    returncode, stopped_early, stop_note = _supervise(
        environment, work, training_log, criterion, out_dir / "ckpt",
        args.early_stop_ckpt_wait, log,
    )
    if returncode != 0:
        raise SystemExit(f"{args.tag} training failed rc={returncode}; "
                         f"see {training_log}")

    # Same hook as the ladder trainer: a standalone oracle is a role training
    # run too. Failures are reported and swallowed inside the publisher.
    from roll.utils.checkpoint_publish import publish_role_checkpoint

    final_checkpoint = max(
        (d for d in (out_dir / "ckpt").glob("global_step*_hf") if d.is_dir()),
        key=lambda d: int(d.name.removeprefix("global_step").removesuffix("_hf")),
        default=None,
    )
    if final_checkpoint is not None:
        publish_role_checkpoint(
            final_checkpoint, run_suffix=args.tag, target=args.source_oracle,
            role=role, log=log,
            extra={"early_stop": stop_note, "steps_this_run": args.steps,
                   "reward_mode": environment.get("GAME_REWARD_MODE", "legacy"),
                   "topic_penalty": environment.get("TOPIC_PENALTY")},
        )

    provenance = {
        "method": "standalone PSRO oracle follow-up run",
        "source_run": str(args.source_run),
        "source_oracle": args.source_oracle,
        "role": role,
        "steps_this_run": args.steps,
        "start_adapter": args.start_adapter or None,
        "base_model": manifest["base_model"],
        "fixed_opponent_pool": pool,
        "save_steps": args.save_steps,
        "max_ckpt_num": args.max_ckpt_num,
        "wandb_run_name": args.wandb_run_name,
        "attacker_objective": args.attacker_objective or None,
        "generate_max_len": args.generate_max_len or 2048,
        "micro_train": args.micro_train,
        "micro_rollout": args.micro_rollout,
        "prompt_max_len": args.prompt_max_len or 2048,
        "early_stop": None if criterion is None else {
            "metric": criterion.metric,
            "threshold": criterion.threshold,
            "guard_metric": criterion.guard_metric,
            "guard_min": criterion.guard_min,
            "window": criterion.window,
            "patience": criterion.patience,
            "min_step": criterion.min_step,
            "ceiling_metric": criterion.ceiling_metric or None,
            "ceiling_max": criterion.ceiling_max,
            "abort_max": criterion.abort_max,
            "triggered": criterion.triggered,
            "collapsed": criterion.collapsed,
            "reason": criterion.reason or None,
            "stopped_at_step": (criterion.measurements[-1].window_end
                                if (criterion.triggered or criterion.collapsed)
                                else None),
            "measurements": [
                {"step": m.window_end, "primary": m.primary, "guard": m.guard,
                 "ceiling": m.ceiling, "primary_ok": m.primary_ok,
                 "guard_ok": m.guard_ok, "ceiling_ok": m.ceiling_ok}
                for m in criterion.measurements
            ],
        },
        "stopped_early": stopped_early,
        "stop_note": stop_note or None,
        "inherited_from_manifest": {
            k: manifest[k] for k in (
                "learning_rate", "lora_rank", "lora_alpha",
                "sft_stop_after_step", "sft_batches_per_step",
                "actor_lr_scheduler", "lr_warmup_ratio", "seed",
                "label_drift_training_policy",
            )
        },
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"{args.tag} done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

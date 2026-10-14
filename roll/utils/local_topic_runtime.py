"""Keep one local label service alive across every oracle and PSRO cell."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request


def stop_process_group(process) -> None:
    if process is None:
        return
    # The parent can have exited while vLLM/Ray descendants are still alive.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=15)


def wait_for_topic_service(process, env: dict[str, str], log_path: Path) -> None:
    attempts = int(env.get("TOPIC_JUDGE_STARTUP_ATTEMPTS", "90"))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(attempts):
        if process.poll() is not None:
            raise RuntimeError(f"Local topic judge exited during startup; see {log_path}")
        try:
            with opener.open(env["TOPIC_API_BASE_URL"].removesuffix("/v1") + "/health", timeout=3) as response:
                status = json.load(response)
            if isinstance(status, dict) and status.get("ok") is True and status.get("model") == env["TOPIC_API_MODEL"]:
                return
        except (OSError, ValueError):
            pass
        time.sleep(5)
    raise RuntimeError(f"Local topic judge did not become ready; see {log_path}")


@contextlib.contextmanager
def local_topic_service(env: dict[str, str], repo: Path):
    """Own one local judge for the whole run, then hand it to every child.

    Training phases start and stop their own judge, but PSRO cells do not: a
    payoff cell worker just calls the topic API and dies with "Topic API
    request failed or timed out" if nothing is listening. The container solved
    this by wrapping the coordinator in run_with_local_topic; the bare
    scripts/slurm path had no equivalent, so the matrix stage could not run at
    all with the local backend.

    Sets TOPIC_JUDGE_MANAGED=1 so role_phase.sh reuses this service instead of
    starting a second engine on the same card.
    """
    port = int(env["TOPIC_JUDGE_PORT"])
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
    except OSError as exc:
        raise RuntimeError(
            f"Topic judge port {port} is already in use ({exc.strerror}); stop the stale "
            f"judge or set TOPIC_JUDGE_PORT."
        ) from exc
    log_path = Path(env["SSP_ROOT"]) / "topic_judge.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    service = None
    try:
        with log_path.open("ab") as log:
            service_env = dict(env, CUDA_VISIBLE_DEVICES=env.get("TOPIC_JUDGE_GPU", "3"),
                               VLLM_USE_V1="0", TOPIC_JUDGE_NAME=env["TOPIC_API_MODEL"])
            service = subprocess.Popen(
                [env["SSP_PYTHON"], str(repo / "scripts/slurm/serve_topic_judge.py")],
                env=service_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            wait_for_topic_service(service, env, log_path)
            env["TOPIC_JUDGE_MANAGED"] = "1"
            os.environ["TOPIC_JUDGE_MANAGED"] = "1"
            yield service
    finally:
        os.environ.pop("TOPIC_JUDGE_MANAGED", None)
        stop_process_group(service)


def run_with_local_topic(command: list[str], env: dict[str, str], repo: Path) -> int:
    """Own the judge and trainer process groups, including failure and shutdown."""
    port = int(env["TOPIC_JUDGE_PORT"])
    # Probing first is deliberate: starting a second judge on an occupied port
    # would silently hand the run whatever already answers there, and a stale
    # judge from a crashed run answers the same contract with different weights.
    # The bare OSError this used to raise ("[Errno 98] Address already in use")
    # says nothing about which port, what is holding it, or how to proceed --
    # and the container always uses the same default, so the first thing anyone
    # hits after a crashed run is exactly this.
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
    except OSError as exc:
        raise RuntimeError(
            f"Topic judge port {port} is already in use ({exc.strerror}). A judge from an "
            f"earlier run is probably still holding it; stop it, or set TOPIC_JUDGE_PORT to "
            f"a free port. Never point this at a service you did not start: the label model "
            f"is part of the run contract."
        ) from exc
    log_path = Path(env["SSP_ROOT"]) / "topic_judge.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    service = trainer = None
    prior_handlers = {}
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            prior_handlers[sig] = signal.signal(sig, interrupted)
        with log_path.open("ab") as log:
            # Honour TOPIC_JUDGE_GPU: hard-coding 3 here made the knob a lie in
            # the managed path, and the whole point of it is that this engine
            # must not land on a card another engine already holds.
            service_env = dict(env, CUDA_VISIBLE_DEVICES=env.get("TOPIC_JUDGE_GPU", "3"),
                               VLLM_USE_V1="0", TOPIC_JUDGE_NAME=env["TOPIC_API_MODEL"])
            service = subprocess.Popen(
                [env["SSP_PYTHON"], str(repo / "scripts/slurm/serve_topic_judge.py")],
                env=service_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            wait_for_topic_service(service, env, log_path)
            print(f"Local topic judge ready on GPU {env.get('TOPIC_JUDGE_GPU', '3')}: {env['TOPIC_API_MODEL']}; log: {log_path}", flush=True)
            trainer = subprocess.Popen(command, env=env, start_new_session=True)
            while trainer.poll() is None:
                if service.poll() is not None:
                    raise RuntimeError(f"Local topic judge exited while training; see {log_path}")
                time.sleep(1)
            return trainer.returncode
    finally:
        try:
            stop_process_group(trainer)
        finally:
            stop_process_group(service)
            for sig, handler in prior_handlers.items():
                signal.signal(sig, handler)
